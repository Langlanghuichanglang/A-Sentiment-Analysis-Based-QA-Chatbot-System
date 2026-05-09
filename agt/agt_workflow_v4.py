import os
import uuid
import json
from datetime import datetime

from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from langgraph.store.base import BaseStore

from langgraph.prebuilt import ToolNode

# from langchain_community.tools import DuckDuckGoSearchRun


SESSION_SUMMARY_EVERY = 5  # 每隔多少轮触发一次 session 摘要
PROFILE_UPDATE_EVERY = 3  # 每隔多少轮触发一次 profile 更新
RECENT_MESSAGES_LIMIT = 5


# tools
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    # mock
    fake_data = {
        "北京": "晴天，25°C，空气质量良好",
        "上海": "多云，22°C，有轻微霾",
        "深圳": "雷阵雨，28°C，湿度较高",
    }
    return fake_data.get(city, f"{city}：数据暂无")  # get(key, default) 如果 key 不存在，返回 default


@tool
def get_mbti_score(text: str) -> dict:
    """分析文本的 MBTI 人格类型倾向"""
    # mock
    return {
        "I": 0.7,  # 内向
        "E": 0.3,  # 外向
        "S": 0.4,  # 实感
        "N": 0.6,  # 直觉
        "T": 0.5,  # 思考
        "F": 0.5,  # 情感
        "J": 0.8,  # 判断
        "P": 0.2,  # 知觉
    }


# search = TavilySearchResults(max_results=3)

tools = [get_weather, get_mbti_score]

# llm
_llm_base = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)
model = _llm_base.bind_tools(tools)  # 绑定工具的模型，用于正常对话和工具调用
llm_plain = _llm_base  # 不绑定工具的模型，用于画像更新时


# memory
## pydantic 数据模型

class MBTIScores(BaseModel):
    """MBTI 八个维度的概率评分，每个值在 0~1 之间"""
    I: float = Field(default=0.5, ge=0.0, le=1.0)  # 内向
    E: float = Field(default=0.5, ge=0.0, le=1.0)  # 外向
    N: float = Field(default=0.5, ge=0.0, le=1.0)  # 直觉
    S: float = Field(default=0.5, ge=0.0, le=1.0)  # 实感
    T: float = Field(default=0.5, ge=0.0, le=1.0)  # 思考
    F: float = Field(default=0.5, ge=0.0, le=1.0)  # 情感
    J: float = Field(default=0.5, ge=0.0, le=1.0)  # 判断
    P: float = Field(default=0.5, ge=0.0, le=1.0)  # 知觉

    def dominant_type(self) -> str:
        """根据当前评分推断 MBTI 类型"""
        ei = "I" if self.I > self.E else "E"
        ns = "N" if self.N > self.S else "S"
        tf = "T" if self.T > self.F else "F"
        jp = "J" if self.J > self.P else "P"
        return f"{ei}{ns}{tf}{jp}"

    def confidence(self) -> float:
        """
        整体置信度：各维度偏离 0.5 的平均程度。
        全部 0.5 → confidence=0（完全不确定）
        某维度 0.9 → 该维度贡献 0.4/0.5=0.8 的确定性
        """
        dims = [
            abs(self.I - 0.5),
            abs(self.N - 0.5),
            abs(self.T - 0.5),
            abs(self.J - 0.5),
        ]
        return round(sum(dims) / (len(dims) * 0.5), 2)

    def partial_merge(self, updates: dict[str, float], weight: float = 0.25) -> "MBTIScores":
        """
        接受部分字段更新（如 LLM 只返回 {"I": 0.72, "N": 0.65}）
        未提供的字段保持原值
        """
        if not updates or not isinstance(updates, dict):
            return self.model_copy()

        # 当前最新值
        current = self.model_dump()

        def weighted(old: float, new_val: float | None) -> float:
            if new_val is None:  # 缺少的字段会是None
                return old
            return round(old * (1 - weight) + new_val * weight, 3)

        return MBTIScores(
            I=weighted(current["I"], updates.get("I")),
            E=weighted(current["E"], updates.get("E")),
            N=weighted(current["N"], updates.get("N")),
            S=weighted(current["S"], updates.get("S")),
            T=weighted(current["T"], updates.get("T")),
            F=weighted(current["F"], updates.get("F")),
            J=weighted(current["J"], updates.get("J")),
            P=weighted(current["P"], updates.get("P")),
        )


class UserProfile(BaseModel):
    """跨会话用户画像，存入 ("data", user_id) namespace 的 profile key"""
    user_id: str
    text_summary: str = ""
    mbti_scores: MBTIScores = Field(default_factory=MBTIScores)
    traits: list[str] = Field(default_factory=list)  # 观察到的性格特质
    conversation_count: int = 0  # 累计对话轮次
    last_updated: str = ""  # ISO 时间戳

    def to_store_value(self) -> dict:
        """存入 Store 时序列化"""
        return self.model_dump()

    @classmethod
    def from_store_value(cls, data: dict) -> "UserProfile":
        """从 Store 读出时反序列化"""
        return cls.model_validate(data)


def _get_profile(store: BaseStore, user_id: str) -> UserProfile:
    namespace = ("data", user_id)
    item = store.get(namespace, "profile")
    if item:
        return UserProfile.from_store_value(item.value)
    return UserProfile(user_id=user_id)


def _save_profile(store: BaseStore, profile: UserProfile):
    profile.last_updated = datetime.now().isoformat()
    namespace = ("data", profile.user_id)
    store.put(namespace, "profile", profile.to_store_value())


def _get_all_facts(store: BaseStore, user_id: str) -> dict:
    """返回 {fact_key: fact_value_dict} 的字典"""
    namespace = ("data", user_id)
    item = store.get(namespace, "facts")
    if item and isinstance(item.value, dict):
        return item.value
    return {}


def _save_facts(store: BaseStore, user_id: str, facts: dict):
    namespace = ("data", user_id)
    store.put(namespace, "facts", facts)


def _get_summaries(store: BaseStore, user_id: str) -> dict:
    namespace = ("data", user_id)
    item = store.get(namespace, "summaries")
    if item and isinstance(item.value, dict):
        return item.value
    return {"sessions": {}}


def _save_summaries(store: BaseStore, user_id: str, summaries: dict):
    namespace = ("data", user_id)
    store.put(namespace, "summaries", summaries)


def _get_latest_summary(store: BaseStore, user_id: str) -> str:
    """优先取 weekly 压缩摘要，没有就返回空"""
    summaries = _get_summaries(store, user_id)
    weekly = summaries.get("weekly_latest", {})
    return weekly.get("text", "")


def _extract_dialogue(messages: list[BaseMessage]) -> str:
    """从消息列表提取人类和 AI 的对话文本，过滤 system/tool"""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"AI：{m.content}")
    return "\n".join(lines)


def _build_system_prompt(
        profile: UserProfile,
        facts: dict,
        # weekly_summary: str,
) -> str:
    parts = [
        "你是一个擅长观察人格的助理，同时也能回答日常问题。",
        "你会在对话中自然地观察用户的性格倾向（MBTI 维度：E/I、S/N、T/F、J/P），持续更新对用户的认知画像，但不要直接给用户贴标签，除非用户主动询问。",

        "你的核心价值是帮助用户进行思维扩充：",
        "1. 识别用户当前的思考风格和可能存在的认知盲区后，主动提供互补视角（尤其是用户较弱的维度）。",
        "2. 对于重要决策、问题分析、创意生成等，主动给出『用户当前倾向视角』 + 『互补视角』的对比分析，帮助用户看到更完整的光谱。",
        "3. 不要模仿用户的思考风格，而是刻意引入差异化思考（例如用户偏直觉时，你补充具体事实与执行细节；用户偏理性时，你补充人际与情感影响）。",
        "4. 定期或在合适时机，帮助用户反思：『这个结论是否受到了某种性格偏好的影响？还有其他角度吗？』",

        "回答时保持自然、专业且有洞察力。不要每句都提性格，只在有明显价值时自然融入。",
        "始终以『帮助用户更好地思考和决策』为目标，而非单纯诊断人格。",
    ]

    mbti_type = profile.mbti_scores.dominant_type()
    conf = profile.mbti_scores.confidence()
    parts.append(
        f"\n【当前 MBTI 推断】{mbti_type}（置信度 {conf:.0%}）\n"
        f"细项评分：{json.dumps(profile.mbti_scores.model_dump(), ensure_ascii=False)}"
    )

    if profile.traits:
        parts.append(f"\n【已观察到的性格特质】\n" + "、".join(profile.traits))

    if facts:
        facts_readable = "\n".join(
            f"  · {k}：{v.get('value', v)}"
            for k, v in facts.items()
        )
        parts.append(f"\n【已知用户信息】\n{facts_readable}")

    # if weekly_summary:
    #     parts.append(f"\n【历史对话摘要】\n{weekly_summary}")

    # parts.append(f"\n已累计对话 {profile.conversation_count} 轮。")
    return "\n".join(parts)


def add_messages_dedupe_system(
        existing: list[BaseMessage],
        new: list[BaseMessage],
) -> list[BaseMessage]:
    merged = add_messages(existing, new)
    last_system: SystemMessage | None = None
    non_system: list[BaseMessage] = []

    for message in merged:
        if isinstance(message, SystemMessage):
            last_system = message
        else:
            non_system.append(message)

    return ([last_system] if last_system else []) + non_system


# graph
## state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages_dedupe_system]
    user_id: str
    session_id: str
    turn_count: int
    session_summary: str  # session 长期记忆


## nodes

def load_profile(state: AgentState, *, store: BaseStore) -> AgentState:
    """
    对话开始前触发。
    从 Store 读取三类记忆，组装 system prompt 注入消息列表。
    """
    user_id = state["user_id"]

    profile = _get_profile(store, user_id)
    facts = _get_all_facts(store, user_id)
    # weekly_summary = _get_latest_summary(store, user_id)

    system_content = _build_system_prompt(profile, facts)

    return {
        "messages": [SystemMessage(content=system_content)]
        # 每次注入 system 时，reducer 只保留最新一条 system
    }


def reflection(state: AgentState, *, store: BaseStore) -> AgentState:
    user_id = state["user_id"]
    session_id = state["session_id"]
    turn_count = state.get("turn_count", 0)
    session_summary = state.get("session_summary", "")

    dialogue = _extract_dialogue(state["messages"])
    if not dialogue.strip():
        return {}

    updates: dict = {}

    # update
    # facts
    _maybe_update_facts(store, user_id, dialogue)

    # profile
    profile = _get_profile(store, user_id)
    if turn_count % PROFILE_UPDATE_EVERY == 0:
        _update_profile(store, user_id, profile, dialogue)
        profile = _get_profile(store, user_id)

    profile.conversation_count += 1
    profile.last_updated = datetime.now().isoformat()
    _save_profile(store, profile)

    # session summary
    new_turn_count = turn_count + 1
    print(f"\n[memory] turn_count={new_turn_count}")
    if new_turn_count % SESSION_SUMMARY_EVERY == 0:
        new_summary = _summarize_session(dialogue, session_summary)
        updates["session_summary"] = new_summary
        print(f"  [memory] session_summary 已更新")

    updates["turn_count"] = new_turn_count
    return updates


def _maybe_update_facts(store: BaseStore, user_id: str, dialogue: str):
    """
    让 LLM 从对话中提取客观事实。
    - 没有新事实 → 返回 {}，不写 Store
    - 已有同 key 且置信度更高 → 跳过
    - 新事实置信度足够 → 写入对应 key
    """
    existing = _get_all_facts(store, user_id)
    existing_readable = json.dumps(
        {k: v.get("value", v) for k, v in existing.items()},
        ensure_ascii=False
    )

    prompt = f"""从以下对话中提取用户的客观事实信息（职业、爱好、居住地、家庭状况等）。

规则：
1. 只提取用户明确说出的信息，不要推断
2. 如果对话没有新的事实信息，返回 {{}}
3. 不要重复已有事实（除非用户纠正了）

已知事实（不要重复）：
{existing_readable}

对话内容：
{dialogue}

以 JSON 格式返回，例如：
{{"occupation": {{"value": "设计师", "confidence": 0.95}}, "city": {{"value": "上海", "confidence": 0.9}}}}
只输出 JSON，没有新事实就输出 {{}}。"""

    try:
        raw = llm_plain.invoke(prompt).content.strip().strip("```json").strip("```").strip()
        new_facts: dict = json.loads(raw)

        updated = False
        for key, fact_data in new_facts.items():
            if not isinstance(fact_data, dict):
                continue
            new_conf = fact_data.get("confidence", 0.5)

            # 已有同 key 时，只在新置信度更高时覆盖
            if key in existing:
                old_conf = existing[key].get("confidence", 0.5)
                if new_conf <= old_conf:
                    continue

            existing[key] = {
                "value": fact_data.get("value", ""),
                "confidence": new_conf,
                "source_session": datetime.now().isoformat()[:10],
            }
            updated = True
            print(f"  [facts] 写入：{key} = {fact_data.get('value')}")

        if updated:
            _save_facts(store, user_id, existing)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [facts] 提取失败，跳过：{e}")


def _update_profile(
        store: BaseStore, user_id: str, profile: UserProfile, dialogue: str
):
    """
    让 LLM 根据对话更新画像摘要、MBTI 评分和性格特质。
    只更新有变化的维度，避免无意义的噪声写入。
    """
    facts = _get_all_facts(store, user_id)
    facts_text = json.dumps(
        {k: v.get("value", v) for k, v in facts.items()},
        ensure_ascii=False
    )

    prompt = f"""
根据以下对话更新用户画像：

已有摘要：{profile.text_summary or '无'}
已知用户信息：{facts_text}
当前 MBTI 评分：{json.dumps(profile.mbti_scores.model_dump(), ensure_ascii=False)}
本轮对话：
{dialogue}

你需要做两件事：
1) 画像摘要/特质：若有新的有效信息，才返回对应字段；否则不要返回该字段。
2) MBTI：判断本轮是否提供了新的倾向线索。若有线索，只需返回有变化的维度；没有线索就不要返回 mbti_update字段。

输出 JSON 示例：
{{
    "text_summary": "100字以内的用户画像摘要",
    "new_traits": ["新观察到的特质1", "特质2", "倾向独立思考", "表达较为简洁"],
    "mbti_update": {{"I": 0.0-1.0, "E": 0.0-1.0, "N": 0.0-1.0, "S": 0.0-1.0, "T": 0.0-1.0, "F": 0.0-1.0, "J": 0.0-1.0, "P": 0.0-1.0}}
}}
或：
{{
    "mbti_update": {{"I": 0.0-1.0, "E": 0.0-1.0}}
}}
只输出 JSON。"""

    try:
        raw = llm_plain.invoke(prompt).content.strip().strip("```json").strip("```").strip()
        updates: dict = json.loads(raw)

        summary = updates.get("text_summary")
        new_traits = updates.get("new_traits")
        mbti_update = updates.get("mbti_update")

        if isinstance(summary, str) and summary.strip():
            profile.text_summary = summary.strip()
            print("  [profile] 摘要已更新")

        if isinstance(new_traits, list) and new_traits:
            profile.traits = list(set(profile.traits + new_traits))
            print(f"  [profile] 新增特质：{new_traits}")

        if isinstance(mbti_update, dict) and mbti_update:
            profile.mbti_scores = profile.mbti_scores.partial_merge(mbti_update, weight=0.25)
            print(
                f"  [profile] MBTI 更新 → {profile.mbti_scores.dominant_type()} "
                f"(置信度 {profile.mbti_scores.confidence():.0%})"
            )
            print(f"  [profile] 更新的维度: {list(mbti_update.keys())}")
        else:
            print("  [profile] 本轮无 MBTI 线索，保持不变")

        _save_profile(store, profile)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [profile] 更新失败，保留旧数据：{e}")


def _summarize_session(dialogue: str, existing_summary: str) -> str:
    """
    将本轮对话压缩进滚动摘要。
    existing_summary 是上一次的摘要（可能为空）。
    返回更新后的摘要字符串。
    """
    prompt = f"""请将【本轮对话】压缩进【已有摘要】，生成一段不超过200字的新摘要。
要求：
1. 保留重要的讨论主题、结论、用户表达的观点
2. 去掉寒暄、重复、无实质内容的部分
3. 只输出摘要文本，不要任何前缀或解释

已有摘要：
{existing_summary or '（本 session 首次生成摘要）'}

本轮对话：
{dialogue}

输出新摘要："""

    try:
        result = llm_plain.invoke(prompt).content.strip()
        print(f"  [session_summary] 摘要已更新（{len(result)}字）")
        return result
    except Exception as e:
        print(f"  [session_summary] 摘要更新失败，保留旧摘要：{e}")
        return existing_summary


def call_model_0(state: AgentState) -> AgentState:
    """
    调用模型，得到回复后更新状态
    输入：当前状态（包含对话历史）
    输出：更新后的状态（包含新回复）
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # 返回的状态只包含新消息，add_messages 把它附加到现有历史中


def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    session_summary = state.get("session_summary", "")

    # 分离 system 和对话消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    # 只取最近 N 条对话
    recent = non_system[-RECENT_MESSAGES_LIMIT:]

    # 把 session_summary 追加进 system prompt（不影响 Store 里的数据）
    if system_msgs and session_summary:
        base_sys = system_msgs[-1]
        enhanced_sys = SystemMessage(
            content=base_sys.content + f"\n\n【本次会话摘要（早期对话压缩）】\n{session_summary}"
        )
        to_send = [enhanced_sys] + recent
    else:
        to_send = (system_msgs[-1:] if system_msgs else []) + recent

    response = model.invoke(to_send)
    return {"messages": [response]}


tool_node = ToolNode(tools)


## edges
### 条件边：根据模型回复判断下一步走哪里
def should_continue(state: AgentState) -> str:
    """
    判断模型回复后下一步走哪里
    如果模型回复里包含 tool_calls，说明它想用工具，就走 "tools" 节点
    """
    last_message = state["messages"][-1]  # messages 是一个列表，[-1] 取最后一条，也就是模型刚产生的回复
    if last_message.tool_calls:  # 如果最后一条消息是工具调用消息，并且里面确实有工具调用，那就去执行工具
        return "tools"
    return "end"


# 组装图
graph = StateGraph(AgentState)

## 添加节点 名字,函数
graph.add_node("call_model", call_model)
graph.add_node("call_tools", tool_node)
graph.add_node("load_profile", load_profile)
graph.add_node("reflection", reflection)

## 添加边

graph.add_edge(START, "load_profile")  # 从 START 进入 load_profile 节点
graph.add_edge("load_profile", "call_model")  # 从 load_profile 进入 call_model 节点

### 条件边 动态路由
graph.add_conditional_edges(
    "call_model",
    should_continue,  # 根据模型回复判断下一步走哪里
    {
        "tools": "call_tools",  # 如果 should_continue 返回 "tools"，就走 call_tools 节点
        "end": "reflection",  # 如果 should_continue 返回 "end"，就进入 reflection_node 节点
    }
)
graph.add_edge("reflection", END)

### 循环
graph.add_edge("call_tools", "call_model")  # 执行完工具后，回到 call_model 节点让模型观察工具结果并总结, 有可能继续调用工具


def chat_loop():
    # 模拟用户和会话
    user_id = "user_059"
    session_id = "chat_profiletest_004"
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}  # thread_id 为会话id

    # store = InMemoryStore()   # temp store
    DB_URL = "postgresql://neondb_owner:npg_TwzFbQy56nAs@ep-misty-meadow-ao6s5bk7.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&options=endpoint%3Dep-misty-meadow-ao6s5bk7"

    # SqliteSaver.from_conn_string 返回的是上下文管理器，需要在 with 中获取真正的 saver 实例
    with SqliteSaver.from_conn_string("./checkpoints.db") as memory:  # 用sqlite作为memory

        with PostgresStore.from_conn_string(DB_URL) as store:  # 用postgres作为store
            store.setup()  # 初始化 store 表结构（只需首次，重复调用安全）
            app = graph.compile(
                checkpointer=memory,
                store=store,
            )

            print("AI: 你好！我是你的助理。输入 'quit' 或 'exit' 退出对话。\n")

            is_first_run = True

            # chat
            while True:
                user_input = input(f"User({user_id}_{session_id}): ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("AI: 再见！\n")
                    break

                # 调试命令
                if user_input.lower() == "profile":
                    _print_profile(store, user_id)
                    continue
                if user_input.lower() == "facts":
                    _print_facts(store, user_id)
                    continue

                input_message = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id,
                    "session_id": session_id,
                }
                if is_first_run:
                    input_message["turn_count"] = 0
                    input_message["session_summary"] = ""
                    is_first_run = False

                # 调用时传入 config
                # 这里不需要手动把之前的历史传进去，LangGraph 会根据 thread_id 自动从 memory 中加载历史
                # final_output = app.invoke(input_message, config=config)
                # last_ai_message = final_output["messages"][-1]
                # print(f"AI: {last_ai_message.content}\n")

                # stream mode
                for event in app.stream(input_message, config=config):
                    for event_name, value in event.items():
                        print(f"\n{'-' * 20} Node: {event_name} {'-' * 20}\n")
                        if not value or "messages" not in value:
                            print("(no messages)")
                            continue

                        last_message = value["messages"][-1]
                        print(last_message.content or f"Calling Tool: {last_message.tool_calls}")


# debug
def _print_profile(store: BaseStore, user_id: str):
    profile = _get_profile(store, user_id)
    print("\n─── 当前 Profile ───")
    print(f"  用户 ID      : {profile.user_id}")
    print(f"  画像摘要    : {profile.text_summary or '（暂无）'}")
    print(f"  MBTI 推断   : {profile.mbti_scores.dominant_type()}")
    print(f"  置信度      : {profile.mbti_scores.confidence():.0%}")
    print(f"  细项评分    : {json.dumps(profile.mbti_scores.model_dump(), ensure_ascii=False)}")
    print(f"  性格特质    : {profile.traits or '（暂无）'}")
    print(f"  累计对话轮次: {profile.conversation_count}")
    print(f"  最后更新    : {profile.last_updated or '（从未）'}")


def _print_facts(store: BaseStore, user_id: str):
    facts = _get_all_facts(store, user_id)
    print("\n─── 已知 Facts ───")
    if not facts:
        print("  （暂无）")
    for k, v in facts.items():
        val = v.get("value", v) if isinstance(v, dict) else v
        conf = v.get("confidence", "?") if isinstance(v, dict) else "?"
        print(f"  {k}: {val}  (置信度 {conf})")


if __name__ == "__main__":
    chat_loop()



