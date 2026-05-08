from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, dropout_rate: float):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn_layer = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        embedded_inputs = self.dropout_layer(self.embedding_layer(inputs))
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            embedded_inputs,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_states, _) = self.rnn_layer(packed_inputs)
        concatenated_states = torch.cat((hidden_states[-2], hidden_states[-1]), dim=1)
        return self.output_layer(concatenated_states)


def english_tokenizer(text: str) -> list[str]:
    if not isinstance(text, str):
        text = str(text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def predict_emotion(sentence: str, model: RNNClassifier, vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    model.eval()
    tokens = [vocab.get(token, vocab["<unk>"]) for token in english_tokenizer(sentence)]
    if not tokens:
        tokens = [vocab["<unk>"]]

    text_tensor = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)
    length_tensor = torch.tensor([len(tokens)]).to(device)

    with torch.no_grad():
        output = model(text_tensor, length_tensor)

    probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_emotion = EMOTIONS[predicted_class]

    emotion_scores = deque()
    for index, emotion in enumerate(EMOTIONS):
        emotion_scores.append({"emotion": emotion, "score": float(probabilities[index])})

    return {
        "task": "sentiment_model_analysis",
        "predicted_emotion": predicted_emotion,
        "predicted_score": float(probabilities[predicted_class]),
        "emotion_scores": list(emotion_scores),
    }


def load_model_bundle(model_dir: Path, device: torch.device) -> tuple[RNNClassifier, dict[str, int], dict[str, Any]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embedding_size"],
        hidden_size=config["hidden_size"],
        output_size=len(EMOTIONS),
        dropout_rate=config["dropout_rate"],
    ).to(device)
    state = torch.load(model_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    return model, vocab, config


