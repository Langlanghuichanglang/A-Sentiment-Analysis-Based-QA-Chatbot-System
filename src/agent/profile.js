function normalizeProfile(profile = {}) {
  const mbtiType = String(profile.mbtiType || "unknown").toUpperCase();
  const communicationStyle = profile.communicationStyle || "balanced";
  const inferred = preferencesFromMbti(mbtiType);

  const styleOverrides = {
    balanced: {},
    structured: { prefersStructure: true, prefersConcreteSteps: true },
    gentle: { prefersValidation: true },
    practical: { prefersConcreteSteps: true },
    reflective: { prefersReflection: true }
  };

  return {
    mbtiType,
    communicationStyle,
    memoryEnabled: Boolean(profile.memoryEnabled),
    preferences: {
      ...inferred,
      ...(styleOverrides[communicationStyle] || {})
    }
  };
}

function preferencesFromMbti(mbtiType) {
  if (!/^[IE][NS][TF][JP]$/.test(mbtiType)) {
    return {
      prefersValidation: true,
      prefersConcreteSteps: true
    };
  }

  return {
    prefersQuietReflection: mbtiType[0] === "I",
    prefersConcreteSteps: mbtiType[1] === "S",
    prefersReflection: mbtiType[1] === "N",
    prefersLogic: mbtiType[2] === "T",
    prefersValidation: mbtiType[2] === "F",
    prefersStructure: mbtiType[3] === "J",
    prefersFlexibility: mbtiType[3] === "P"
  };
}

module.exports = {
  normalizeProfile,
  preferencesFromMbti
};
