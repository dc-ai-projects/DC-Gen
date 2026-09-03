NORMALIZE_DIC = {
    "subject_consistency": {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal_flickering": {"Min": 0.6293, "Max": 1.0},
    "motion_smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic_degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic_quality": {"Min": 0.0, "Max": 1.0},
    "imaging_quality": {"Min": 0.0, "Max": 1.0},
    "object_class": {"Min": 0.0, "Max": 1.0},
    "multiple_objects": {"Min": 0.0, "Max": 1.0},
    "human_action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial_relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance_style": {"Min": 0.0009, "Max": 0.2855},
    "temporal_style": {"Min": 0.0, "Max": 0.364},
    "overall_consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject_consistency": 1,
    "background_consistency": 1,
    "temporal_flickering": 1,
    "motion_smoothness": 1,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1,
    "imaging_quality": 1,
    "object_class": 1,
    "multiple_objects": 1,
    "human_action": 1,
    "color": 1,
    "spatial_relationship": 1,
    "scene": 1,
    "appearance_style": 1,
    "temporal_style": 1,
    "overall_consistency": 1,
}

QUALITY_LIST = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]

SEMANTIC_LIST = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4
