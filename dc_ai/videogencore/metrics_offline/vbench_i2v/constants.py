# modified from https://github.com/Vchitect/VBench/blob/master/scripts/constant.py


DIM_WEIGHT_I2V = {
    "camera_motion": 0.1,
    "i2v_subject": 1,
    "i2v_background": 1,
    "subject_consistency": 1,
    "background_consistency": 1,
    "motion_smoothness": 1,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1,
    "imaging_quality": 1,
}


NORMALIZE_DIC_I2V = {
    "camera_motion": {"Min": 0.0, "Max": 1.0},
    "i2v_subject": {"Min": 0.1462, "Max": 1.0},
    "i2v_background": {"Min": 0.2615, "Max": 1.0},
    "subject_consistency": {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "motion_smoothness": {"Min": 0.7060, "Max": 0.9975},
    "dynamic_degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic_quality": {"Min": 0.0, "Max": 1.0},
    "imaging_quality": {"Min": 0.0, "Max": 1.0},
}


I2V_LIST = [
    "camera_motion",
    "i2v_subject",
    "i2v_background",
]

I2V_QUALITY_LIST = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0
