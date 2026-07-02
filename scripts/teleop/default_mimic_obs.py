import numpy as np


DEFAULT_MIMIC_OBS_G1 = np.concatenate(
    [
        np.array([0.0, 0.0], dtype=np.float32),  # xy velocity
        np.array([0.8], dtype=np.float32),  # root z
        np.array([0.0, 0.0], dtype=np.float32),  # roll / pitch
        np.array([0.0], dtype=np.float32),  # yaw angular velocity
        np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    ]
)


DEFAULT_MIMIC_OBS = {
    "unitree_g1": DEFAULT_MIMIC_OBS_G1,
    "unitree_g1_with_hands": DEFAULT_MIMIC_OBS_G1,
}
