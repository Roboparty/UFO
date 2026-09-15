"""Neura 4NE1 G3.3 humanoid robot configuration for mjlab.

This is the default 4NE1 model (``g3_3``) selected through
``teleopit.robots.neura_4ne1.load_constants`` (training task
``General-Tracking-4NE1``). The robot XML, the finger-welding ``get_spec``
and the frozen body/joint contract live in ``model.py`` and are shared with
the ``neura_4ne1_g3_3_constants_sysid`` variant.

Robot model: ``neura_4ne1_g3_3.xml`` (generated from ``neura_4ne1_g3_3.urdf``
via ``config/g3_3.yaml`` + postprocess). Unlike the G3.1 "without hands"
model this MJCF ships both hands: 12 finger joints per hand
(``{side}_{thumb_{1..4},index_{1,2},middle_{1,2},ring_{1,2},little_{1,2}}_joint``).

Actuator model parameters are carried over from
``g3_1_no_shells/neura_4ne1_constants_systemid_from_Liuxin_0810.py`` (the
iteration-199 fit at ``logs/pace/neura_4ne1/26_08_05_16-42-05/mean_199.pt``),
with one exception: the ankle actuators use the updated hardware spec table
(received 2026-08-28) instead of the 0810 fit --

  ================================  ==============  ==============
  Parameter                         ankle_y (pitch) ankle_x (roll)
  ================================  ==============  ==============
  Velocity limit (no-load speed)    10.97 rad/s     16.18 rad/s
  Effort/torque saturation limit    374 N*m         252 N*m
  Rated effort/torque limit         134 N*m         90.26 N*m
  Coulomb friction limit            1.88 N*m        1.27 N*m
  Viscous friction coeff            0.547 N*m*s/rad 0.249 N*m*s/rad
  Reflected armature inertia        0.197 kg*m^2    0.0895 kg*m^2
  ================================  ==============  ==============

The ankle entries are therefore removed from ``IDENTIFIED_JOINT_PARAMS`` (so
``_apply_identified_params`` passes the ankle groups through untouched) and
the values above live directly on the ankle group cfgs.

The identified global command delay is 3.5477375984191895 physics steps; it is
recorded in ``IDENTIFIED_DELAY`` but not applied in ``DELAY_KW`` -- enabling
``delay_max_lag > 0`` still crashes any task that also uses
``dr.effort_limits`` (see the delay block below).

Joint physics (``damping``, ``frictionloss``, ``armature``) are deliberately
absent from the G3.3 XML (see ``config/g3_3.yaml``). Each actuator cfg below
supplies them via its ``viscous_damping``, ``frictionloss``, and ``armature``
fields -- the cfg is the single place to set any value per actuator group.

Hands: locomotion training must not pay for (or be perturbed by) the hands,
so ``get_spec`` neutralizes them entirely while KEEPING their mass:

  * the 24 finger joints (``FINGER_JOINT_NAMES``) are stripped from the XML,
    welding each finger rigidly to the hand at the zero pose -- no extra
    DOFs, no dangling dynamics, no state for the solver;
  * every hand collision geom (hand base + finger hulls, 13 per side) is
    stripped, so the hands generate no contacts at all -- neither with the
    terrain, the robot, nor each other;
  * the hand BODIES and their inertials stay, so arm dynamics still feel the
    real hand mass, and the visual meshes stay for rendering.

As a result the compiled model has exactly the same joint set, actuator set,
and ordering as the G3.1 "without hands" configs: policy action and
observation dimensions are unchanged (see ``LOCOMOTION_JOINT_NAMES``).

Two actuator backends are supported, selected by ``USE_DC_MOTOR_ACTUATOR``:

  * ``False``: MuJoCo built-in ``<position>`` actuators
    (``BuiltinPositionActuatorCfg``). MuJoCo evaluates the PD law natively;
    the ``implicitfast`` integrator treats the velocity term implicitly,
    giving good stability at the 2.5 ms timestep.

  * ``True`` (default): ``DcMotorActuatorCfg`` from
    ``mjlab.actuator.dc_actuator``. This installs MuJoCo ``<motor>``
    actuators (effort transmission) and computes the PD law plus a DC motor
    torque-speed curve in Python:
    ``tau = clip(Kp·(q_ref - q) - Kd·dq + tau_ff, torque_speed_curve)``
    where the upper/lower envelopes are
    ``±saturation_effort · (1 ∓ q̇/velocity_limit)``, then further clipped
    to ``±effort_limit``. The two-stage clip yields stall torque
    ``saturation_effort`` at zero velocity, zero torque at the no-load
    speed ``velocity_limit``, and a continuous-rating ceiling of
    ``effort_limit``.
"""
from __future__ import annotations

import dataclasses
from typing import TypedDict, TypeVar

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.actuator.actuator import ActuatorCfg, TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# ── Model contract (shared by the G3.3 actuator variants) ───────────────────
# The robot XML, the finger-welding ``get_spec`` transform and the frozen
# BODY_NAMES / JOINT_NAMES layout live in ``model.py`` so the two actuator
# identifications cannot drift apart on the model side. The names below are
# re-exported so every consumer sees the same API on each variant module (see
# ``teleopit.robots.neura_4ne1.REQUIRED_SYMBOLS``).
from teleopit.robots.neura_4ne1_g3_3.model import (  # noqa: F401
  BODY_NAMES,
  FEET_BODY_NAMES,
  FINGER_JOINT_NAMES,
  FOOT_CONTACT_GEOM_NAMES,
  FOOT_CONTACT_GEOM_REGEX,
  FULL_QPOS_DIM,
  JOINT_NAMES,
  NEURA_4NE1_DIR,
  NEURA_4NE1_SIM2SIM_XML,
  NEURA_4NE1_WITHOUT_HANDS_XML,
  NEURA_4NE1_XML,
  NUM_JOINTS,
  SOLE_CONTACT_OFFSETS,
  build_welded_xml,
  compile_model,
  get_spec,
  verify_model_contract,
)

MODEL_NAME = "g3_3"

# ── Actuator backend toggle ─────────────────────────────────────────────────
# False -> use MuJoCo BuiltinPositionActuatorCfg.
# True  -> use mjlab DcMotorActuatorCfg (PD + DC motor torque-speed curve).
USE_DC_MOTOR_ACTUATOR = True


# ── Actuator Configurations ─────────────────────────────────────────────────


# Shared delay settings: command targets arrive 0-3 physics timesteps late
# when ``delay_max_lag > 0``; with the current value (0) the
# mjlab-patched ``BuiltinPositionActuatorCfg`` / ``DcMotorActuatorCfg``
# build path skips the ``DelayedActuator`` wrap entirely (see
# ``delay_*`` fields on stock mjlab >= 1.5 ``ActuatorCfg``), so this
# block is a strict no-op until someone opts in.
#
# The 0810 identification reports a global command delay of
# 3.5477375984191895 physics steps (``IDENTIFIED_DELAY``). It is not applied
# here because enabling ``delay_max_lag > 0`` exposes a latent upstream mjlab
# 1.2.0 bug in ``dr.effort_limits``, which does not unwrap ``DelayedActuator``
# before its ``isinstance`` dispatch (only ``dr.pd_gains`` does). Activating
# delay DR here therefore crashes any task that also uses
# ``dr.effort_limits`` with
# ``TypeError: effort_limits only supports BuiltinPositionActuator,
# XmlPositionActuator, and IdealPdActuator, got DelayedActuator``.
# See ``neura_4ne1_constants_systemid_from_Liuxin_0810.py`` for the
# enablement recipe.
class _DelayKw(TypedDict):
  delay_min_lag: int
  delay_max_lag: int
  delay_hold_prob: float
  delay_update_period: int
  delay_per_env_phase: bool


IDENTIFIED_DELAY: float = 3.5477375984191895

DELAY_KW: _DelayKw = {
  "delay_min_lag": 0,
  "delay_max_lag": 0,
  "delay_hold_prob": 0.0,
  "delay_update_period": 0,
  "delay_per_env_phase": True,
}


# ── System-identified joint physics (0810 update) ───────────────────────────
# Per-joint fit of the passive joint model
#   tau_passive = armature * qdd + viscous_damping * qd
#                 + coulomb_friction * sign(qd)
# The keys map onto the actuator cfg fields as:
#   armature          -> ActuatorCfg.armature
#   viscous_damping   -> ActuatorCfg.viscous_damping (passive <joint damping>)
#   coulomb_friction  -> ActuatorCfg.frictionloss
# Joints absent from this table keep their group values below. That covers
# the wrists, head, and fingers (never identified) AND the four ankle joints:
# the ankles use the 2026-08-28 hardware spec table on the group cfgs instead
# of the 0810 fit, so their 0810 entries are removed here.
class _JointParams(TypedDict):
  armature: float
  viscous_damping: float
  coulomb_friction: float


IDENTIFIED_JOINT_PARAMS: dict[str, _JointParams] = {
  "torso_z": {
    "armature": 0.5419993996620178,
    "viscous_damping": 4.899504661560059,
    "coulomb_friction": 4.989860534667969,
  },
  "torso_y": {
    "armature": 1.2511764764785767,
    "viscous_damping": 12.423195838928223,
    "coulomb_friction": 2.524285078048706,
  },
  "torso_x": {
    "armature": 0.3873822093009949,
    "viscous_damping": 5.55401086807251,
    "coulomb_friction": 4.888368606567383,
  },
  "right_hip_z": {
    "armature": 0.13580648601055145,
    "viscous_damping": 3.5411858558654785,
    "coulomb_friction": 3.4635605812072754,
  },
  "right_hip_x": {
    "armature": 0.3178637623786926,
    "viscous_damping": 18.333251953125,
    "coulomb_friction": 5.5050506591796875,
  },
  "right_hip_y": {
    "armature": 0.556557297706604,
    "viscous_damping": 8.033621788024902,
    "coulomb_friction": 3.9963836669921875,
  },
  "right_knee": {
    "armature": 0.21907813847064972,
    "viscous_damping": 2.858776569366455,
    "coulomb_friction": 3.2511513233184814,
  },
  "left_hip_z": {
    "armature": 0.14749769866466522,
    "viscous_damping": 3.9553463459014893,
    "coulomb_friction": 3.3464794158935547,
  },
  "left_hip_x": {
    "armature": 0.2669678330421448,
    "viscous_damping": 19.03363800048828,
    "coulomb_friction": 5.777322769165039,
  },
  "left_hip_y": {
    "armature": 0.5733145475387573,
    "viscous_damping": 8.303862571716309,
    "coulomb_friction": 4.1222124099731445,
  },
  "left_knee": {
    "armature": 0.23207047581672668,
    "viscous_damping": 2.430616617202759,
    "coulomb_friction": 4.843557834625244,
  },
  "right_shoulder_y": {
    "armature": 0.15019643306732178,
    "viscous_damping": 3.6089301109313965,
    "coulomb_friction": 4.350053787231445,
  },
  "right_shoulder_x": {
    "armature": 1.394777536392212,
    "viscous_damping": 18.11485481262207,
    "coulomb_friction": 5.866274356842041,
  },
  "right_shoulder_z": {
    "armature": 0.09023027122020721,
    "viscous_damping": 0.1331108808517456,
    "coulomb_friction": 2.144036293029785,
  },
  "right_elbow": {
    "armature": 0.04860277473926544,
    "viscous_damping": 0.06096184253692627,
    "coulomb_friction": 1.9425913095474243,
  },
  "left_shoulder_y": {
    "armature": 0.17330707609653473,
    "viscous_damping": 3.5879902839660645,
    "coulomb_friction": 3.502495050430298,
  },
  "left_shoulder_x": {
    "armature": 1.395184874534607,
    "viscous_damping": 8.706703186035156,
    "coulomb_friction": 5.724769592285156,
  },
  "left_shoulder_z": {
    "armature": 0.06978773325681686,
    "viscous_damping": 0.072479248046875,
    "coulomb_friction": 1.0699013471603394,
  },
  "left_elbow": {
    "armature": 0.16358907520771027,
    "viscous_damping": 1.0804390907287598,
    "coulomb_friction": 3.7523088455200195,
  },
}

# Constant encoder offset (rad) reported by the 0810 identification run:
# ``q_measured = q_true + bias``. There is no actuator cfg field for it, so it
# is not applied to the simulation; kept here as reference data for anyone
# wiring it into an observation-bias term or a sim-to-real calibration.
IDENTIFIED_ENCODER_BIAS: dict[str, float] = {
  "torso_z": 0.004165492951869965,
  "torso_y": 0.09840374439954758,
  "torso_x": -0.04287726804614067,
  "right_hip_z": -0.07170666754245758,
  "right_hip_x": -0.03524043411016464,
  "right_hip_y": -0.02870602160692215,
  "right_knee": 0.0709049180150032,
  "right_ankle_y": 0.010606423020362854,
  "right_ankle_x": -0.06323603540658951,
  "left_hip_z": 0.07566211372613907,
  "left_hip_x": 0.04065997153520584,
  "left_hip_y": -0.005969323217868805,
  "left_knee": 0.07363619655370712,
  "left_ankle_y": 0.03612873703241348,
  "left_ankle_x": 0.019748590886592865,
  "right_shoulder_y": 0.08467025309801102,
  "right_shoulder_x": -0.09834154695272446,
  "right_shoulder_z": -0.09642478078603745,
  "right_elbow": -0.08167074620723724,
  "left_shoulder_y": -0.021528884768486023,
  "left_shoulder_x": 0.09377355128526688,
  "left_shoulder_z": 0.09626764804124832,
  "left_elbow": -0.09295926988124847,
}


# Joints whose identified inertia is rejected in favour of the nominal group
# value from the actuator cfg below. All armatures from this fit are retained.
ARMATURE_FROM_GROUP_JOINTS: frozenset[str] = frozenset()


# ── PD gains ────────────────────────────────────────────────────────────────
# Carried over from ``neura_4ne1_constants_systemid_from_Liuxin_0810.py``
# (originally ``neura_4ne1_constants_no_sysid_new_pd_flexion.py``). Named per
# group so the two backends cannot drift apart: each pair below is spread into
# both the BuiltinPositionActuatorCfg and the DcMotorActuatorCfg of the same
# group. The group ``armature`` / ``frictionloss`` / ``viscous_damping`` are
# the nominal values on both backends: every joint in
# ``IDENTIFIED_JOINT_PARAMS`` overrides them in ``_apply_identified_params``,
# so they take effect for the ankle, wrist, and head groups only.
class _PdKw(TypedDict):
  stiffness: float
  damping: float


TORSO_PD: _PdKw = {"stiffness": 208.298, "damping": 17.681}
HIP_Z_PD: _PdKw = {"stiffness": 208.298, "damping": 17.681}
HIP_X_PD: _PdKw = {"stiffness": 387.727, "damping": 32.911}
HIP_Y_PD: _PdKw = {"stiffness": 763.685, "damping": 64.824}
KNEE_PD: _PdKw = {"stiffness": 387.727, "damping": 32.911}

### This proved to be too high, on the robot it was too stiff and shaky
# ANKLE_Y_PD: _PdKw = {"stiffness": 650.0, "damping": 25.0}
# ANKLE_X_PD: _PdKw = {"stiffness": 250.0, "damping": 10.0}
#######################################################################
ANKLE_Y_PD: _PdKw = {"stiffness": 350.0, "damping": 30.0}
ANKLE_X_PD: _PdKw = {"stiffness": 180.0, "damping": 10.0}
### This proved to be too low, on the robot it was floppy
# ANKLE_Y_PD: _PdKw = {"stiffness": 194.0, "damping": 8.0}
# ANKLE_X_PD: _PdKw = {"stiffness": 89.0, "damping": 4.0}
#######################################################################

SHOULDER_YX_PD: _PdKw = {"stiffness": 208.298, "damping": 17.681}
SHOULDER_Z_ELBOW_PD: _PdKw = {"stiffness": 92.824, "damping": 7.879}
WRIST_PD: _PdKw = {"stiffness": 10.0, "damping": 1.0}
HEAD_PD: _PdKw = {"stiffness": 15.0, "damping": 0.75}


# ── Ankle actuator spec (2026-08-28 table) ──────────────────────────────────
# Shared between the builtin and DC ankle groups so the two backends cannot
# drift apart. ``saturation_effort`` / ``velocity_limit`` only exist on the
# DC backend (they parameterize the torque-speed envelope).
class _AnkleSpec(TypedDict):
  effort_limit: float
  saturation_effort: float
  velocity_limit: float
  armature: float
  frictionloss: float
  viscous_damping: float


ANKLE_Y_SPEC: _AnkleSpec = {
  "effort_limit": 134.0,  # rated torque [N*m]
  "saturation_effort": 374.0,  # stall torque [N*m]
  "velocity_limit": 10.97,  # no-load speed [rad/s]
  "armature": 0.197,  # reflected inertia [kg*m^2]
  "frictionloss": 1.88,  # Coulomb friction [N*m]
  "viscous_damping": 0.547,  # viscous coeff [N*m*s/rad]
}

ANKLE_X_SPEC: _AnkleSpec = {
  "effort_limit": 90.26,
  "saturation_effort": 252.0,
  "velocity_limit": 16.18,
  "armature": 0.0895,
  "frictionloss": 1.27,
  "viscous_damping": 0.249,
}


def _builtin_kw(spec: _AnkleSpec) -> dict[str, float]:
  """Drop the DC-only torque-speed fields for the builtin backend."""
  return {
    k: v
    for k, v in spec.items()
    if k not in ("saturation_effort", "velocity_limit")
  }


# ── Builtin <position> actuator groups ──────────────────────────────────────

TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_z", "torso_y", "torso_x"),
  transmission_type=TransmissionType.JOINT,
  **TORSO_PD,
  effort_limit=60.0,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

HIP_Z_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_hip_z", "right_hip_z"),
  transmission_type=TransmissionType.JOINT,
  **HIP_Z_PD,
  effort_limit=60.0,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

HIP_X_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_hip_x", "right_hip_x"),
  transmission_type=TransmissionType.JOINT,
  **HIP_X_PD,
  effort_limit=100.0,
  armature=0.2016,
  frictionloss=3.3487,
  viscous_damping=1.752,
  **DELAY_KW,
)

HIP_Y_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_hip_y", "right_hip_y"),
  transmission_type=TransmissionType.JOINT,
  **HIP_Y_PD,
  effort_limit=110.0,
  armature=0.48165,
  frictionloss=4.6949,
  viscous_damping=1.386,
  **DELAY_KW,
)

KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_knee", "right_knee"),
  transmission_type=TransmissionType.JOINT,
  **KNEE_PD,
  effort_limit=100.0,
  armature=0.2016,
  frictionloss=3.3487,
  viscous_damping=1.752,
  **DELAY_KW,
)

ANKLE_Y_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_ankle_y", "right_ankle_y"),
  transmission_type=TransmissionType.JOINT,
  **ANKLE_Y_PD,
  **_builtin_kw(ANKLE_Y_SPEC),
  **DELAY_KW,
)

ANKLE_X_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_ankle_x", "right_ankle_x"),
  transmission_type=TransmissionType.JOINT,
  **ANKLE_X_PD,
  **_builtin_kw(ANKLE_X_SPEC),
  **DELAY_KW,
)

SHOULDER_YX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_y",
    "right_shoulder_y",
    "left_shoulder_x",
    "right_shoulder_x",
  ),
  transmission_type=TransmissionType.JOINT,
  **SHOULDER_YX_PD,
  effort_limit=60.0,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

SHOULDER_Z_ELBOW_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_z",
    "right_shoulder_z",
    "left_elbow",
    "right_elbow",
  ),
  transmission_type=TransmissionType.JOINT,
  **SHOULDER_Z_ELBOW_PD,
  effort_limit=30.0,
  armature=0.0666,
  frictionloss=1.1530,
  viscous_damping=0.782,
  **DELAY_KW,
)

# Not covered by the identification: physics stays hand-tuned.
WRIST_Z_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("left_wrist_z", "right_wrist_z"),
  transmission_type=TransmissionType.JOINT,
  **WRIST_PD,
  effort_limit=30.0,
  armature=0.0666,
  frictionloss=4.66,
  viscous_damping=1.1969,
  **DELAY_KW,
)

WRIST_YX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_wrist_y",
    "right_wrist_y",
    "left_wrist_x",
    "right_wrist_x",
  ),
  transmission_type=TransmissionType.JOINT,
  **WRIST_PD,
  effort_limit=15.0,
  armature=0.01,
  frictionloss=2.5,
  viscous_damping=2.0,
  **DELAY_KW,
)

HEAD_XY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("head_x", "head_y"),
  transmission_type=TransmissionType.JOINT,
  **HEAD_PD,
  effort_limit=15.0,
  armature=0.01,
  frictionloss=2.5,
  viscous_damping=1.0,
  **DELAY_KW,
)

HEAD_Z_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("head_z",),
  transmission_type=TransmissionType.JOINT,
  **HEAD_PD,
  effort_limit=10.0,
  armature=0.0192,
  frictionloss=2.5,
  viscous_damping=1.0,
  **DELAY_KW,
)

# ── DC Motor Actuator Configurations ────────────────────────────────────────


TORSO_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("torso_z", "torso_y", "torso_x"),
  transmission_type=TransmissionType.JOINT,
  **TORSO_PD,
  effort_limit=60.0,
  saturation_effort=110.0,
  velocity_limit=9.0059,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

HIP_Z_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_hip_z", "right_hip_z"),
  transmission_type=TransmissionType.JOINT,
  **HIP_Z_PD,
  effort_limit=60.0,
  saturation_effort=110.0,
  velocity_limit=9.0059,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

HIP_X_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_hip_x", "right_hip_x"),
  transmission_type=TransmissionType.JOINT,
  **HIP_X_PD,
  effort_limit=100.0,
  saturation_effort=190.0,
  velocity_limit=8.9012,
  armature=0.2016,
  frictionloss=3.3487,
  viscous_damping=1.752,
  **DELAY_KW,
)

HIP_Y_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_hip_y", "right_hip_y"),
  transmission_type=TransmissionType.JOINT,
  **HIP_Y_PD,
  effort_limit=110.0,
  saturation_effort=225.0,
  velocity_limit=10.2625,
  armature=0.48165,
  frictionloss=4.6949,
  viscous_damping=1.386,
  **DELAY_KW,
)

KNEE_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_knee", "right_knee"),
  transmission_type=TransmissionType.JOINT,
  **KNEE_PD,
  effort_limit=100.0,
  saturation_effort=190.0,
  velocity_limit=8.9012,
  armature=0.2016,
  frictionloss=3.3487,
  viscous_damping=1.752,
  **DELAY_KW,
)

ANKLE_Y_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_ankle_y", "right_ankle_y"),
  transmission_type=TransmissionType.JOINT,
  **ANKLE_Y_PD,
  **ANKLE_Y_SPEC,
  **DELAY_KW,
)

ANKLE_X_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_ankle_x", "right_ankle_x"),
  transmission_type=TransmissionType.JOINT,
  **ANKLE_X_PD,
  **ANKLE_X_SPEC,
  **DELAY_KW,
)

SHOULDER_YX_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(
    "left_shoulder_y",
    "right_shoulder_y",
    "left_shoulder_x",
    "right_shoulder_x",
  ),
  transmission_type=TransmissionType.JOINT,
  **SHOULDER_YX_PD,
  effort_limit=60.0,
  saturation_effort=110.0,
  velocity_limit=9.0059,
  armature=0.14416,
  frictionloss=4.77,
  viscous_damping=2.456,
  **DELAY_KW,
)

SHOULDER_Z_ELBOW_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(
    "left_shoulder_z",
    "right_shoulder_z",
    "left_elbow",
    "right_elbow",
  ),
  transmission_type=TransmissionType.JOINT,
  **SHOULDER_Z_ELBOW_PD,
  effort_limit=30.0,
  saturation_effort=60.0,
  velocity_limit=7.8540,
  armature=0.0666,
  frictionloss=1.1530,
  viscous_damping=0.782,
  **DELAY_KW,
)

# Not covered by the identification: physics stays hand-tuned.
WRIST_Z_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("left_wrist_z", "right_wrist_z"),
  transmission_type=TransmissionType.JOINT,
  **WRIST_PD,
  effort_limit=30.0,
  saturation_effort=45.0,
  velocity_limit=8.0,
  armature=0.0666,
  frictionloss=4.66,
  viscous_damping=1.1969,
  **DELAY_KW,
)

WRIST_YX_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(
    "left_wrist_y",
    "right_wrist_y",
    "left_wrist_x",
    "right_wrist_x",
  ),
  transmission_type=TransmissionType.JOINT,
  **WRIST_PD,
  effort_limit=15.0,
  saturation_effort=22.5,
  velocity_limit=8.0,
  armature=0.01,
  frictionloss=2.5,
  viscous_damping=2.0,
  **DELAY_KW,
)

HEAD_XY_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("head_x", "head_y"),
  transmission_type=TransmissionType.JOINT,
  **HEAD_PD,
  effort_limit=15.0,
  saturation_effort=22.5,
  velocity_limit=8.0,
  armature=0.01,
  frictionloss=2.5,
  viscous_damping=1.0,
  **DELAY_KW,
)

HEAD_Z_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=("head_z",),
  transmission_type=TransmissionType.JOINT,
  **HEAD_PD,
  effort_limit=10.0,
  saturation_effort=15.0,
  velocity_limit=8.0,
  armature=0.0192,
  frictionloss=2.5,
  viscous_damping=1.0,
  **DELAY_KW,
)

# ── Articulation ─────────────────────────────────────────────────────────────
# The group cfgs above supply the PD gains and the effort / velocity limits;
# for every joint present in ``IDENTIFIED_JOINT_PARAMS`` the group is split
# into single-joint cfgs whose armature / frictionloss / viscous_damping come
# from the identification. Joints in ``ARMATURE_FROM_GROUP_JOINTS`` keep the
# group's nominal armature. Groups with no identified joint -- ankles,
# wrists, head -- are passed through untouched. The passive fingers have no
# actuator group at all; their joint physics is set in ``get_spec``.
_ActuatorCfgT = TypeVar("_ActuatorCfgT", bound=ActuatorCfg)


def _apply_identified_params(
  groups: tuple[_ActuatorCfgT, ...],
) -> tuple[_ActuatorCfgT, ...]:
  out: list[_ActuatorCfgT] = []
  for group in groups:
    if not any(n in IDENTIFIED_JOINT_PARAMS for n in group.target_names_expr):
      out.append(group)
      continue
    for name in group.target_names_expr:
      params = IDENTIFIED_JOINT_PARAMS.get(name)
      if params is None:
        out.append(dataclasses.replace(group, target_names_expr=(name,)))
        continue
      if name in ARMATURE_FROM_GROUP_JOINTS:
        assert group.armature is not None, (
          f"{name} takes its inertia from its actuator group, so group "
          f"{group.target_names_expr} must define armature."
        )
        armature = group.armature
      else:
        armature = params["armature"]
      out.append(
        dataclasses.replace(
          group,
          target_names_expr=(name,),
          armature=armature,
          frictionloss=params["coulomb_friction"],
          viscous_damping=params["viscous_damping"],
        )
      )
  return tuple(out)


_BUILTIN_POSITION_ACTUATOR_GROUPS: tuple[BuiltinPositionActuatorCfg, ...] = (
  TORSO_ACTUATOR_CFG,
  HIP_Z_ACTUATOR_CFG,
  HIP_X_ACTUATOR_CFG,
  HIP_Y_ACTUATOR_CFG,
  KNEE_ACTUATOR_CFG,
  ANKLE_Y_ACTUATOR_CFG,
  ANKLE_X_ACTUATOR_CFG,
  SHOULDER_YX_ACTUATOR_CFG,
  SHOULDER_Z_ELBOW_ACTUATOR_CFG,
  WRIST_Z_ACTUATOR_CFG,
  WRIST_YX_ACTUATOR_CFG,
  HEAD_XY_ACTUATOR_CFG,
  HEAD_Z_ACTUATOR_CFG,
)

_DC_MOTOR_ACTUATOR_GROUPS: tuple[DcMotorActuatorCfg, ...] = (
  TORSO_DC_ACTUATOR_CFG,
  HIP_Z_DC_ACTUATOR_CFG,
  HIP_X_DC_ACTUATOR_CFG,
  HIP_Y_DC_ACTUATOR_CFG,
  KNEE_DC_ACTUATOR_CFG,
  ANKLE_Y_DC_ACTUATOR_CFG,
  ANKLE_X_DC_ACTUATOR_CFG,
  SHOULDER_YX_DC_ACTUATOR_CFG,
  SHOULDER_Z_ELBOW_DC_ACTUATOR_CFG,
  WRIST_Z_DC_ACTUATOR_CFG,
  WRIST_YX_DC_ACTUATOR_CFG,
  HEAD_XY_DC_ACTUATOR_CFG,
  HEAD_Z_DC_ACTUATOR_CFG,
)

_BUILTIN_POSITION_ACTUATORS = _apply_identified_params(
  _BUILTIN_POSITION_ACTUATOR_GROUPS
)
_DC_MOTOR_ACTUATORS = _apply_identified_params(_DC_MOTOR_ACTUATOR_GROUPS)

NEURA_4NE1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _DC_MOTOR_ACTUATORS if USE_DC_MOTOR_ACTUATOR else _BUILTIN_POSITION_ACTUATORS
  ),
  soft_joint_pos_limit_factor=0.95,
)


# ── Initial State ────────────────────────────────────────────────────────────
# Same standing pose as the XML's ``standing`` keyframe. The base height
# 0.942027 m is the height at which the lowest vertex of the foot collision
# hulls meets the floor plane (see ``config/g3_3.yaml``).
#
# NOTE: ``LOCOMOTION_JOINT_NAMES`` / ``NEURA_4NE1_TRAINING_JOINT_NAMES`` are
# derived from this dict's insertion order, but they act as a joint SET: mjlab
# orders the action vector by entity joint order (``JOINT_NAMES``). The 32
# entries replicate the G3.1 dict exactly. The finger joints are absent
# on purpose: ``get_spec`` welds them out of the model, so a joint_pos entry
# for them would reference a joint that no longer exists.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.942027),
  rot=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
  joint_pos={
    "torso_z": 0.0,
    "torso_y": 0.0,
    "torso_x": 0.0,
    "right_hip_z": 0.0,
    "right_hip_x": 0.0,
    "right_hip_y": -0.1,
    "right_knee": 0.3,
    "right_ankle_y": -0.2,
    "right_ankle_x": 0.0,
    "left_hip_z": 0.0,
    "left_hip_x": 0.0,
    "left_hip_y": -0.1,
    "left_knee": 0.3,
    "left_ankle_y": -0.2,
    "left_ankle_x": 0.0,
    "right_shoulder_y": 0.2,
    "right_shoulder_x": -0.05,
    "right_shoulder_z": 0.0,
    "right_elbow": -0.4,
    "right_wrist_z": 0.0,
    "right_wrist_y": 0.0,
    "right_wrist_x": 0.0,
    "left_shoulder_y": 0.2,
    "left_shoulder_x": 0.05,
    "left_shoulder_z": 0.0,
    "left_elbow": -0.4,
    "left_wrist_z": 0.0,
    "left_wrist_y": 0.0,
    "left_wrist_x": 0.0,
    "head_x": 0.0,
    "head_y": 0.0,
    "head_z": 0.0,
  },
  joint_vel={".*": 0.0},
)


# ── Collision ────────────────────────────────────────────────────────────────
# Terrain contact + self-collision ride on separate contype/conaffinity bits
# (MuJoCo rule: two geoms collide iff
#   (A.contype & B.conaffinity) | (B.contype & A.conaffinity) != 0).
#
# Bit 0 ("terrain channel") is what mjlab's terrain/plane uses by default
# (contype=1, conaffinity=1). Bit 1 is reserved for robot↔robot self-collision.
# Every remaining ``*_collision`` geom is contype=2 / conaffinity=3, so it
# collides with BOTH the terrain (bit 0: ground contact + the
# illegal_contacts termination) and other robot parts (bit 1:
# self-collision). The hand collision geoms are already stripped in
# ``get_spec``, so nothing here matches them.
#
# Feet: ``get_spec`` moves each ankle_x hull onto a welded
# ``sensor_{side}_foot_fts`` body and renames it ``{side}_foot_collision``
# (``FOOT_CONTACT_GEOM_REGEX``); on G3.1 the equivalent terrain-contact geoms
# are the four ``*_sole_p*`` pads, and each model exposes its own regex.

_FOOT_COLLISION_REGEX = FOOT_CONTACT_GEOM_REGEX

COLLISION_CFG = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    ".*_collision": 2,  # terrain + self-collision emit
  },
  conaffinity={
    ".*_collision": 3,  # receives from terrain (bit 0) and self-collision (bit 1)
  },
  # Feet win contact-parameter mixing against the terrain plane. MuJoCo
  # combines two equal-priority geoms' friction by the element-wise MAXIMUM
  # (the higher-priority geom wins outright otherwise). The ground plane sits
  # at its default mu=1.0, so with equal priority the per-episode
  # ``foot_friction`` DR (which samples foot friction in [0.4, 1.5]) would be
  # masked below 1.0 -- max(foot, 1.0) -- and the policy would never feel a
  # slippery contact. Raising the feet's priority makes each foot-ground
  # contact use the foot's (randomized) friction verbatim, so the DR spans the
  # full [0.4, 1.5] effective range.
  priority={
    _FOOT_COLLISION_REGEX: 1,
  },
  condim={
    ".*_collision": 3,
  },
  friction={
    _FOOT_COLLISION_REGEX: (1.0, 0.01, 0.01),
  },
)


# ── Robot Config Factory ─────────────────────────────────────────────────────
def get_neura_4ne1_robot_cfg() -> EntityCfg:
  """Return a fresh Neura 4NE1 G3.3 EntityCfg."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(COLLISION_CFG,),
    spec_fn=get_spec,
    articulation=NEURA_4NE1_ARTICULATION,
  )


# Head DOFs are part of the 32 tracked actions in this repo (see
# ``NEURA_4NE1_TRAINING_JOINT_NAMES``); ``LOCOMOTION_JOINT_NAMES`` (29 DOFs,
# head excluded) is kept for tooling that mirrors the AMP configs. The finger
# DOFs are welded out of the model entirely by ``get_spec``, so the action /
# observation layout matches the G3.1 configs exactly.
HEAD_JOINT_NAMES: tuple[str, ...] = ("head_z", "head_x", "head_y")

LOCOMOTION_JOINT_NAMES: tuple[str, ...] = tuple(
  name for name in INIT_STATE.joint_pos if name not in HEAD_JOINT_NAMES
)

# ── Policy Joint Names (ordered as in IsaacLab action config) ───────────────
POLICY_JOINT_NAMES: tuple[str, ...] = (
  "torso_z",
  "torso_y",
  "torso_x",
  "right_hip_z",
  "right_hip_x",
  "right_hip_y",
  "right_knee",
  "right_ankle_y",
  "right_ankle_x",
  "left_hip_z",
  "left_hip_x",
  "left_hip_y",
  "left_knee",
  "left_ankle_y",
  "left_ankle_x",
  "right_shoulder_y",
  "right_shoulder_x",
  "right_shoulder_z",
  "right_elbow",
  "left_shoulder_y",
  "left_shoulder_x",
  "left_shoulder_z",
  "left_elbow",
)

# ── Action Scale ─────────────────────────────────────────────────────────────
# Per-joint action scale maps a normalized policy output ``a`` to a joint
# target offset ``Δq = scale * a`` (then ``q_target = q_default + Δq``). The
# formula ``_ACTION_SCALE_FACTOR * effort_limit / stiffness`` makes the
# position term of the PD law at ``|a| = 1`` equal to
# ``_ACTION_SCALE_FACTOR * effort_limit`` (the Unitree G1 convention, 0.25).
#
# NOTE: the action scale is part of the environment the policy trains against.
# Changing it invalidates existing checkpoints and requires retraining /
# fine-tuning -- do not replay an old checkpoint under a different factor.
# The updated ankle effort limits (134 / 90.26 N*m vs. 60 / 35 on G3.1) widen
# the ankle scales accordingly, so G3.1 checkpoints do not transfer as-is.
_ACTION_SCALE_FACTOR = 0.25
_ACTIVE_ACTUATORS = (
  _DC_MOTOR_ACTUATORS if USE_DC_MOTOR_ACTUATOR else _BUILTIN_POSITION_ACTUATORS
)

NEURA_4NE1_ACTION_SCALE: dict[str, float] = {}
for _a in _ACTIVE_ACTUATORS:
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None and _s is not None and _s > 0.0, (
    f"Actuator group {_a.target_names_expr} must have non-zero stiffness and "
    f"a finite effort_limit to compute an action scale; got "
    f"stiffness={_s}, effort_limit={_e}."
  )
  for _n in _a.target_names_expr:
    NEURA_4NE1_ACTION_SCALE[_n] = _ACTION_SCALE_FACTOR * _e / _s


# ── Observation Joint Names (ordered as in IsaacLab observation config) ─────
OBS_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_y",
  "right_hip_y",
  "torso_z",
  "left_hip_x",
  "right_hip_x",
  "torso_x",
  "left_hip_z",
  "right_hip_z",
  "torso_y",
  "left_knee",
  "right_knee",
  "left_shoulder_y",
  "right_shoulder_y",
  "left_ankle_y",
  "right_ankle_y",
  "left_shoulder_x",
  "right_shoulder_x",
  "left_ankle_x",
  "right_ankle_x",
  "left_shoulder_z",
  "right_shoulder_z",
  "left_elbow",
  "right_elbow",
)
