"""Bridge between omniteleop's ``record_components`` vocabulary and the
dexdata Spec's per-feature topic paths.

``dexdata`` is component-agnostic: the Spec only knows topics. ``omniteleop``
has a coarser UX-level concept ("toggle the left arm", "toggle the head depth
camera") where a single user-facing toggle may gate multiple Spec topics
(state qpos + state qvel + action qpos, or force + torque). This module is
the single source of truth for that 1→N mapping.

Anyone in this repo who needs to translate ``record_components`` keys into
Spec topics MUST import from here — do not redefine the table inline.
"""

from __future__ import annotations

from collections.abc import Mapping

from dexdata.spec import Spec

# YAML toggle name → tuple of Spec topics it gates.
#
# Topics not appearing as a value here are *unconditional*: they are kept
# whenever the Spec declares them. /chassis/action/velocity is the current
# example — a robot either has a chassis (spec declares the topic) or it
# does not (vega_1u variants); there is no per-episode toggle.
COMPONENT_TO_TOPICS: dict[str, tuple[str, ...]] = {
    "left_arm": (
        "/robot/state/left_arm/qpos",
        "/robot/state/left_arm/qvel",
        "/robot/action/left_arm/qpos",
    ),
    "right_arm": (
        "/robot/state/right_arm/qpos",
        "/robot/state/right_arm/qvel",
        "/robot/action/right_arm/qpos",
    ),
    "head": (
        "/robot/state/head/qpos",
        "/robot/action/head/qpos",
    ),
    "torso": (
        "/robot/state/torso/qpos",
        "/robot/action/torso/qpos",
    ),
    "left_hand": (
        "/robot/state/left_hand/qpos",
        "/robot/action/left_hand/qpos",
    ),
    "right_hand": (
        "/robot/state/right_hand/qpos",
        "/robot/action/right_hand/qpos",
    ),
    "left_wrist_wrench": (
        "/robot/state/left_hand/wrench/force",
        "/robot/state/left_hand/wrench/torque",
    ),
    "right_wrist_wrench": (
        "/robot/state/right_hand/wrench/force",
        "/robot/state/right_hand/wrench/torque",
    ),
    "head_left_rgb":  ("/camera/head_left/rgb/video",),
    "head_right_rgb": ("/camera/head_right/rgb/video",),
    "head_left_depth": ("/camera/head_left/depth",),
    "left_wrist_rgb": ("/camera/left_wrist/rgb/video",),
    "right_wrist_rgb": ("/camera/right_wrist/rgb/video",),
}

# Reverse index: topic → the component that gates it (or absent → unconditional).
TOPIC_TO_COMPONENT: dict[str, str] = {
    topic: component
    for component, topics in COMPONENT_TO_TOPICS.items()
    for topic in topics
}


# Deployer-side sensor-output short names → Spec topics. These are the keys
# the online-policy deployer uses in its per-frame images dict (a training-
# data naming convention preserved end-to-end). As of spec 0.2.0 these names
# match the YAML toggle names in COMPONENT_TO_TOPICS one-for-one — the depth
# stream is ``head_left_depth`` on both sides — so the two vocabularies no
# longer diverge.
SENSOR_OUTPUT_TO_TOPIC: dict[str, str] = {
    "head_left_rgb":   "/camera/head_left/rgb/video",
    "head_right_rgb":  "/camera/head_right/rgb/video",
    "head_left_depth": "/camera/head_left/depth",
    "left_wrist_rgb":  "/camera/left_wrist/rgb/video",
    "right_wrist_rgb": "/camera/right_wrist/rgb/video",
}


def resolve_topics(
    spec: Spec,
    record_components: Mapping[str, bool],
) -> set[str]:
    """Resolve the set of Spec topics to record.

    Inputs are mixed at the *omniteleop* layer (after the recorder has
    adjusted ``record_components`` to drop entries the live robot lacks).
    The output is what to hand to :func:`dexdata.spec.restrict`.

    Rules:

    * **Unknown toggle** — a key in ``record_components`` that is not in
      :data:`COMPONENT_TO_TOPICS` — raises ``ValueError``. Catches typos
      and stale YAMLs at startup.
    * **Toggle ON but no matching topic in spec** raises ``ValueError``.
      e.g. ``head_left_depth: true`` on a depth-less spec, or ``torso: true``
      on a ``vega_1u`` (upper-body) spec.
    * **Toggle OFF** — its topics are excluded.
    * **Topics with no component** (no entry in ``TOPIC_TO_COMPONENT``) are
      always kept if the spec declares them.
    """
    unknown = set(record_components) - set(COMPONENT_TO_TOPICS)
    if unknown:
        raise ValueError(
            f"unknown record_components keys: {sorted(unknown)} "
            f"(known: {sorted(COMPONENT_TO_TOPICS)})"
        )

    spec_topics = set(spec.topics())
    keep: set[str] = {
        topic for topic in spec_topics if topic not in TOPIC_TO_COMPONENT
    }

    for component, enabled in record_components.items():
        if not enabled:
            continue
        gated = set(COMPONENT_TO_TOPICS[component])
        present = gated & spec_topics
        if not present:
            raise ValueError(
                f"record_components[{component!r}] is True but spec "
                f"{spec.spec_id!r} declares none of its topics "
                f"(expected one of {sorted(gated)})"
            )
        keep |= present

    return keep
