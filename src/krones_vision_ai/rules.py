"""Published defect categories and their image-area decision thresholds."""

from typing import TypedDict


class DecisionChannel(TypedDict):
    cats: tuple[str, ...]
    thr: int | None


DECISION_CHANNELS: dict[str, DecisionChannel] = {
    "air_bubble": {"cats": ("Air bubble",), "thr": 500},
    "chip": {"cats": ("Chip",), "thr": 200},
    "contamination_light": {"cats": ("Contamination light",), "thr": 180},
    "glass_imperfection": {"cats": ("Glass imperfection",), "thr": 100},
    "scuffing": {"cats": ("Scuffing",), "thr": 75_000},
    "scuffing_heavy": {"cats": ("Scuffing heavy",), "thr": 1_200},
    "always_faulty": {
        "cats": (
            "Break / Crack",
            "Circlip",
            "Contamination dark",
            "Crown cap",
            "Foil / Semitransparent",
            "Foreign object - manual cleaning",
            "Foreign object - washing machine",
            "Glass shard",
            "Insect",
            "Label",
            "Liquid",
            "Mold",
            "No base visible",
            "Paint residue",
            "Straw",
            "Yeast residue",
        ),
        "thr": 1,
    },
    "harmless": {
        "cats": ("Embossing", "Foam residue", "Water drop"),
        "thr": None,
    },
}

CHANNEL_NAMES = list(DECISION_CHANNELS)
CHANNEL_THRESHOLDS: list[int | None] = [
    DECISION_CHANNELS[name]["thr"] for name in CHANNEL_NAMES
]
N_CHANNELS = len(CHANNEL_NAMES)
AGGREGATED_CHANNEL_INDICES = frozenset(
    CHANNEL_NAMES.index(name) for name in ("scuffing", "scuffing_heavy")
)
