"""FROZEN scenario schema — the single source of truth for the record both the
generator (Michael) and the labeling UI (Cara) build against.

The contract is bidirectional:
  - Generator -> UI: emit every field except gold_action (null) and
    acceptable_actions ([]); base_solve_rate is null (calibration fills it).
  - UI -> Generator: the labeler writes gold_action + acceptable_actions back
    into the *same* record, unchanged otherwise. Use `apply_label()` so the
    write path is pinned, not improvised.

`json_schema()` exports the same contract as JSON Schema for non-Python UIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from goldilocks_eval.prompting import node_id_str

ENVS = ("placement",)
SPLITS = ("train", "heldout")

# The frozen field set. Order is documentation; JSON is key-addressed.
FIELDS = (
    "scenario_id", "game_id", "board_seed", "pick_index", "env",
    "serialized_state", "legal_actions", "gold_action", "acceptable_actions",
    "base_solve_rate", "split",
)


@dataclass
class Scenario:
    scenario_id: str
    game_id: str                      # = board seed; grouping key for the split
    board_seed: int
    env: str                          # "placement" for v1
    serialized_state: Any             # Catanatron GameEncoder JSON
    legal_actions: List[str]          # canonical "node_<int>" ids
    split: str                        # "train" | "heldout"
    pick_index: Optional[int] = None  # 1..4 (placement-specific)
    gold_action: Optional[str] = None       # champion label; null until labeled
    acceptable_actions: List[str] = field(default_factory=list)
    base_solve_rate: Optional[float] = None  # filled by calibration

    @property
    def is_labeled(self) -> bool:
        return self.gold_action is not None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=d["scenario_id"],
            game_id=str(d.get("game_id", d.get("board_seed", ""))),
            board_seed=d["board_seed"],
            env=d.get("env", "placement"),
            serialized_state=d.get("serialized_state"),
            legal_actions=[node_id_str(a) for a in d["legal_actions"]],
            split=d.get("split", "heldout"),
            pick_index=d.get("pick_index"),
            gold_action=(node_id_str(d["gold_action"])
                         if d.get("gold_action") is not None else None),
            acceptable_actions=[node_id_str(a) for a in d.get("acceptable_actions", [])],
            base_solve_rate=d.get("base_solve_rate"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "game_id": self.game_id,
            "board_seed": self.board_seed,
            "pick_index": self.pick_index,
            "env": self.env,
            "serialized_state": self.serialized_state,
            "legal_actions": list(self.legal_actions),
            "gold_action": self.gold_action,
            "acceptable_actions": list(self.acceptable_actions),
            "base_solve_rate": self.base_solve_rate,
            "split": self.split,
        }


def new_unlabeled(scenario_id: str, board_seed: int, env: str,
                  serialized_state: Any, legal_actions: List[Any], split: str,
                  pick_index: Optional[int] = None) -> Dict[str, Any]:
    """Generator-side constructor. Labels intentionally empty."""
    return Scenario(
        scenario_id=scenario_id, game_id=str(board_seed), board_seed=board_seed,
        env=env, serialized_state=serialized_state,
        legal_actions=[node_id_str(a) for a in legal_actions], split=split,
        pick_index=pick_index,
    ).to_dict()


def apply_label(record: Dict[str, Any], gold_action: Any,
                acceptable_actions: Optional[List[Any]] = None) -> Dict[str, Any]:
    """UI-side write-back (the pinned UI -> generator direction).

    Validates that every labeled node is legal, normalizes ids, and returns the
    SAME record with only the label fields set. Raises ValueError on an illegal
    or empty gold."""
    legal = {node_id_str(a) for a in record["legal_actions"]}
    gold = node_id_str(gold_action)
    if gold not in legal:
        raise ValueError(f"gold_action {gold} not in legal_actions")
    acc = []
    for a in (acceptable_actions or []):
        a = node_id_str(a)
        if a not in legal:
            raise ValueError(f"acceptable action {a} not in legal_actions")
        if a != gold and a not in acc:
            acc.append(a)
    out = dict(record)
    out["gold_action"] = gold
    out["acceptable_actions"] = acc
    return out


def validate(record: Dict[str, Any]) -> List[str]:
    """Return a list of contract violations (empty = valid)."""
    errs: List[str] = []
    for f in ("scenario_id", "board_seed", "env", "serialized_state",
              "legal_actions", "split"):
        if f not in record or record[f] is None:
            errs.append(f"missing required field: {f}")
    if record.get("split") not in SPLITS:
        errs.append(f"split must be one of {SPLITS}")
    la = record.get("legal_actions")
    if not isinstance(la, list) or not la:
        errs.append("legal_actions must be a non-empty list")
    g = record.get("gold_action")
    if g is not None and la and node_id_str(g) not in {node_id_str(a) for a in la}:
        errs.append("gold_action not in legal_actions")
    return errs


def json_schema() -> Dict[str, Any]:
    """Language-agnostic contract for non-Python consumers (e.g. a JS UI)."""
    node = {"type": "string", "pattern": r"^node_\d+$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PlacementScenario",
        "type": "object",
        "additionalProperties": False,
        "required": ["scenario_id", "game_id", "board_seed", "env",
                     "serialized_state", "legal_actions", "gold_action",
                     "acceptable_actions", "split"],
        "properties": {
            "scenario_id": {"type": "string"},
            "game_id": {"type": "string", "description": "= board seed; split grouping key"},
            "board_seed": {"type": "integer"},
            "pick_index": {"type": ["integer", "null"], "minimum": 1, "maximum": 4},
            "env": {"type": "string", "enum": list(ENVS)},
            "serialized_state": {"type": "object", "description": "Catanatron GameEncoder JSON"},
            "legal_actions": {"type": "array", "items": node, "minItems": 1},
            "gold_action": {"oneOf": [node, {"type": "null"}],
                            "description": "champion label; null until labeled (UI writes this)"},
            "acceptable_actions": {"type": "array", "items": node,
                                   "description": "near-optimal alts (UI writes this)"},
            "base_solve_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "split": {"type": "string", "enum": list(SPLITS)},
        },
    }
