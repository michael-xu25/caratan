"""Node-id -> 2D board position, derived purely from `serialized_state`.

This resolves the one genuine generator/UI integration risk (freeze-schema-task
#3): can the labeling UI place every `legal_actions` node on a rendered board
using only what we serialize? **Yes.** Each node in `serialized_state["nodes"]`
carries `tile_coordinate` (cube) + `direction` (NodeRef corner) — and

    node_position = cube_to_pixel(tile_coordinate) + node_delta(direction)

is exactly what Catanatron's own renderer does. No serialization change needed.

Math copied verbatim from catanatron's `gym/envs/pygame_renderer.py`
(`cube_to_pixel` + `get_node_delta`), whose docstring notes it "matches the
frontend getNodeDelta function" — so a JS UI can mirror this 1:1. Positions here
are unit-scaled and un-centered; the UI applies its own `size` and canvas
center.
"""
from __future__ import annotations

import math
from typing import Mapping, Tuple

from goldilocks_eval.prompting import node_id_int

SQRT3 = math.sqrt(3)

# Offset from tile center to each corner, in units of `size` (from get_node_delta:
# w = sqrt3*size, h = 2*size; e.g. NORTH=(0,-h/2)=(0,-size), NE=(w/2,-h/4)).
_NODE_DELTA = {
    "NORTH":     (0.0,        -1.0),
    "NORTHEAST": (SQRT3 / 2,  -0.5),
    "SOUTHEAST": (SQRT3 / 2,   0.5),
    "SOUTH":     (0.0,         1.0),
    "SOUTHWEST": (-SQRT3 / 2,  0.5),
    "NORTHWEST": (-SQRT3 / 2, -0.5),
}


def tile_center(cube, size: float = 1.0) -> Tuple[float, float]:
    """Cube coordinate -> pixel center (Catanatron's flat-top axial mapping)."""
    x, _y, z = cube
    q, r = x, z
    px = size * (SQRT3 * q + SQRT3 / 2 * r)
    py = size * (1.5 * r)
    return (px, py)


def node_position(serialized_state: Mapping, node_id, size: float = 1.0
                  ) -> Tuple[float, float]:
    """2D position of a node (canonical id or int), from serialized_state only."""
    node = serialized_state["nodes"][str(node_id_int(node_id))]
    cx, cy = tile_center(node["tile_coordinate"], size)
    dx, dy = _NODE_DELTA[node["direction"]]
    return (cx + dx * size, cy + dy * size)
