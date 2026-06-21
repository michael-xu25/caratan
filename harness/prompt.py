"""Prompt rendering + reply parsing.

Reuses Michael's canonical `goldilocks_eval.prompt` verbatim (single source of
truth). The ONLY thing added here is a reasoning ON/OFF system-prompt selector,
for the policy "model reasoning off during training/runs, on during testing".
"""
from __future__ import annotations

# Re-export Michael's rendering/parsing so there's one implementation.
from goldilocks_eval.prompt import (  # noqa: F401
    render_action,
    render_state,
    render_actions,
    build_user_prompt,
    parse_choice,
    SYSTEM_PROMPT,
)

# With reasoning ON we use Michael's prompt (asks for {action, reasoning}).
SYSTEM_PROMPT_WITH_REASONING = SYSTEM_PROMPT

# With reasoning OFF the model returns only the index — cheaper/faster, for
# training/production runs where we don't want (or pay for) model reasoning.
SYSTEM_PROMPT_ACTION_ONLY = (
    "You are an expert Settlers of Catan player in a 1v1 game (first to 15 "
    "victory points wins). You will be given the current game state and a "
    "numbered list of legal actions. Choose the single best action.\n\n"
    "Reply with ONLY a JSON object on one line:\n"
    '{"action": <index>}\n'
    "The index must be one of the listed action indices. Do not add prose "
    "outside the JSON."
)


def system_prompt(capture_reasoning: bool) -> str:
    return SYSTEM_PROMPT_WITH_REASONING if capture_reasoning else SYSTEM_PROMPT_ACTION_ONLY
