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
    RULES_1V1,
)

# With reasoning ON we use Michael's prompt (rules primer + asks for {action, reasoning}).
SYSTEM_PROMPT_WITH_REASONING = SYSTEM_PROMPT

# With reasoning OFF the model returns only the index — cheaper/faster, for
# training/production runs. Same rules primer (the model still needs the rules),
# just a terser output contract.
SYSTEM_PROMPT_ACTION_ONLY = RULES_1V1 + (
    "\n\nReply with ONLY a JSON object on one line:\n"
    '{"action": <index>}\n'
    "The index must be one of the listed action indices. No prose outside the JSON."
)


def system_prompt(capture_reasoning: bool) -> str:
    return SYSTEM_PROMPT_WITH_REASONING if capture_reasoning else SYSTEM_PROMPT_ACTION_ONLY
