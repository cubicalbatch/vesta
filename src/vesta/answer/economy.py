"""Token economy for the agent answer path.

Local CPU inference is prefill-bound: a measured live chat turn spent 14 000
input tokens to produce 105 output tokens (291 s wall time on an i5-11320H —
prefill is ~95 % of the turn). The benchmark distribution says the median
question is already lean (a ~4.2 k one-shot); ~76 % of all input tokens live
in the ~28 % of questions that enter tool rounds, where every round re-prefills
the whole transcript — one tail question grew 4 236 -> 48 892 input tokens.
The budget therefore targets the tail (tool-round growth, runaway output), not
the median: the pre-seed keeps full passages (starving it cut gold facts and
pushed one-shot questions into thrashing tool loops), the passage cap is an
outlier guard only, and the read window stays generous enough to keep the
scored passage's neighbourhood intact.

This module resolves ONE :class:`EconomyBudget` per turn from the
``answer.agent.*`` settings (declared in :mod:`vesta.answer`). Resolution
rule, per knob:

* a user-set value (one that differs from the setting's registered default)
  ALWAYS wins — the economy never overrides an explicit operator choice;
* otherwise, when the economy is active, the leaner CPU default applies;
* otherwise the registered default (the full-context behaviour).

Economy active means ``answer.agent.economy`` is ``"on"``, or ``"auto"`` AND
the bound local runtime reported CPU-only hardware (``hardware == "cpu"``).
``"auto"`` with remote/unknown hardware (``None``) or a GPU stays on the
full-context defaults.

The benchmark forces the effective economy value via its ``--economy`` flag
(recorded in the run's ``config_json``), so A/B measurement works on any
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vesta.answer import (
    ANSWER_AGENT_AGE_TOOL_CHARS,
    ANSWER_AGENT_COMPACT_PROMPT,
    ANSWER_AGENT_CONTEXT_PROFILE,
    ANSWER_AGENT_ECONOMY,
    ANSWER_AGENT_MAX_OUTPUT_TOKENS,
    ANSWER_AGENT_OUTPUT_RESERVE_TOKENS,
    ANSWER_AGENT_PRESEED_PASSAGE_MAX_CHARS,
    ANSWER_AGENT_PRESEED_PASSAGES,
    ANSWER_AGENT_READ_MAX_CHARS,
    ANSWER_AGENT_SEARCH_ENTRIES,
    ANSWER_AGENT_SEARCH_SNIPPET_CHARS,
    ANSWER_AGENT_TOOL_BUDGET_CHARS,
    ANSWER_AGENT_TOOL_BUDGET_TOKENS,
)
from vesta.config.settings import Setting, SettingsSnapshot


@dataclass(frozen=True)
class EconomyBudget:
    """The resolved per-turn context budget for the agent answer path.

    ``preseed_passages``/``preseed_passage_max_chars`` bound the Round-0
    pre-seed; ``read_max_chars`` bounds each ``read_article`` result (a
    score-aware window, never a head cut); ``max_output_tokens`` is the
    model-request output budget; ``tool_budget_chars`` caps the CUMULATIVE
    tool-result characters inserted over one turn — the re-prefill multiplier
    that makes tool-round tails explode (each round re-sends every insert);
    ``age_tool_chars`` truncates OLDER tool results per-request (the last two
    ``search_entries``/``search_snippet_chars`` shape the compact ``search``
    opt-in knob (``answer.agent.compact_prompt``), True only when the economy
    is active AND the user set it; ``window_tokens``/``output_reserve``/
    ``tool_budget_tokens`` form the context window plan (0 = not
    budgeted — profile ``full``).
    """

    preseed_passages: int
    preseed_passage_max_chars: int
    read_max_chars: int
    max_output_tokens: int
    tool_budget_chars: int
    age_tool_chars: int
    search_entries: int
    search_snippet_chars: int
    compact_prompt: bool = False
    #: The context window this turn is planned against, in
    #: tokens. 0 = no window budgeting (profile ``full``): the char knobs
    #: alone govern.
    window_tokens: int = 0
    #: Tokens held back for the model's OUTPUT inside the window (a context
    #: window covers prompt + completion). 0 when ``window_tokens`` is 0.
    output_reserve: int = 0
    #: Cumulative tool-result insert allowance in ESTIMATED tokens
    #: (:func:`vesta.answer.tokens.estimate_tokens`). 0 = derive per turn
    #: from the window arithmetic (filled in by ``api/agent_chat`` once the
    #: pre-seed exists) or uncapped at ``full``. A user-set value binds.
    tool_budget_tokens: int = 0


#: Leaner budget applied when the economy is active and the knob is at its
#: default. Iteration-3 calibration: strict-accuracy rose +6pp with
#: over-refusal halved, but tokens fell only 1.5% — the tail persists (40/150
#: questions hold 73% of all tokens; the worst ran 3 searches + 3 reads over
#: 6 rounds to 67.9k input, because NOTHING bounded cumulative inserts). So
#: the pre-seed matches full context (6 passages; the passage cap is an
#: outlier guard — typical chunks are <= 3.9k chars), the read window is 6k
#: (Branigan-class facts live well inside 6k of a focused window), repeated
#: tool calls are deduplicated harness-side (see api/agent_chat), and a
#: turn-level tool-insert budget (12k chars) bounds the re-prefill multiplier.
#:
#: Tried and DISABLED by bench measurement (kept as opt-in knobs):
#: * (iter 4) per-request context aging — the truncated stubs nudged borderline
#:   one-shot questions into tool rounds (93->85 one-shots, strict -2.6pp)
#:   while total input stayed flat (+0.4%), because the insert budget already
#:   bounded the re-prefill mass (``answer.agent.age_tool_chars``).
#: * (iter 5/6) compact system prompt + 5x350 search snippets — the control
#:   experiment (iter3 code re-run on today's endpoint) reproduced iter3's
#:   numbers exactly (1.661M input, strict 0.740, 82 one-shots), so the gains
#:   were real, not drift: cheaper on 109/150 questions (-211k) but
#:   destabilized 41 into extra tool rounds (+386k, net +130k) for +1.6pp
#:   strict. Token reduction outranks the score, so the DEFAULTS revert to
#:   iter3 semantics (search 6x400, full prompt) and the compact prompt stays
#:   available as ``answer.agent.compact_prompt`` (``compact_prompt`` on the
#:   budget: True only when the economy is active AND the knob is set).
CPU_ECONOMY_DEFAULTS = EconomyBudget(
    preseed_passages=6,
    preseed_passage_max_chars=6000,
    read_max_chars=6000,
    max_output_tokens=2048,
    tool_budget_chars=12000,
    age_tool_chars=0,
    search_entries=6,
    search_snippet_chars=400,
)


@dataclass(frozen=True)
class _WindowPlan:
    """One context profile's derived knob values.

    By measurement, ``preseed_passages`` stays 6 at both sizes (reducing to 4
    increased over-refusal without gold-containment gains). The 8k plan therefore
    finds its tokens in the per-passage caps, the compact system prompt, the
    1280 output reserve (measured max output 1305; a 1024 reserve would truncate
    1 answer in 150) and whatever the window arithmetic leaves for the tool
    ledger (~2.2-2.4k estimated tokens with the 3.0 chars/token estimator).
    """

    window_tokens: int
    output_reserve: int
    preseed_passages: int
    preseed_passage_max_chars: int
    read_max_chars: int
    max_output_tokens: int
    compact_prompt: bool


#: The named context-window plans. A plan ACTIVATES window budgeting
#: (``window_tokens > 0`` on the resolved budget) wherever it resolves —
#: including on remote hardware where the hardware-gated economy is off — which
#: is what makes the bench A/B on any box. ``full`` is not a plan: it is the
#: absence of one (uncapped baseline).
#:
#: By benchmark measurement: plain ``8k`` uses the full system prompt + the
#: 2400-char passage cap. A narrower 1800-char passage cap or compact prompt
#: lost accuracy and increased refusal during testing, so the best-measured
#: 8k configuration is the default. The compact arm remains reproducible by
#: forcing an 8k profile and setting ``answer.agent.compact_prompt=true`` +
#: ``answer.agent.preseed_passage_max_chars=1800`` (ladder rung one binds both
#: over the plan).
CONTEXT_PLANS: dict[str, _WindowPlan] = {
    "8k": _WindowPlan(
        window_tokens=8_192,
        output_reserve=1_280,
        preseed_passages=6,
        preseed_passage_max_chars=2_400,
        read_max_chars=4_500,
        max_output_tokens=1_280,
        compact_prompt=False,
    ),
    # Benchmark axis: the 8k plan with the narrower 1800-char passage cap and
    # full system prompt. Auto never selects it — force-only, like a bench axis.
    "8k-fullprompt": _WindowPlan(
        window_tokens=8_192,
        output_reserve=1_280,
        preseed_passages=6,
        preseed_passage_max_chars=1_800,
        read_max_chars=4_500,
        max_output_tokens=1_280,
        compact_prompt=False,
    ),
    # Values-identical to "8k" above — kept as the explicit alias for
    # benchmark reproducibility. The equivalence is pinned by test; it must
    # not drift. Auto never selects it — force-only, like a bench axis.
    "8k-fullprompt-wide": _WindowPlan(
        window_tokens=8_192,
        output_reserve=1_280,
        preseed_passages=6,
        preseed_passage_max_chars=2_400,
        read_max_chars=4_500,
        max_output_tokens=1_280,
        compact_prompt=False,
    ),
    "16k": _WindowPlan(
        window_tokens=16_384,
        output_reserve=1_792,
        preseed_passages=6,
        preseed_passage_max_chars=2_400,
        read_max_chars=8_000,
        max_output_tokens=1_792,
        compact_prompt=False,
    ),
}


def _get(sn: SettingsSnapshot | None, setting: Setting[Any]) -> Any:
    """Read one setting, falling back to its default (no snapshot / key absent)."""
    if sn is None:
        return setting.default
    try:
        return sn.get(setting)
    except Exception:
        return setting.default


def resolve_budget(
    sn: SettingsSnapshot | None,
    hardware: str | None,
    window_tokens: int | None = None,
) -> EconomyBudget:
    """Resolve the turn's :class:`EconomyBudget` from settings + hardware +
    context window.

    ``hardware`` is the bound local runtime's tag (``"cpu"``/``"gpu"``) or
    ``None`` for remote/unknown — the fifth element of
    ``api/agent_chat._resolve_llm``'s result. ``window_tokens`` is the live
    context window when the runtime is local (``None`` remote).

    Two independent gates feed one three-rung ladder per knob — user-set
    value wins → else the active plan's derived value → else the registered
    default:

    * the hardware economy (``answer.agent.economy``) as before;
    * the context profile (``answer.agent.context_profile``): a forced
      ``8k``/``16k`` resolves that plan REGARDLESS of the runtime (bench A/B
      on any hardware) and activates window budgeting even where the
      economy is off; ``auto`` maps a local ``window_tokens`` onto a plan
      (≤8192 → 8k, ≤16384 → 16k — budgeting against the REAL window, not
      the plan's name, so a 4096 box never plans for 8192) and means
      ``full`` when there is no window (remote); ``full`` is the uncapped
      behaviour.

    When a plan is active its derived value replaces the CPU-economy
    default as rung two for the knobs it covers; the knobs it does not
    cover (search shaping, aging, the char tool budget) keep the plain
    economy ladder.
    """
    economy = str(_get(sn, ANSWER_AGENT_ECONOMY))
    active = economy == "on" or (economy == "auto" and hardware == "cpu")

    profile = str(_get(sn, ANSWER_AGENT_CONTEXT_PROFILE))
    plan: _WindowPlan | None = None
    window = 0
    if profile in CONTEXT_PLANS:
        plan = CONTEXT_PLANS[profile]
        window = plan.window_tokens
    elif profile == "auto" and window_tokens is not None and window_tokens > 0:
        if window_tokens <= CONTEXT_PLANS["8k"].window_tokens:
            plan = CONTEXT_PLANS["8k"]
            window = int(window_tokens)
        elif window_tokens <= CONTEXT_PLANS["16k"].window_tokens:
            plan = CONTEXT_PLANS["16k"]
            window = int(window_tokens)

    def rung2(setting: Setting[Any], cpu_default: int, plan_value: int | None) -> int:
        """Ladder rung two: the plan's value when one is active, else the
        economy's CPU default, else the registered default."""
        if plan is not None and plan_value is not None:
            return plan_value
        return cpu_default if active else int(setting.default)

    def eff(setting: Setting[Any], cpu_default: int, plan_value: int | None = None) -> int:
        """One knob's effective value: user-set wins, else rung two."""
        value = _get(sn, setting)
        if value != setting.default:
            return int(value)  # user-set — the plan never overrides it
        return rung2(setting, cpu_default, plan_value)

    # Output reserve: user-set (nonzero) wins; else the plan's (0 at full —
    # nothing is reserved when nothing is budgeted). The window covers
    # prompt + completion, so the output cap must fit inside the reserve:
    # clamp the ladder's max_output_tokens down to it when a plan is active
    # and the user did not set the output cap explicitly.
    output_reserve = int(_get(sn, ANSWER_AGENT_OUTPUT_RESERVE_TOKENS))
    if output_reserve == 0 and plan is not None:
        output_reserve = plan.output_reserve

    max_output_tokens = eff(
        ANSWER_AGENT_MAX_OUTPUT_TOKENS,
        CPU_ECONOMY_DEFAULTS.max_output_tokens,
        plan.max_output_tokens if plan else None,
    )
    if (
        plan is not None
        and _get(sn, ANSWER_AGENT_MAX_OUTPUT_TOKENS) == ANSWER_AGENT_MAX_OUTPUT_TOKENS.default
        and output_reserve > 0
    ):
        max_output_tokens = min(max_output_tokens, output_reserve)

    if plan is None:
        compact_prompt = active and bool(_get(sn, ANSWER_AGENT_COMPACT_PROMPT))
    else:
        # A plan that wants the compact prompt takes it; elsewhere a user-set
        # True still wins over the plan's False — allowing explicit opt-in to
        # compact prompts.
        compact_prompt = bool(_get(sn, ANSWER_AGENT_COMPACT_PROMPT)) or plan.compact_prompt

    # By benchmark measurement, aging under a windowed plan is a net loss, so
    # an ACTIVE plan's rung two is -1 — the off marker, normalized to 0 by the
    # agent path — and only an explicit user value binds. Without a plan the
    # knob defaults to 0 (off), so the marker never leaks into a non-windowed
    # budget.
    age_tool_chars = eff(
        ANSWER_AGENT_AGE_TOOL_CHARS,
        CPU_ECONOMY_DEFAULTS.age_tool_chars,
        -1 if plan is not None else None,
    )
    if plan is None and age_tool_chars < 0:
        age_tool_chars = 0

    return EconomyBudget(
        preseed_passages=eff(
            ANSWER_AGENT_PRESEED_PASSAGES,
            CPU_ECONOMY_DEFAULTS.preseed_passages,
            plan.preseed_passages if plan else None,
        ),
        preseed_passage_max_chars=eff(
            ANSWER_AGENT_PRESEED_PASSAGE_MAX_CHARS,
            CPU_ECONOMY_DEFAULTS.preseed_passage_max_chars,
            plan.preseed_passage_max_chars if plan else None,
        ),
        read_max_chars=eff(
            ANSWER_AGENT_READ_MAX_CHARS,
            CPU_ECONOMY_DEFAULTS.read_max_chars,
            plan.read_max_chars if plan else None,
        ),
        max_output_tokens=max_output_tokens,
        tool_budget_chars=eff(
            ANSWER_AGENT_TOOL_BUDGET_CHARS,
            CPU_ECONOMY_DEFAULTS.tool_budget_chars,
        ),
        age_tool_chars=age_tool_chars,
        search_entries=eff(
            ANSWER_AGENT_SEARCH_ENTRIES,
            CPU_ECONOMY_DEFAULTS.search_entries,
        ),
        search_snippet_chars=eff(
            ANSWER_AGENT_SEARCH_SNIPPET_CHARS,
            CPU_ECONOMY_DEFAULTS.search_snippet_chars,
        ),
        compact_prompt=compact_prompt,
        window_tokens=window,
        output_reserve=output_reserve if window > 0 else 0,
        # User-set token tool budget binds; 0 = derive per turn (the window
        # arithmetic in api/agent_chat fills it once the pre-seed exists).
        tool_budget_tokens=int(_get(sn, ANSWER_AGENT_TOOL_BUDGET_TOKENS)),
    )


__all__ = ["CONTEXT_PLANS", "CPU_ECONOMY_DEFAULTS", "EconomyBudget", "resolve_budget"]
