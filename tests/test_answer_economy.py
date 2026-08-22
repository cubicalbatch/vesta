"""Unit tests for the agent token-economy resolver (answer/economy.py).

The matrix the contract fixes: economy setting (auto/on/off) by hardware
(cpu/gpu/None) → which budget applies, plus the user-set-wins rule (a knob
whose effective value differs from its registered default always wins over the
economy's CPU defaults; a knob AT its default falls through to the CPU default
when the economy is active).
"""

from __future__ import annotations

from typing import Any

from vesta.answer.economy import CPU_ECONOMY_DEFAULTS, EconomyBudget, resolve_budget
from vesta.config.settings import SettingsSnapshot, all_settings

#: The full-context defaults (the registered setting defaults).
FULL_DEFAULTS = EconomyBudget(
    preseed_passages=6,
    preseed_passage_max_chars=0,
    read_max_chars=0,
    max_output_tokens=4096,
    tool_budget_chars=0,
    age_tool_chars=0,
    search_entries=6,
    search_snippet_chars=400,
    compact_prompt=False,
)

#: The budget an ACTIVE economy resolves to: the CPU knob defaults with every
#: opt-in knob off — identical to the measured iter3 control (1.661M input,
#: strict 0.740), i.e. economy-on defaults now reproduce iter3 semantics.
ACTIVE_DEFAULTS = CPU_ECONOMY_DEFAULTS


def make_snapshot(**overrides: Any) -> SettingsSnapshot:
    """A snapshot holding every setting's registered default plus overrides."""
    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update(overrides)
    return SettingsSnapshot(values=values)


def test_auto_cpu_activates_economy() -> None:
    assert resolve_budget(make_snapshot(), "cpu") == ACTIVE_DEFAULTS


def test_auto_gpu_keeps_full_defaults() -> None:
    assert resolve_budget(make_snapshot(), "gpu") == FULL_DEFAULTS


def test_auto_unknown_hardware_keeps_full_defaults() -> None:
    assert resolve_budget(make_snapshot(), None) == FULL_DEFAULTS


def test_on_forces_economy_regardless_of_hardware() -> None:
    for hardware in ("cpu", "gpu", None):
        assert resolve_budget(make_snapshot(**{"answer.agent.economy": "on"}), hardware) == (
            ACTIVE_DEFAULTS
        ), f"economy=on must force the CPU budget with hardware={hardware!r}"


def test_off_disables_economy_even_on_cpu() -> None:
    assert resolve_budget(make_snapshot(**{"answer.agent.economy": "off"}), "cpu") == FULL_DEFAULTS


def test_no_snapshot_means_defaults_then_hardware_gate() -> None:
    # No snapshot: every knob reads its default, economy reads "auto".
    assert resolve_budget(None, "cpu") == ACTIVE_DEFAULTS
    assert resolve_budget(None, "gpu") == FULL_DEFAULTS
    assert resolve_budget(None, None) == FULL_DEFAULTS


def test_user_set_knobs_win_over_cpu_defaults() -> None:
    sn = make_snapshot(
        **{
            "answer.agent.economy": "on",
            "answer.agent.preseed_passages": 8,
            "answer.agent.preseed_passage_max_chars": 99,
            "answer.agent.read_max_chars": 123,
            "answer.agent.max_output_tokens": 512,
            "answer.agent.tool_budget_chars": 5000,
            "answer.agent.age_tool_chars": 777,
            "answer.agent.search_entries": 4,
            "answer.agent.search_snippet_chars": 200,
        }
    )
    assert resolve_budget(sn, "cpu") == EconomyBudget(8, 99, 123, 512, 5000, 777, 4, 200, False)


def test_knob_set_to_its_default_is_not_user_set() -> None:
    """A knob AT its registered default falls through to the CPU default when
    the economy is active — "is user-set" means differs-from-default, and a
    default-valued knob carries no explicit user intent."""
    sn = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.max_output_tokens": 4096})
    budget = resolve_budget(sn, "cpu")
    assert budget.max_output_tokens == CPU_ECONOMY_DEFAULTS.max_output_tokens == 2048
    assert budget.preseed_passages == 6


def test_user_set_knobs_win_even_when_economy_off() -> None:
    """An explicit knob value wins regardless of the economy gate."""
    sn = make_snapshot(
        **{"answer.agent.read_max_chars": 777, "answer.agent.max_output_tokens": 1024}
    )
    assert resolve_budget(sn, "gpu") == EconomyBudget(6, 0, 777, 1024, 0, 0, 6, 400, False)


def test_tool_budget_matrix() -> None:
    """The turn-level tool-insert budget knob: registered default 0 (off),
    CPU economy default 12000, user-set always wins."""
    # Off (economy inactive): 0 = no cap.
    assert resolve_budget(make_snapshot(), "gpu").tool_budget_chars == 0
    # Economy active (on any hardware): the leaner 12000 cap applies.
    assert (
        resolve_budget(make_snapshot(**{"answer.agent.economy": "on"}), None).tool_budget_chars
        == 12000
    )
    # User-set wins even on CPU.
    sn = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.tool_budget_chars": 3000})
    assert resolve_budget(sn, "cpu").tool_budget_chars == 3000


def test_age_tool_chars_matrix() -> None:
    """The context-aging knob: registered default -1 (off — measured
    aging under the window a net loss; the CPU economy also
    measures better with it off) and CPU economy default 0 too;
    user-set always wins. An ACTIVE window plan resolves the default to the
    -1 off-marker (normalized to 0 by the agent path); an explicit 0 under a
    plan stays 0 — the derive-me signal api/agent_chat acts on."""
    assert resolve_budget(make_snapshot(), "gpu").age_tool_chars == 0
    assert resolve_budget(make_snapshot(**{"answer.agent.economy": "on"}), None).age_tool_chars == 0
    assert (
        resolve_budget(
            make_snapshot(**{"answer.agent.context_profile": "8k"}), "gpu"
        ).age_tool_chars
        == -1
    )
    assert (
        resolve_budget(
            make_snapshot(
                **{"answer.agent.context_profile": "8k", "answer.agent.age_tool_chars": 0}
            ),
            "gpu",
        ).age_tool_chars
        == 0
    )
    sn = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.age_tool_chars": 900})
    assert resolve_budget(sn, "cpu").age_tool_chars == 900


def test_search_shaping_matrix() -> None:
    """The compact-search shaping knobs: registered defaults 6 entries x 400
    chars (the old hard-coded shape). CPU economy defaults reverted to the
    same 6 x 400 (iter6's 5 x 350 destabilized 41 questions into extra tool
    rounds); user-set always wins."""
    assert resolve_budget(make_snapshot(), "gpu").search_entries == 6
    assert resolve_budget(make_snapshot(), "gpu").search_snippet_chars == 400
    active = resolve_budget(make_snapshot(**{"answer.agent.economy": "on"}), None)
    assert (active.search_entries, active.search_snippet_chars) == (6, 400)
    sn = make_snapshot(
        **{
            "answer.agent.economy": "on",
            "answer.agent.search_entries": 7,
            "answer.agent.search_snippet_chars": 250,
        }
    )
    assert (
        resolve_budget(sn, "cpu").search_entries,
        resolve_budget(sn, "cpu").search_snippet_chars,
    ) == (7, 250)


def test_compact_prompt_knob_matrix() -> None:
    """The iter6 compact prompt is opt-in: OFF by default even under an active
    economy (round-stable, the measured control); True only when the economy
    is active AND the user set the knob."""
    assert not resolve_budget(make_snapshot(**{"answer.agent.economy": "on"}), None).compact_prompt
    assert not resolve_budget(None, "cpu").compact_prompt
    assert not resolve_budget(make_snapshot(), "gpu").compact_prompt
    # User-set True wins — but only under an active economy.
    sn = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.compact_prompt": True})
    assert resolve_budget(sn, None).compact_prompt
    sn_off = make_snapshot(**{"answer.agent.economy": "off", "answer.agent.compact_prompt": True})
    assert not resolve_budget(sn_off, "cpu").compact_prompt


def test_partial_snapshot_falls_back_to_defaults() -> None:
    """A snapshot missing the new keys (mid-merge persistence) resolves via the
    registered defaults instead of raising."""
    sn = SettingsSnapshot(values={"answer.agent.economy": "on"})
    assert resolve_budget(sn, None) == ACTIVE_DEFAULTS
