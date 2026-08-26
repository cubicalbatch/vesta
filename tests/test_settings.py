"""Settings: declaration, layered resolution, schema, snapshot, coercion."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from vesta import config
from vesta.config.settings import Setting, SettingSchema


def test_judge_inference_settings_declared() -> None:
    """The judge LLM has its own endpoint/model/api-key group."""
    items = {i.key: i for i in config.schema()}
    for key in ("eval.judge.endpoint_url", "eval.judge.api_key", "eval.judge.model"):
        assert key in items, f"{key} must be declared"
        assert items[key].group == "Judge Inference / LLM"
        assert items[key].type == "string"
        assert items[key].hot is True


def test_chat_history_and_trace_settings_declared() -> None:
    """Chat history and trace retention knobs are registered with defaults and bounds."""
    import vesta.answer  # noqa: F401 — importing registers the chat settings

    schema = {s.key: s for s in config.schema()}
    expected: dict[str, tuple[object, str]] = {
        "chat.history.max_turns": (10, "integer"),
        "chat.trace_retention_days": (7, "integer"),
    }
    for key, (default, typ) in expected.items():
        assert key in schema, f"{key} not registered"
        entry = schema[key]
        assert entry.default == default, f"{key}: default {entry.default!r} != {default!r}"
        assert entry.type == typ, f"{key}: type {entry.type!r} != {typ!r}"
        assert entry.hot is True, f"{key}: should be hot (live-tunable)"
        assert entry.group, f"{key}: missing group"
        assert entry.help, f"{key}: missing help text"

    assert (schema["chat.history.max_turns"].min, schema["chat.history.max_turns"].max) == (1, 50)
    assert schema["chat.trace_retention_days"].min == 0


def test_rerank_truncate_tokens_setting_is_cold() -> None:
    """retrieval.rerank.truncate_tokens is baked into the cached cross-encoder at
    construction time and must be marked hot=False (AUDIT_0824 Z3)."""
    import vesta.encoders  # noqa: F401

    schema = {s.key: s for s in config.schema()}
    assert "retrieval.rerank.truncate_tokens" in schema
    assert schema["retrieval.rerank.truncate_tokens"].hot is False


def test_schema_describes_every_declared_setting() -> None:
    items = config.schema()
    assert len(items) >= 7  # see config/__init__.py
    # Every schema item carries its UI metadata — that's what makes the UI free.
    for item in items:
        assert isinstance(item, SettingSchema)
        assert item.type in {"string", "integer", "float", "boolean"}
        assert item.group
        assert item.help
        assert item.hot in (True, False)
    keys = {i.key for i in items}
    assert {
        "server.host",
        "server.port",
        "data.dir",
        "jobs.max_concurrent.noop",
        "logging.level",
        "db.busy_timeout_ms",
    } <= keys


def test_resolution_default() -> None:
    config.configure(env={})
    assert config.get(config.SERVER_HOST) == "127.0.0.1"
    assert config.get(config.SERVER_PORT) == 8080
    assert config.get(config.LOGGING_LEVEL) == "INFO"


def test_env_overrides_default() -> None:
    config.configure(env={"server.port": "9090", "logging.level": "DEBUG"})
    assert config.get(config.SERVER_PORT) == 9090
    assert config.get(config.LOGGING_LEVEL) == "DEBUG"


def test_settings_table_beats_env() -> None:
    """The table is authoritative over env (UI can change anything)."""
    config.configure(env={"server.port": "9090"}, db_values={"server.port": "7000"})
    assert config.get(config.SERVER_PORT) == 7000


def test_bool_coercion() -> None:
    """Bool settings coerce from text forms (true/1/yes/on → True)."""
    flag = config.setting("test.bool_flag", bool, False, group="Test", help="h")
    config.configure(env={"test.bool_flag": "on"}, db_values={})
    assert config.get(flag) is True
    config.configure(env={"test.bool_flag": "no"}, db_values={})
    assert config.get(flag) is False
    config.configure(env={"test.bool_flag": "1"}, db_values={})
    assert config.get(flag) is True


def test_snapshot_is_immutable_pin() -> None:
    config.configure(env={"server.port": "8080"})
    snap = config.snapshot()
    assert snap.get(config.SERVER_PORT) == 8080
    # A snapshot taken before a change keeps the old value.
    config.configure(env={"server.port": "9999"})
    assert snap.get(config.SERVER_PORT) == 8080  # unchanged — the pin held


def test_snapshot_records_every_setting() -> None:
    config.configure(env={})
    snap = config.snapshot()
    assert set(snap.values) >= {s.key for s in config.schema()}


def test_validate_and_coerce_enforces_bounds() -> None:
    with pytest.raises(ValueError):
        config.validate_and_coerce(config.SERVER_PORT, "0")  # below min 1
    with pytest.raises(ValueError):
        config.validate_and_coerce(config.SERVER_PORT, "99999")  # above max 65535
    assert config.validate_and_coerce(config.SERVER_PORT, "8080") == 8080


def test_validate_and_coerce_enforces_choices() -> None:
    with pytest.raises(ValueError):
        config.validate_and_coerce(config.LOGGING_LEVEL, "LOUD")
    assert config.validate_and_coerce(config.LOGGING_LEVEL, "INFO") == "INFO"
    assert config.validate_and_coerce(config.LOGGING_LEVEL, "DEBUG") == "DEBUG"


def test_setting_is_frozen_and_generic() -> None:
    s = config.SERVER_PORT
    assert isinstance(s, Setting)
    assert s.default == 8080
    with pytest.raises(Exception):
        s.default = 1  # type: ignore[misc]


def test_get_without_configure_raises() -> None:
    config.reset_for_test()
    try:
        with pytest.raises(RuntimeError):
            config.get(config.SERVER_HOST)
    finally:
        config.configure(env={})


# ── Secret redaction (AUDIT_0824 N3) ───────────────────────────────────────


def _register_secret_settings() -> None:
    """Import the owning modules so their settings join the registry."""
    import vesta.eval.golden
    import vesta.inference  # noqa: F401


def test_credential_settings_flagged_secret_in_schema() -> None:
    _register_secret_settings()
    items = {i.key: i for i in config.schema()}
    for key in ("inference.llm.api_key", "eval.judge.api_key"):
        assert key in items, f"{key} must be declared"
        assert items[key].secret is True, f"{key} must be marked secret"
    # Non-credential knobs stay public.
    assert items["server.port"].secret is False


def test_redact_and_strip_helpers_leave_inputs_untouched() -> None:
    _register_secret_settings()
    values: dict[str, object] = {
        "inference.llm.api_key": "sk-live",
        "eval.judge.api_key": "",
        "server.port": 8080,
    }
    assert config.redact_values(values) == {
        "inference.llm.api_key": config.SECRET_MASK,
        "eval.judge.api_key": "",
        "server.port": 8080,
    }
    assert config.strip_secret_values(values) == {
        "server.port": 8080,
    }
    assert values["inference.llm.api_key"] == "sk-live"


@pytest.mark.asyncio
async def test_get_and_put_settings_mask_configured_secrets(
    app_client: httpx.AsyncClient,
) -> None:
    _register_secret_settings()
    resp = await app_client.put(
        "/api/settings",
        json={
            "values": {
                "inference.llm.api_key": "sk-live-inference",
                "eval.judge.api_key": "sk-live-judge",
            }
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["values"]["inference.llm.api_key"] == config.SECRET_MASK
    assert body["values"]["eval.judge.api_key"] == config.SECRET_MASK

    got = (await app_client.get("/api/settings")).json()["values"]
    assert got["inference.llm.api_key"] == config.SECRET_MASK
    assert got["eval.judge.api_key"] == config.SECRET_MASK
    assert "sk-live" not in json.dumps(got)


@pytest.mark.asyncio
async def test_put_blank_or_masked_secret_preserves_stored_value(
    app_client: httpx.AsyncClient,
) -> None:
    _register_secret_settings()
    await app_client.put("/api/settings", json={"values": {"inference.llm.api_key": "sk-keep-me"}})

    def stored_value() -> str | None:
        db_path = Path(os.environ["data.dir"]) / "vesta.db"
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'inference.llm.api_key'"
            ).fetchone()
        return None if row is None else str(row[0])

    # Blank and masked round-trips are no-ops, never wipes.
    for unchanged in ("", config.SECRET_MASK):
        resp = await app_client.put(
            "/api/settings", json={"values": {"inference.llm.api_key": unchanged}}
        )
        assert resp.status_code == 200
        assert "inference.llm.api_key" not in resp.json()["applied"]
        assert stored_value() == "sk-keep-me"

    # A fresh value replaces the stored one.
    resp = await app_client.put(
        "/api/settings", json={"values": {"inference.llm.api_key": "sk-replaced"}}
    )
    assert resp.status_code == 200
    assert stored_value() == "sk-replaced"


def test_eval_run_config_json_strips_secret_snapshot_values() -> None:
    from vesta.eval.metrics import LatencyPercentiles, RunMetrics
    from vesta.eval.runner import RunRecord

    rec = RunRecord(
        id=0,
        started_at="t",
        profile_name="p",
        profile_hash="h",
        profile_yaml="",
        golden_hash="",
        archive_path="",
        archive_checksum="",
        settings_snapshot={
            "inference.llm.api_key": "sk-secret-eval",
            "server.port": 8080,
        },
        git_sha="",
        machine_id="",
        metrics=RunMetrics({}, LatencyPercentiles(), False, (), 0),
        per_query=(),
    )
    cfg = rec.to_config_json()
    assert "sk-secret-eval" not in json.dumps(cfg)
    snapshot = cfg["settings_snapshot"]
    assert isinstance(snapshot, dict)
    # Non-secret pins survive — runs stay comparable.
    assert snapshot["server.port"] == 8080
