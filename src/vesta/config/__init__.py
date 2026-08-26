"""Configuration package: settings registry + capabilities.

Importing this package declares core settings into the registry, which is what
makes ``GET /api/settings/schema`` describe the whole tuning surface without
any per-knob wiring.

Public surface re-exported here so callers do ``from vesta.config import get``.
"""

from __future__ import annotations

from vesta.config import capabilities, resolution, settings
from vesta.config.capabilities import (
    Capability,
    CapabilitySet,
    compute_capabilities,
    register_probe,
)
from vesta.config.resolution import (
    configure,
    get,
    get_or_default,
    reset_for_test,
    set_db_values,
    snapshot,
    validate_and_coerce,
)
from vesta.config.settings import (
    SECRET_MASK,
    Setting,
    SettingSchema,
    SettingsSnapshot,
    all_settings,
    is_secret,
    redact_values,
    schema,
    setting,
    strip_secret_values,
)

# ── Server & Storage settings ─────────────────────────────────────────────
# Each declaration carries its own UI metadata so the settings schema endpoint
# needs no extra wiring. Bounds where a number would be nonsensical.

SERVER_HOST = setting(
    "server.host",
    str,
    "127.0.0.1",
    group="Server",
    help="Network interface to bind. 127.0.0.1 keeps an unauthenticated app local; "
    "set 0.0.0.0 only on a trusted network (no auth layer exists yet).",
    hot=False,  # requires restart
)
SERVER_PORT = setting(
    "server.port",
    int,
    8080,
    group="Server",
    help="TCP port to listen on.",
    min=1,
    max=65535,
    hot=False,  # requires restart
)
DATA_DIR = setting(
    "data.dir",
    str,
    "./data",
    group="Storage",
    help="Path to the data directory holding zims/, models/, vesta.db, cache/.",
    hot=False,
)
DB_BUSY_TIMEOUT_MS = setting(
    "db.busy_timeout_ms",
    int,
    5000,
    group="Storage",
    help="How long SQLite waits for a write lock before returning SQLITE_BUSY. "
    "Protects per-second job-progress writes under WAL.",
    min=0,
    max=60000,
    hot=False,
)
JOBS_MAX_CONCURRENT_NOOP = setting(
    "jobs.max_concurrent.noop",
    int,
    1,
    group="Jobs",
    help="Max concurrently-running jobs of type 'noop'. Other types declare their "
    "own jobs.max_concurrent.<type> setting as they land.",
    min=1,
    max=32,
    hot=True,
)
LOGGING_LEVEL = setting(
    "logging.level",
    str,
    "INFO",
    group="Logging",
    help="Minimum severity for structured JSON logs.",
    choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    hot=True,
)

# ── ZIM & Query settings ──────────────────────────────────────────────────
# Each knob carries its own UI metadata; the schema endpoint needs no per-knob
# wiring. Bounds where a number would be nonsensical.

ZIM_CLUSTER_CACHE_MB = setting(
    "zim.cluster_cache_mb",
    int,
    256,
    group="ZIM / Storage",
    help="libzim's cluster cache is GLOBAL across all open archives and defaults to "
    "16 MB, which thrashes on multi-archive fan-out. Raise at startup.",
    min=16,
    max=4096,
    hot=False,  # applied once at startup
)
ZIM_READ_POOL_SIZE = setting(
    "zim.read_pool_size",
    int,
    4,
    group="ZIM / Storage",
    help="Bounded thread pool for blocking libzim reads/extracts. Single user — "
    "threads scale negatively past this so keep it modest.",
    min=1,
    max=32,
    hot=False,
)
QUERY_STOPWORDS_LIST = setting(
    "query.stopwords.list",
    str,
    (
        "a,an,the,and,or,of,to,in,on,at,by,for,with,from,into,is,are,was,were,be,been,"
        "being,do,does,did,doing,have,has,had,i,you,he,she,it,we,they,this,that,these,"
        "those,my,your,me,him,her,us,them,how,what,why,who,whom,when,where,which,whose,"
        "explain,tell,about,there,here"
    ),
    group="Query / Preprocessing",
    help="Comma-separated stopword + interrogative list. Query preprocessing "
    "uses its own fixed list; this one budgets the chat read_article tool's "
    "focused view.",
    hot=True,
)

__all__ = [
    "DATA_DIR",
    "DB_BUSY_TIMEOUT_MS",
    "JOBS_MAX_CONCURRENT_NOOP",
    "LOGGING_LEVEL",
    "QUERY_STOPWORDS_LIST",
    "SECRET_MASK",
    "SERVER_HOST",
    "SERVER_PORT",
    "ZIM_CLUSTER_CACHE_MB",
    "ZIM_READ_POOL_SIZE",
    "Capability",
    "CapabilitySet",
    "Setting",
    "SettingSchema",
    "SettingsSnapshot",
    "all_settings",
    "capabilities",
    "compute_capabilities",
    "configure",
    "get",
    "get_or_default",
    "is_secret",
    "redact_values",
    "register_probe",
    "reset_for_test",
    "resolution",
    "schema",
    "set_db_values",
    "setting",
    "settings",
    "snapshot",
    "strip_secret_values",
    "validate_and_coerce",
]
