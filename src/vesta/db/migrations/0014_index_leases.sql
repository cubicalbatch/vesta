-- Migration 0014 — cross-process index-build leases (AUDIT_0822 M7).
--
-- Detached CLI indexing (`vesta index`, runs in its own process) and
-- server-side indexing (POST /api/zims/{id}/index → JobRunner → IndexZimJob)
-- write the SAME tables for one zim_id — articles, chunks, vectors_dN — and
-- both stamp zims.index_*. Until now their only mutual exclusion was
-- in-process (the runner's job-row scan, the CLI's cancel of stranded rows),
-- so neither could see the other and two builders could interleave partial
-- batches. One row here = "some process holds the right to build this
-- archive's index".
--
-- Deployment is one container on one host, so an OS pid is a valid liveness
-- signal: the next acquirer probes the holder's pid with os.kill(pid, 0). A
-- dead pid — or a lease older than the generous staleness ceiling in
-- vesta/index/leases.py (reboot / pid-reuse escape hatch) — is taken over; a
-- live foreign holder fails the claim fast (CLI exits nonzero, API answers
-- 409). Claimed/released around EVERY build by both entry points via
-- vesta/index/leases.py; a hard-killed owner just leaves a dead-pid row that
-- the next acquirer takes.
--
-- FK-cascades with the zims row so deleting an archive cannot leak its lease.

CREATE TABLE index_leases (
    zim_id      INTEGER PRIMARY KEY REFERENCES zims(id) ON DELETE CASCADE,
    owner_id    TEXT    NOT NULL,  -- who holds it: 'cli' | 'server' | ...
    pid         INTEGER NOT NULL,  -- OS pid of the holding process (liveness probe)
    acquired_at TEXT    NOT NULL   -- UTC ISO-8601 (jobs-table timestamp convention)
);
