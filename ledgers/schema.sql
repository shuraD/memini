-- Memini ledger (owner 2026-11-12): SQL text committed with the code, loaded into
-- SQLite for checks. One INSERT per line in features.sql / bugs.sql / changes.sql.

CREATE TABLE feature (
  id           TEXT PRIMARY KEY CHECK (id GLOB 'F-*-[0-9][0-9][0-9]'),  -- F-<module>-NNN: module-scoped so parallel worktrees never collide
  description  TEXT NOT NULL,
  layer        TEXT NOT NULL CHECK (layer IN ('infra', 'core', 'extensions')),
  proof        TEXT NOT NULL,               -- paper section or owner decision
  -- active: built and tested | planned: a deferred requirement owned by a module (no test yet) | superseded
  status       TEXT NOT NULL CHECK (status IN ('active', 'planned', 'superseded')),
  owner_module TEXT CHECK (status <> 'planned' OR owner_module IS NOT NULL),
  added_in     TEXT NOT NULL                -- change id
);

CREATE TABLE feature_test (
  feature_id TEXT NOT NULL REFERENCES feature (id),
  test_id    TEXT NOT NULL,                -- pytest node id
  PRIMARY KEY (feature_id, test_id)
);

CREATE TABLE bug (
  id              TEXT PRIMARY KEY CHECK (id GLOB 'B-*-[0-9][0-9][0-9]'),  -- B-<module>-NNN
  description     TEXT NOT NULL,
  root_cause      TEXT,
  status          TEXT NOT NULL CHECK (status IN ('open', 'fixed', 'withdrawn')),
  regression_test TEXT,                    -- pytest node id (required when fixed)
  fixed_in        TEXT,                    -- change id
  -- a row that turned out wrong is never deleted; it is withdrawn with its reason
  -- (owner 2026-11-12: "the history should be unchanged and then ... withdrawn")
  withdrawn_reason TEXT CHECK (
    status <> 'withdrawn' OR (withdrawn_reason IS NOT NULL AND trim(withdrawn_reason) <> '')
  )
);

CREATE TABLE change (
  id       TEXT PRIMARY KEY,               -- C-YYYYMMDD-NNNN
  date     TEXT NOT NULL,
  author   TEXT NOT NULL,                  -- human, Claude, Agy, OpenCode, ...
  module   TEXT NOT NULL,
  summary  TEXT NOT NULL,
  features TEXT,                           -- comma-separated feature ids touched
  bugs     TEXT,                           -- comma-separated bug ids touched
  checks   TEXT,                           -- pipeline result summary
  review   TEXT,                           -- review report path
  approval TEXT                            -- OWNER-APPROVED-REMOVAL ref, if any
);
