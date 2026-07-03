MIGRATIONS: list[str] = [
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );""",
    # v4–v7: tables from earlier iterations (conversations/turns removed as dead code)
    """CREATE TABLE IF NOT EXISTS personality (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        data JSON NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );""",
    """CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS reflection_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        old_personality JSON,
        new_personality JSON NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );""",
    """CREATE TABLE IF NOT EXISTS mind_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );""",
]
