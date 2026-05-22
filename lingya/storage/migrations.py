MIGRATIONS: list[str] = [
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );""",
    """CREATE TABLE IF NOT EXISTS personality (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        data JSON NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );""",
]
