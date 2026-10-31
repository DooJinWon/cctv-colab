"""SQLite keeps long-video frame metadata off the Python heap."""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
          CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS frames (idx INTEGER PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY, first_frame INTEGER, last_frame INTEGER, observations INTEGER);
          CREATE TABLE IF NOT EXISTS events (frame INTEGER, kind TEXT, detail TEXT);
          CREATE INDEX IF NOT EXISTS event_frame ON events(frame);
        """
        )

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, record):
        idx = record["index"]
        self.db.execute("INSERT OR REPLACE INTO frames VALUES (?, ?)", (idx, json.dumps(record)))
        for person in record["people"]:
            sid = person.get("segment_id")
            if sid:
                self.db.execute(
                    """INSERT INTO segments VALUES (?, ?, ?, 1)
                  ON CONFLICT(id) DO UPDATE SET last_frame=excluded.last_frame,
                    observations=segments.observations+1""",
                    (sid, idx, idx),
                )
        for kind in sorted(set(record["flags"])):
            self.db.execute("INSERT INTO events VALUES (?, ?, ?)", (idx, kind, ""))

    def get(self, index):
        row = self.db.execute("SELECT payload FROM frames WHERE idx=?", (int(index),)).fetchone()
        if not row:
            raise ValueError(
                f"분석 결과에 {index}번 프레임이 없습니다. 분석 완료 여부를 확인하세요."
            )
        return json.loads(row[0])

    def records(self):
        for row in self.db.execute("SELECT payload FROM frames ORDER BY idx"):
            yield json.loads(row[0])

    def segments(self):
        return [
            {"id": r[0], "first_frame": r[1], "last_frame": r[2], "observations": r[3]}
            for r in self.db.execute("SELECT * FROM segments ORDER BY first_frame, id")
        ]

    def events(self, limit=300):
        return [
            {"frame": r[0], "kind": r[1]}
            for r in self.db.execute(
                "SELECT frame, kind FROM events ORDER BY frame LIMIT ?", (int(limit),)
            )
        ]

    def event_counts(self):
        return dict(self.db.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind"))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()
