"""
聊天灵感插件 - 数据库层
使用同步 sqlite3，线程安全，无额外依赖
- FTS5 全文搜索
- 消息去重 (message_id)
- 数据清理 & 过期
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional


class InspirationDB:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def init(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT DEFAULT '',
                group_id TEXT NOT NULL,
                group_name TEXT DEFAULT '',
                user_id TEXT NOT NULL,
                user_name TEXT DEFAULT '',
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                platform TEXT DEFAULT '',
                categorized INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_messages_group
                ON messages(group_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_messages_categorized
                ON messages(group_id, categorized);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_dedup
                ON messages(group_id, message_id)
                WHERE message_id != '';

            CREATE TABLE IF NOT EXISTS inspirations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source_message_ids TEXT DEFAULT '[]',
                category TEXT DEFAULT '未分类',
                created_at REAL NOT NULL,
                source_range_start REAL,
                source_range_end REAL
            );
            CREATE INDEX IF NOT EXISTS idx_inspirations_group
                ON inspirations(group_id, created_at);

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                summary_type TEXT NOT NULL DEFAULT 'daily',
                start_time REAL,
                end_time REAL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_group
                ON summaries(group_id, created_at);

            CREATE TABLE IF NOT EXISTS categorize_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                inspiration_count INTEGER DEFAULT 0,
                trigger_type TEXT DEFAULT 'auto',
                created_at REAL NOT NULL
            );
        """)
        # FTS5 全文索引（带触发器自动同步）
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, user_name, group_id,
                content=messages, content_rowid=id,
                tokenize='unicode61 remove_diacritics 1'
            );

            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, user_name, group_id)
                VALUES (new.id, new.content, new.user_name, new.group_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, user_name, group_id)
                VALUES ('delete', old.id, old.content, old.user_name, old.group_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, user_name, group_id)
                VALUES ('delete', old.id, old.content, old.user_name, old.group_id);
                INSERT INTO messages_fts(rowid, content, user_name, group_id)
                VALUES (new.id, new.content, new.user_name, new.group_id);
            END;
        """)
        conn.commit()

    # ---------- 消息存储（带去重） ----------

    def store_message(
        self,
        group_id: str,
        group_name: str,
        user_id: str,
        user_name: str,
        content: str,
        platform: str = "",
        message_id: str = "",
    ) -> Optional[int]:
        """存储消息，支持 message_id 去重；重复返回 None"""
        conn = self._get_conn()
        if message_id:
            exist = conn.execute(
                "SELECT id FROM messages WHERE group_id=? AND message_id=?",
                (group_id, message_id),
            ).fetchone()
            if exist:
                return None  # 重复，跳过
        cursor = conn.execute(
            "INSERT INTO messages (message_id, group_id, group_name, user_id, user_name, content, timestamp, platform) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (message_id, group_id, group_name, user_id, user_name, content, time.time(), platform),
        )
        conn.commit()
        return cursor.lastrowid

    def get_uncategorized_count(self, group_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE group_id=? AND categorized=0", (group_id,)
        ).fetchone()
        return row[0] if row else 0

    def get_uncategorized_messages(self, group_id: str, limit: int = 200) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE group_id=? AND categorized=0 ORDER BY timestamp ASC LIMIT ?",
            (group_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_messages_categorized(self, message_ids: list[int]):
        if not message_ids:
            return
        conn = self._get_conn()
        ph = ",".join("?" * len(message_ids))
        conn.execute(f"UPDATE messages SET categorized=1 WHERE id IN ({ph})", message_ids)
        conn.commit()

    # ---------- FTS5 全文搜索 ----------

    def search_messages_fts(
        self,
        group_id: str,
        keywords: list[str],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 300,
    ) -> list[dict]:
        """多关键词 FTS5 全文搜索。每个关键词用 OR 连接"""
        conn = self._get_conn()
        if not keywords:
            return []
        # 每个词用 FTS5 语法包裹： "keyword"
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        sql = """
            SELECT m.* FROM messages m
            JOIN messages_fts fts ON m.id = fts.rowid
            WHERE messages_fts MATCH ? AND m.group_id = ?
        """
        params: list = [fts_query, group_id]
        if start_time is not None:
            sql += " AND m.timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            sql += " AND m.timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY m.timestamp ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_messages_in_range(
        self, group_id: str, start_time: float, end_time: float, limit: int = 500
    ) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE group_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC LIMIT ?",
            (group_id, start_time, end_time, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_user_messages_in_range(
        self, group_id: str, user_id: str, start_time: float, end_time: float, limit: int = 500
    ) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE group_id=? AND user_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC LIMIT ?",
            (group_id, user_id, start_time, end_time, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 灵感 ----------

    def store_inspiration(
        self, group_id: str, content: str, source_message_ids: list[int],
        category: str = "未分类",
        source_range_start: Optional[float] = None,
        source_range_end: Optional[float] = None,
    ) -> int:
        conn = self._get_conn()
        c = conn.execute(
            "INSERT INTO inspirations (group_id,content,source_message_ids,category,created_at,source_range_start,source_range_end) VALUES (?,?,?,?,?,?,?)",
            (group_id, content, json.dumps(source_message_ids, ensure_ascii=False), category, time.time(), source_range_start, source_range_end),
        )
        conn.commit()
        return c.lastrowid

    def query_inspirations(
        self, group_id: str,
        start_time: Optional[float] = None, end_time: Optional[float] = None,
        category: Optional[str] = None, limit: int = 50,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        conn = self._get_conn()
        sql = "SELECT * FROM inspirations WHERE group_id=?"
        params: list = [group_id]
        if start_time is not None:
            sql += " AND created_at>=?"
            params.append(start_time)
        if end_time is not None:
            sql += " AND created_at<=?"
            params.append(end_time)
        if category is not None:
            sql += " AND category=?"
            params.append(category)
        if user_id is not None:
            sql += " AND source_message_ids LIKE ?"
            params.append(f'%{user_id}%')
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ---------- 总结 ----------

    def store_summary(
        self, group_id: str, summary_type: str, content: str,
        start_time: Optional[float] = None, end_time: Optional[float] = None,
    ) -> int:
        conn = self._get_conn()
        c = conn.execute(
            "INSERT INTO summaries (group_id,summary_type,start_time,end_time,content,created_at) VALUES (?,?,?,?,?,?)",
            (group_id, summary_type, start_time, end_time, content, time.time()),
        )
        conn.commit()
        return c.lastrowid

    def get_summaries(self, group_id: str, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM summaries WHERE group_id=? ORDER BY created_at DESC LIMIT ?", (group_id, limit)
        ).fetchall()]

    def has_daily_summary_today(self, group_id: str) -> bool:
        """检查今天是否已有每日总结"""
        conn = self._get_conn()
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        row = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE group_id=? AND summary_type='daily' AND created_at>=?",
            (group_id, today_start),
        ).fetchone()
        return (row[0] or 0) > 0

    def get_summary_by_range(
        self, group_id: str, start_time: float, end_time: float
    ) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content FROM summaries WHERE group_id=? AND summary_type='daily' AND start_time=? AND end_time=?",
            (group_id, start_time, end_time),
        ).fetchone()
        return row[0] if row else None

    # ---------- 日志 ----------

    def log_categorize(
        self, group_id: str, message_count: int, inspiration_count: int, trigger_type: str = "auto"
    ):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO categorize_log (group_id,message_count,inspiration_count,trigger_type,created_at) VALUES (?,?,?,?,?)",
            (group_id, message_count, inspiration_count, trigger_type, time.time()),
        )
        conn.commit()

    # ---------- 统计 ----------

    def get_data_stats(self, group_id: str) -> dict:
        conn = self._get_conn()
        msg_row = conn.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN categorized=0 THEN 1 ELSE 0 END) as uncategorized FROM messages WHERE group_id=?",
            (group_id,),
        ).fetchone()
        insp_row = conn.execute("SELECT COUNT(*) FROM inspirations WHERE group_id=?", (group_id,)).fetchone()
        summ_row = conn.execute("SELECT COUNT(*) FROM summaries WHERE group_id=?", (group_id,)).fetchone()
        last_cat = conn.execute(
            "SELECT * FROM categorize_log WHERE group_id=? ORDER BY created_at DESC LIMIT 1", (group_id,)
        ).fetchone()
        return {
            "total_messages": msg_row[0] if msg_row else 0,
            "uncategorized_messages": msg_row[1] if msg_row else 0,
            "total_inspirations": insp_row[0] if insp_row else 0,
            "total_summaries": summ_row[0] if summ_row else 0,
            "last_categorize": {
                "message_count": last_cat[1] if last_cat else 0,
                "inspiration_count": last_cat[2] if last_cat else 0,
                "trigger_type": last_cat[3] if last_cat else "",
                "time": last_cat[4] if last_cat else None,
            },
        }

    # ---------- 数据清理 ----------

    def delete_messages_range(self, group_id: str, start_time: float, end_time: float) -> int:
        conn = self._get_conn()
        c = conn.execute(
            "DELETE FROM messages WHERE group_id=? AND timestamp>=? AND timestamp<=?",
            (group_id, start_time, end_time),
        )
        conn.commit()
        return c.rowcount

    def cleanup_expired(self, group_id: str, retention_days: int) -> dict:
        """清理超过 N 天的消息，保留灵感和总结"""
        cutoff = time.time() - retention_days * 86400
        conn = self._get_conn()
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE group_id=? AND timestamp<?",
            (group_id, cutoff),
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM messages WHERE group_id=? AND timestamp<?",
            (group_id, cutoff),
        )
        conn.commit()
        return {"deleted_messages": msg_count, "cutoff_time": datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d %H:%M")}

    # ---------- 数据导出 ----------

    def export_all(self, group_id: str) -> dict:
        conn = self._get_conn()
        return {
            "group_id": group_id,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [dict(r) for r in conn.execute("SELECT * FROM messages WHERE group_id=? ORDER BY timestamp ASC", (group_id,)).fetchall()],
            "inspirations": [dict(r) for r in conn.execute("SELECT * FROM inspirations WHERE group_id=? ORDER BY created_at ASC", (group_id,)).fetchall()],
            "summaries": [dict(r) for r in conn.execute("SELECT * FROM summaries WHERE group_id=? ORDER BY created_at ASC", (group_id,)).fetchall()],
            "categorize_log": [dict(r) for r in conn.execute("SELECT * FROM categorize_log WHERE group_id=? ORDER BY created_at ASC", (group_id,)).fetchall()],
        }

    def export_range(self, group_id: str, start_time: float, end_time: float) -> dict:
        conn = self._get_conn()
        return {
            "group_id": group_id,
            "range_start": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "range_end": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [dict(r) for r in conn.execute("SELECT * FROM messages WHERE group_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC", (group_id, start_time, end_time)).fetchall()],
            "inspirations": [dict(r) for r in conn.execute("SELECT * FROM inspirations WHERE group_id=? AND created_at>=? AND created_at<=? ORDER BY created_at ASC", (group_id, start_time, end_time)).fetchall()],
            "summaries": [dict(r) for r in conn.execute("SELECT * FROM summaries WHERE group_id=? AND created_at>=? AND created_at<=? ORDER BY created_at ASC", (group_id, start_time, end_time)).fetchall()],
        }

    def get_user_ids(self, group_id: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT DISTINCT user_id FROM messages WHERE group_id=?", (group_id,)).fetchall()
        return [r[0] for r in rows]
