"""Incremental semantic index stored in SQLite (vectors as float16 blobs)."""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .embed import MODEL_DIR, Embedder
from .extract import extract

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index.db"
CONFIG_PATH = ROOT / "config.json"
CHUNK_TOKENS = 220   # leaves room for the "passage: " prefix and special tokens within SEQ_LEN
OVERLAP_TOKENS = 40
CLOUD_ONLY = 0x400000 | 0x1000  # OneDrive placeholders: reading them would trigger a download

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER,
                                  n_chunks INTEGER, error TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, chunk_no INTEGER,
                                   text TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, timeout=30)
    # WAL lets searches read while an index build is writing.
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def walk(cfg: dict):
    """Yield (path, stat) for every file the config says to index."""
    exts = {e.lower() for e in cfg["doc_ext"] + cfg["code_ext"]}
    excluded = {d.lower() for d in cfg["exclude_dirs"]}
    max_bytes = cfg["max_file_mb"] * 1024 * 1024
    for root in cfg["roots"]:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in excluded
                           and not (cfg["skip_hidden_dirs"] and d.startswith("."))]
            for f in filenames:
                if os.path.splitext(f)[1].lower() not in exts:
                    continue
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p, follow_symlinks=False)
                except OSError:
                    continue
                if getattr(st, "st_file_attributes", 0) & CLOUD_ONLY or st.st_size > max_bytes:
                    continue
                yield p, st


def chunk(text: str, tok: Tokenizer) -> list[str]:
    enc = tok.encode(text, add_special_tokens=False)
    offsets = enc.offsets
    if not offsets:
        return []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    out = []
    for start in range(0, len(offsets), step):
        end = min(start + CHUNK_TOKENS, len(offsets))
        piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if piece:
            out.append(piece)
        if end == len(offsets):
            break
    return out


def build(device: str | None = None, verbose: bool = True) -> dict:
    cfg = load_config()
    db = connect()
    known = {p: (m, s) for p, m, s in db.execute("SELECT path, mtime, size FROM files")}
    seen, todo = set(), []
    for p, st in walk(cfg):
        seen.add(p)
        if known.get(p) != (st.st_mtime, st.st_size):
            todo.append((p, st))

    removed = [p for p in known if p not in seen]
    for p in removed:
        db.execute("DELETE FROM chunks WHERE path=?", (p,))
        db.execute("DELETE FROM files WHERE path=?", (p,))
    db.commit()

    stats = {"scanned": len(seen), "changed": len(todo), "removed": len(removed),
             "chunks_added": 0, "errors": 0}
    if not todo:
        return stats

    emb = Embedder(device or cfg["device"])
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    t0 = time.time()
    for i, (p, st) in enumerate(todo, 1):
        error, pieces = None, []
        try:
            pieces = chunk(extract(Path(p), cfg["max_chars_per_file"]), tok)
        except Exception as ex:  # unreadable / encrypted / malformed files are recorded and skipped
            error = f"{type(ex).__name__}: {ex}"[:300]
            stats["errors"] += 1
        db.execute("DELETE FROM chunks WHERE path=?", (p,))
        if pieces:
            vecs = emb.embed_passages(pieces).astype(np.float16)
            db.executemany("INSERT INTO chunks (path, chunk_no, text, vec) VALUES (?,?,?,?)",
                           [(p, n, t, v.tobytes()) for n, (t, v) in enumerate(zip(pieces, vecs))])
            stats["chunks_added"] += len(pieces)
        db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)",
                   (p, st.st_mtime, st.st_size, len(pieces), error, time.time()))
        if i % 20 == 0 or i == len(todo):
            db.commit()
            if verbose:
                print(f"[{i}/{len(todo)}] {stats['chunks_added']} chunks, {time.time() - t0:.0f}s",
                      file=sys.stderr, flush=True)
    db.commit()
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


class Searcher:
    """Keeps the vector matrix in memory and reloads it when the index changes."""

    def __init__(self, device: str | None = None):
        self.emb = Embedder(device or load_config()["device"])
        self._version = None

    def _load(self):
        db = connect()
        version = db.execute("SELECT COUNT(*), MAX(indexed_at) FROM files").fetchone()
        if version != self._version:
            rows = db.execute("SELECT id, vec FROM chunks").fetchall()
            self.ids = np.array([r[0] for r in rows], dtype=np.int64)
            self.mat = (np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16)
                        .reshape(len(rows), -1).astype(np.float32) if rows else np.zeros((0, 384)))
            self._version = version
        return db

    def search(self, query: str, k: int = 5, path_filter: str | None = None) -> list[dict]:
        db = self._load()
        if not len(self.ids):
            return []
        scores = self.mat @ self.emb.embed_query(query)
        results, seen_files = [], set()
        for idx in np.argsort(-scores):
            cid = int(self.ids[idx])
            path, chunk_no, text = db.execute(
                "SELECT path, chunk_no, text FROM chunks WHERE id=?", (cid,)).fetchone()
            if path_filter and path_filter.lower() not in path.lower():
                continue
            if path in seen_files:  # best chunk per file only
                continue
            seen_files.add(path)
            results.append({"path": path, "score": round(float(scores[idx]), 4),
                            "chunk": chunk_no, "text": text})
            if len(results) >= k:
                break
        return results


def status() -> dict:
    db = connect()
    files, chunks = db.execute("SELECT COUNT(*), COALESCE(SUM(n_chunks),0) FROM files").fetchone()
    errors = db.execute("SELECT COUNT(*) FROM files WHERE error IS NOT NULL").fetchone()[0]
    last = db.execute("SELECT MAX(indexed_at) FROM files").fetchone()[0]
    return {"files": files, "chunks": chunks, "errors": errors,
            "last_indexed": time.strftime("%Y-%m-%d %H:%M", time.localtime(last)) if last else None,
            "db_mb": round(DB_PATH.stat().st_size / 1048576, 1) if DB_PATH.exists() else 0}
