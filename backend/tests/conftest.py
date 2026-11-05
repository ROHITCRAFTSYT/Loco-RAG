"""Test-session setup.

Point the app's SQLite DB and vector-store dirs at a throwaway temp directory
*before* any ``app`` module is imported (``app.db`` creates its engine at import
time), so tests never touch ``backend/data``.
"""
from __future__ import annotations

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="loco-rag-tests-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/app.db")
os.environ.setdefault("CHROMA_DIR", f"{_tmp}/chroma")
os.environ.setdefault("LANCEDB_DIR", f"{_tmp}/lancedb")
