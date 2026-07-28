"""Content fingerprint used for chunk provenance / re-upload detection."""
from __future__ import annotations

import hashlib

from app.services.ingest import content_sha256


def test_matches_hashlib_and_is_hex():
    data = b"hello world"
    h = content_sha256(data)
    assert h == hashlib.sha256(data).hexdigest()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_identical_bytes_hash_equal():
    assert content_sha256(b"same") == content_sha256(b"same")


def test_different_bytes_hash_differ():
    assert content_sha256(b"a") != content_sha256(b"b")


def test_empty_input():
    assert content_sha256(b"") == hashlib.sha256(b"").hexdigest()
