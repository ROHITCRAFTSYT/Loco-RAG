"""Document parsing dispatch and text decoding.

parse() routes by filename/content-type to the PDF/DOCX/text readers; the text
path must survive non-UTF-8 bytes and treat whitespace-only input as empty. PDF
and DOCX readers need their optional libs, so they're covered indirectly — this
pins the routing and the always-available text path.
"""
from __future__ import annotations

from app.services.ingest import _read_text, parse


def test_read_text_decodes_utf8():
    pages = _read_text("hello world".encode("utf-8"))
    assert pages == [(None, "hello world")]


def test_read_text_replaces_invalid_bytes_without_crashing():
    pages = _read_text(b"\xff\xfe not valid utf8 \xff")
    assert len(pages) == 1
    assert isinstance(pages[0][1], str)


def test_read_text_whitespace_only_is_empty():
    assert _read_text(b"   \n\t ") == []


def test_parse_routes_plain_text_by_extension():
    pages = parse("notes.md", None, b"# heading\ntext")
    assert pages and pages[0][0] is None
    assert "heading" in pages[0][1]


def test_parse_defaults_unknown_extension_to_text():
    # csv/code/etc. fall through to the text reader rather than erroring.
    pages = parse("data.csv", None, b"a,b,c")
    assert pages == [(None, "a,b,c")]
