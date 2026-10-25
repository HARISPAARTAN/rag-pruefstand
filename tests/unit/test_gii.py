import io
import urllib.request

import pytest

from pruefstand.corpus.gii import (
    build_units,
    fetch_laws,
    norm_url,
    parse_law_xml,
    read_xml_from_zip,
    unit_id_for,
)
from pruefstand.errors import PruefstandError
from pruefstand.io import sha256_bytes, write_json

BASE = "https://www.gesetze-im-internet.de"


def _parse(root):
    xml = (root / "tests" / "fixtures" / "gii_sample.xml").read_bytes()
    return parse_law_xml(xml, "testg", BASE, "2026-10-25")


def test_sample_units_and_ids(root):
    units, stats = _parse(root)
    assert [u.unit_id for u in units] == [
        "testg:§1:abs1",
        "testg:§1:abs2",
        "testg:§2:abs0",
        "testg:§4a:abs1",
        "testg:§4a:abs1_2",
    ]
    assert stats.n_units == 5
    assert stats.n_sections == 4
    assert stats.n_repealed_skipped == 1
    assert stats.skipped_designations == ["Inhaltsübersicht", "Anlage 1"]


def test_footnotes_removed_and_counted(root):
    units, stats = _parse(root)
    assert units[0].text == "(1) Zweck dieses Gesetzes ist die sichere Versorgung."
    assert all("Hinweis" not in u.text and "+++" not in u.text for u in units)
    assert stats.n_footnotes_removed == 5


def test_lists_continuations_and_metadata(root):
    units, _ = _parse(root)
    by_id = {u.unit_id: u for u in units}
    assert (
        by_id["testg:§2:abs0"].text
        == "Im Sinne dieses Gesetzes ist 1. Anlage: jede Einrichtung, 2. Betreiber: wer die Anlage betreibt."
    )
    assert by_id["testg:§4a:abs1"].text == "(1) Die Frist beträgt zwei Wochen. Satz 1 gilt entsprechend."
    unit = by_id["testg:§1:abs2"]
    assert unit.law_abbr == "TestG"
    assert unit.law_title == "Testgesetz"
    assert unit.heading == "Zweck"
    assert unit.stand == "Zuletzt geändert durch Art. 1 G v. 1.1.2026 I Nr. 1"
    assert unit.source_url == f"{BASE}/testg/__1.html"
    assert unit.label == "TestG § 1 Abs. 2"
    assert by_id["testg:§2:abs0"].label == "TestG § 2"
    assert unit.norm_char_start == len(by_id["testg:§1:abs1"].text) + 1


def test_helpers():
    assert unit_id_for("enwg_2005", "§ 14a", "1") == "enwg_2005:§14a:abs1"
    assert norm_url(BASE, "enwg_2005", "§ 14a") == f"{BASE}/enwg_2005/__14a.html"


def test_msbg_fixture_unit_count(root):
    xml = read_xml_from_zip(root / "fixtures" / "ci" / "gii" / "messbg.xml.zip")
    units, stats = parse_law_xml(xml, "messbg", BASE, "2026-10-25")
    assert stats.n_units == len(units) == 220
    assert units[0].law_abbr == "MsbG"
    assert len({u.unit_id for u in units}) == len(units)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_laws_wraps_a_corrupt_zip_and_keeps_the_old_file(tmp_path, monkeypatch):
    old = tmp_path / "messbg.xml.zip"
    old.write_bytes(b"previous good download")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"not a zip"))
    with pytest.raises(PruefstandError, match="download or unpacking failed"):
        fetch_laws(["messbg"], BASE, tmp_path, tmp_path / "snapshot.json")
    assert old.read_bytes() == b"previous good download"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["messbg.xml.zip"]


def test_fetch_laws_replaces_the_zip_atomically(root, tmp_path, monkeypatch):
    data = (root / "fixtures" / "ci" / "gii" / "messbg.xml.zip").read_bytes()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(data))
    snapshot = fetch_laws(["messbg"], BASE, tmp_path, tmp_path / "snapshot.json")
    assert (tmp_path / "messbg.xml.zip").read_bytes() == data
    assert snapshot["laws"]["messbg"]["bytes"] == len(data)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["messbg.xml.zip", "snapshot.json"]


def test_build_units_wraps_a_corrupt_zip(tmp_path):
    broken = b"not a zip"
    (tmp_path / "messbg.xml.zip").write_bytes(broken)
    write_json(
        tmp_path / "snapshot.json",
        {"fetched_at": "2026-10-25", "laws": {"messbg": {"sha256": sha256_bytes(broken)}}},
    )
    with pytest.raises(PruefstandError, match="cannot read or parse"):
        build_units(["messbg"], BASE, tmp_path, tmp_path / "snapshot.json", tmp_path / "units.jsonl.gz")
