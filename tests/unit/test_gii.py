from pruefstand.corpus.gii import norm_url, parse_law_xml, read_xml_from_zip, unit_id_for

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
