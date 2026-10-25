"""Fetch and parse consolidated federal law XML from gesetze-im-internet.de (GII).

Laws are public domain under § 5 Abs. 1 UrhG. GII texts are non-official consolidations; the source is cited
in the README, the site and every unit's deep link.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl

USER_AGENT = "rag-pruefstand/0.1 (+https://github.com/HARISPAARTAN/rag-pruefstand)"
NORM_RE = re.compile(r"^§ (\d+[a-z]*)$")
ABSATZ_RE = re.compile(r"^\((\d+[a-z]*)\)\s*")
DROP_TAGS = {"FnR", "Footnotes", "Footnote", "noindex", "kommentar"}
SPACED_TAGS = {"BR", "DT", "DD", "LA", "entry", "row", "P", "DL", "table", "tgroup", "tbody", "thead"}
REPEALED_TEXTS = {"(weggefallen)", "(aufgehoben)", "-", "(XXXX)"}


@dataclass(frozen=True)
class Unit:
    unit_id: str
    law_slug: str
    law_abbr: str
    law_title: str
    norm: str
    heading: str
    absatz: str
    text: str
    norm_char_start: int
    norm_char_end: int
    source_url: str
    snapshot_date: str
    stand: str

    @property
    def label(self) -> str:
        """Citation label as the answer prompt asks for it, e.g. 'EnWG § 14a Abs. 1'."""
        base = f"{self.law_abbr} {self.norm}"
        return base if self.absatz == "0" else f"{base} Abs. {self.absatz}"


@dataclass
class ParseStats:
    law_slug: str
    n_sections: int = 0
    n_units: int = 0
    n_repealed_skipped: int = 0
    n_non_section_skipped: int = 0
    n_footnotes_removed: int = 0
    skipped_designations: list[str] = field(default_factory=list)


def zip_url(base_url: str, slug: str) -> str:
    return f"{base_url}/{slug}/xml.zip"


def norm_url(base_url: str, slug: str, norm: str) -> str:
    """Deep link to the HTML page of one section, e.g. .../enwg_2005/__14a.html."""
    number = norm.replace("§", "").strip()
    return f"{base_url}/{slug}/__{number}.html"


def unit_id_for(slug: str, norm: str, absatz: str) -> str:
    return f"{slug}:§{norm.replace('§', '').strip()}:abs{absatz}"


def element_text(element: ET.Element) -> str:
    """Flatten an element to text: drop footnote markup, separate list items and table cells by spaces."""
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag in DROP_TAGS:
            if node.tail:
                parts.append(node.tail)
            return
        if node.tag in SPACED_TAGS:
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node.tag in SPACED_TAGS:
            parts.append(" ")
        if node.tail:
            parts.append(node.tail)

    walk(element)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _count_footnotes(element: ET.Element | None) -> int:
    if element is None:
        return 0
    return sum(1 for node in element.iter() if node.tag in ("FnR", "Footnote"))


def read_xml_from_zip(zip_path: str | Path) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected exactly one XML file, found {names}")
        return archive.read(names[0])


def law_metadata(root: ET.Element) -> dict:
    """Abbreviation, title, 'Stand' note and build date from the first <norm> of a GII document."""
    meta = root.find("norm/metadaten")
    if meta is None:
        raise ValueError("GII document has no <norm>/<metadaten>")
    abbr = meta.findtext("amtabk") or meta.findtext("jurabk") or ""
    title_el = meta.find("kurzue")
    if title_el is None:
        title_el = meta.find("langue")
    stand = ""
    for item in meta.findall("standangabe"):
        if item.findtext("standtyp") == "Stand":
            stand = (item.findtext("standkommentar") or "").strip()
    return {
        "law_abbr": abbr.strip(),
        "law_title": element_text(title_el) if title_el is not None else "",
        "stand": stand,
        "builddate": root.get("builddate", ""),
    }


def parse_law_xml(
    xml_bytes: bytes, slug: str, base_url: str, snapshot_date: str
) -> tuple[list[Unit], ParseStats]:
    """Split one law into units: one unit per Absatz of every § section; Anlagen and the TOC are skipped."""
    root = ET.fromstring(xml_bytes)
    meta = law_metadata(root)
    stats = ParseStats(law_slug=slug)
    units: list[Unit] = []
    for norm_el in root.findall("norm"):
        designation = (norm_el.findtext("metadaten/enbez") or "").strip()
        stats.n_footnotes_removed += _count_footnotes(norm_el.find("textdaten/text"))
        fussnoten = norm_el.find("textdaten/fussnoten")
        if fussnoten is not None and len(fussnoten):
            stats.n_footnotes_removed += 1
        if not designation:
            continue
        match = NORM_RE.match(designation)
        if not match:
            stats.n_non_section_skipped += 1
            stats.skipped_designations.append(designation)
            continue
        content = norm_el.find("textdaten/text/Content")
        if content is None:
            stats.n_repealed_skipped += 1
            continue
        stats.n_sections += 1
        norm = f"§ {match.group(1)}"
        title_el = norm_el.find("metadaten/titel")
        heading = element_text(title_el) if title_el is not None else ""
        pieces: list[list[str]] = []
        for paragraph in content.findall("P"):
            text = element_text(paragraph)
            if not text:
                continue
            absatz_match = ABSATZ_RE.match(text)
            if absatz_match:
                pieces.append([absatz_match.group(1), text])
            elif pieces:
                pieces[-1][1] = f"{pieces[-1][1]} {text}"
            else:
                pieces.append(["0", text])
        if not pieces or (len(pieces) == 1 and pieces[0][1] in REPEALED_TEXTS):
            stats.n_repealed_skipped += 1
            continue
        seen: dict[str, int] = {}
        offset = 0
        for absatz, text in pieces:
            seen[absatz] = seen.get(absatz, 0) + 1
            unique = absatz if seen[absatz] == 1 else f"{absatz}_{seen[absatz]}"
            units.append(
                Unit(
                    unit_id=unit_id_for(slug, norm, unique),
                    law_slug=slug,
                    law_abbr=meta["law_abbr"],
                    law_title=meta["law_title"],
                    norm=norm,
                    heading=heading,
                    absatz=unique,
                    text=text,
                    norm_char_start=offset,
                    norm_char_end=offset + len(text),
                    source_url=norm_url(base_url, slug, norm),
                    snapshot_date=snapshot_date,
                    stand=meta["stand"],
                )
            )
            offset += len(text) + 1
    stats.n_units = len(units)
    return units, stats


def fetch_laws(slugs: list[str], base_url: str, zip_dir: Path, snapshot_path: Path) -> dict:
    """Download xml.zip per law, record URL, size, SHA-256 and GII build date in snapshot.json."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source": "gesetze-im-internet.de, Bundesministerium der Justiz; nichtamtliche konsolidierte Fassung",
        "licence_note": "Gesetze und Verordnungen sind nach § 5 Abs. 1 UrhG gemeinfrei",
        "fetched_at": date.today().isoformat(),
        "laws": {},
    }
    for slug in slugs:
        url = zip_url(base_url, slug)
        target = zip_dir / f"{slug}.xml.zip"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())
            root = ET.fromstring(read_xml_from_zip(target))
        except (OSError, urllib.error.URLError, zipfile.BadZipFile, ET.ParseError, ValueError) as exc:
            raise PruefstandError(f"{url}: download or unpacking failed: {exc}") from exc
        snapshot["laws"][slug] = {
            "url": url,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "builddate": root.get("builddate", ""),
            "law_abbr": law_metadata(root)["law_abbr"],
        }
    write_json(snapshot_path, snapshot)
    return snapshot


def build_units(
    slugs: list[str], base_url: str, zip_dir: Path, snapshot_path: Path, units_path: Path
) -> list[ParseStats]:
    """Parse every law zip named in the snapshot into units; verify each zip against its recorded SHA-256."""
    if not Path(snapshot_path).is_file():
        raise PruefstandError(f"{snapshot_path} not found; run `pruefstand corpus fetch` first")
    snapshot = read_json(snapshot_path)
    all_units: list[Unit] = []
    all_stats: list[ParseStats] = []
    for slug in slugs:
        if slug not in snapshot["laws"]:
            raise PruefstandError(f"{snapshot_path}: no entry for {slug}; run `pruefstand corpus fetch`")
        entry = snapshot["laws"][slug]
        zip_path = zip_dir / f"{slug}.xml.zip"
        if not zip_path.is_file():
            raise PruefstandError(f"{zip_path} not found; run `pruefstand corpus fetch` first")
        actual = sha256_file(zip_path)
        if actual != entry["sha256"]:
            raise PruefstandError(f"{zip_path}: SHA-256 {actual} does not match snapshot {entry['sha256']}")
        units, stats = parse_law_xml(read_xml_from_zip(zip_path), slug, base_url, snapshot["fetched_at"])
        all_units.extend(units)
        all_stats.append(stats)
    write_jsonl(units_path, (asdict(unit) for unit in all_units))
    return all_stats


def load_units(units_path: str | Path) -> list[Unit]:
    if not Path(units_path).is_file():
        raise PruefstandError(f"{units_path} not found; run `pruefstand corpus build` first")
    return [Unit(**row) for row in read_jsonl(units_path)]
