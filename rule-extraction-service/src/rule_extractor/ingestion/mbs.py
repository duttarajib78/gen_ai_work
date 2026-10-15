"""MBS Schedule XML -> deterministic billing facts and a grep-friendly item lookup file (zero LLM calls).

The XML is external input: it is parsed with a hardened parser (no entity expansion, no DTD loading, no network)
and the item element is auto-detected so small schema differences between schedule releases do not break the build.
Tag names are matched case-insensitively against ``FIELD_ALIASES``; adjust them if your release differs, then check
the counts printed by ``rule-extractor corpus verify`` (a previous run on MBS-XML-20260801 reported 6,046 items and
1,132 EMSN-cap / derived-fee facts).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..errors import IngestionError
from ..util import clean_text

MAX_XML_BYTES = 400 * 1024 * 1024
NO_VALUE = {"", "nc", "n/a", "na", "none", "null", "-", "0", "0.00"}

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "item_num": ("itemnum", "itemnumber", "itemno", "item"),
    "description": ("description", "itemdescription", "desc"),
    "category": ("category",),
    "group": ("group",),
    "sub_group": ("subgroup",),
    "schedule_fee": ("schedulefee", "fee"),
    "derived_fee": ("derivedfee",),
    "derived_fee_start": ("derivedfeestartdate",),
    "item_start": ("itemstartdate",),
    "item_end": ("itemenddate",),
}
EMSN_ALIASES: dict[str, tuple[str, ...]] = {
    "emsn_cap": ("emsncap",),
    "emsn_fixed_cap_amount": ("emsnfixedcapamount",),
    "emsn_maximum_cap": ("emsnmaximumcap",),
    "emsn_percentage_cap": ("emsnpercentagecap",),
    "emsn_description": ("emsndescription",),
    "emsn_start_date": ("emsnstartdate",),
    "emsn_end_date": ("emsnenddate",),
}


@dataclass
class MbsItem:
    item_num: str
    description: str = ""
    category: str = ""
    group: str = ""
    schedule_fee: str = ""
    derived_fee: str = ""
    derived_fee_start: str = ""
    item_start: str = ""
    item_end: str = ""
    emsn: dict[str, str] = field(default_factory=dict)


def _parser():
    from lxml import etree

    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, dtd_validation=False,
                           huge_tree=False, remove_comments=True, remove_pis=True)


def _first(values: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        if alias in values:
            return values[alias]
    return ""


def parse_mbs_xml(path: Path | str) -> list[MbsItem]:
    from lxml import etree

    p = Path(path)
    if not p.is_file():
        raise IngestionError(f"MBS XML not found: {p.name}")
    if p.stat().st_size > MAX_XML_BYTES:
        raise IngestionError("MBS XML is unreasonably large")
    try:
        root = etree.parse(str(p), parser=_parser()).getroot()
    except etree.XMLSyntaxError as exc:
        raise IngestionError(f"MBS XML is not well-formed: {exc}") from exc
    counts = Counter(el.tag for el in root.iter() if isinstance(el.tag, str) and el is not root and len(el) >= 3)
    if not counts:
        raise IngestionError("could not find repeating item elements in the MBS XML")
    item_tag = counts.most_common(1)[0][0]
    items: list[MbsItem] = []
    for el in root.iter(item_tag):
        values: dict[str, str] = {}
        for child in el:
            if isinstance(child.tag, str) and child.tag.lower() not in values:
                values[child.tag.lower()] = clean_text(child.text or "")
        item_num = _first(values, FIELD_ALIASES["item_num"])
        if not item_num:
            continue
        items.append(MbsItem(
            item_num=item_num,
            description=_first(values, FIELD_ALIASES["description"]),
            category=_first(values, FIELD_ALIASES["category"]),
            group=_first(values, FIELD_ALIASES["group"]),
            schedule_fee=_first(values, FIELD_ALIASES["schedule_fee"]),
            derived_fee=_first(values, FIELD_ALIASES["derived_fee"]),
            derived_fee_start=_first(values, FIELD_ALIASES["derived_fee_start"]),
            item_start=_first(values, FIELD_ALIASES["item_start"]),
            item_end=_first(values, FIELD_ALIASES["item_end"]),
            emsn={name: _first(values, aliases) for name, aliases in EMSN_ALIASES.items() if _first(values, aliases)},
        ))
    if not items:
        raise IngestionError("the MBS XML contained no items with an item number")
    return items


def _has_value(value: str) -> bool:
    return value.strip().lower() not in NO_VALUE


def billing_facts(items: list[MbsItem]) -> Iterator[dict]:
    """EMSN-cap and derived-fee facts read straight off the structured fields."""
    for it in items:
        emsn = {k: v for k, v in it.emsn.items() if _has_value(v)}
        if emsn:
            yield {"fact_type": "emsn_cap", "item_num": it.item_num, "description": it.description[:200],
                   "emsn": emsn, "source": "mbs_xml"}
        if _has_value(it.derived_fee):
            yield {"fact_type": "derived_fee", "item_num": it.item_num, "description": it.description[:200],
                   "derived_fee": it.derived_fee, "derived_fee_start_date": it.derived_fee_start,
                   "schedule_fee": it.schedule_fee, "source": "mbs_xml"}


def lookup_lines(items: list[MbsItem]) -> Iterator[str]:
    """One line per item so the agent can ``Grep`` an item number and read its context."""
    for it in items:
        desc = " ".join(it.description.split())[:300]
        yield (f"ITEM {it.item_num} | category {it.category or '?'} | group {it.group or '?'} | "
               f"fee {it.schedule_fee or '?'} | {desc}")
