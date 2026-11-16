"""
Streaming samplers for the Bureau of Customs yearly CSVs (stdlib only).

One pass over each yearly file produces TWO samples at the same time:

  * natural sample : uniform reservoir of K_NAT rows per year. Keeps the real
                     (very imbalanced) chapter distribution. Used for the
                     "realistic" test set and for the sampling-bias check.
  * balanced pool  : one reservoir of up to R rows PER HS CHAPTER per year
                     (stratified reservoir sampling, Vitter's Algorithm R run
                     separately inside each chapter). A chapter with fewer than
                     R rows in that year keeps all of them. Nothing is ever
                     duplicated or oversampled.

The per-chapter population counts of every year are logged, so the true
distribution is always known and any reweighting can be done later.

Eligibility (applied BEFORE sampling, so each sample is uniform over the
eligible rows): digits-only HS code, non-empty goods description, and a valid
2-digit chapter (01-97, 77 is reserved by the WCO).

HS code normalisation: Philippine codes are 11 digits (8-digit AHTN + 3 digit
statistical suffix). A code that lost its leading zero when exported as a number
(e.g. 1012100000) is left-padded with zeros to 11 digits before taking the
chapter. The length histogram is logged so this assumption can be checked.
"""

import csv
import os
import re

ALIASES = {
    "year": ["ty", "year"],
    "month": ["tm", "month"],
    "hs_code": ["hscode"],
    "goods_description": ["goodsdescription"],
    "country_origin": ["countryorigin_iso3", "countryorigin"],
    "country_export": ["countryexport_iso3", "countryexport"],
    "quantity": ["q", "quantity"],
    "uom": ["uom"],
    "currency": ["currency"],
    "dutiable_value_php": ["dutiablevaluephp"],
    "duty": ["dutypaid", "duty"],
    "duties_taxes_total": ["dutiestaxes"],
    "pref_code": ["prefcode", "pref_code"],
    "fines_penalties": ["finesandpenalties"],
}
OUT_COLUMNS = list(ALIASES) + ["hs_chapter", "source_file", "source_row"]

HS_RE = re.compile(r"^\d+$")
VALID_CHAPTERS = frozenset(f"{c:02d}" for c in range(1, 98) if c != 77)


def hs_chapter(hs_code):
    """First two digits of the 11-digit code (left-padded with zeros)."""
    return hs_code.zfill(11)[:2]


def clean_lines(fh):
    for line in fh:
        yield line.replace("\x00", "") if "\x00" in line else line


def build_index(header):
    pos = {name.strip().lower(): i for i, name in enumerate(header)}
    return {col: next((pos[n] for n in names if n in pos), None) for col, names in ALIASES.items()}


def _get(row, i):
    return "" if i is None or i >= len(row) else row[i].strip()


class Reservoir:
    """Algorithm R: keeps a uniform random sample of up to `size` items from a stream."""

    def __init__(self, size, rng):
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.items[j] = item


def sample_file(path, per_chapter, k_natural, rng, year_hint):
    """One streaming pass. Returns (pool_rows, natural_rows, stats)."""
    stats = {"rows_read": 0, "malformed": 0, "ineligible": 0, "bad_chapter": 0,
             "eligible": 0, "hs_length_hist": {}, "chapter_counts": {}}
    natural = Reservoir(k_natural, rng)
    by_chapter = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(clean_lines(fh))
        header = next(reader)
        index = build_index(header)
        ncols = len(header)
        for need in ("hs_code", "goods_description"):
            if index[need] is None:
                raise SystemExit(f"{path}: no column for {need}. Header: {header}")
        n = 0
        for row in reader:
            stats["rows_read"] += 1
            if len(row) != ncols:
                stats["malformed"] += 1
                continue
            hs = _get(row, index["hs_code"])
            desc = _get(row, index["goods_description"])
            if not HS_RE.match(hs) or not desc:
                stats["ineligible"] += 1
                continue
            ch = hs_chapter(hs)
            if ch not in VALID_CHAPTERS:
                stats["bad_chapter"] += 1
                continue
            n += 1
            L = str(len(hs))
            stats["hs_length_hist"][L] = stats["hs_length_hist"].get(L, 0) + 1
            stats["chapter_counts"][ch] = stats["chapter_counts"].get(ch, 0) + 1
            item = (n, ch, row)
            natural.add(item)
            if ch not in by_chapter:
                by_chapter[ch] = Reservoir(per_chapter, rng)
            by_chapter[ch].add(item)
        stats["eligible"] = n

    fname = os.path.basename(path)

    def to_dicts(items):
        out = []
        for n_, ch, row in sorted(items, key=lambda t: t[0]):
            d = {col: _get(row, index[col]) for col in ALIASES}
            if not d["year"]:
                d["year"] = str(year_hint)
            d["hs_chapter"] = ch
            d["source_file"] = fname
            d["source_row"] = n_
            out.append(d)
        return out

    pool_items = [it for ch in sorted(by_chapter) for it in by_chapter[ch].items]
    return to_dicts(pool_items), to_dicts(natural.items), stats
