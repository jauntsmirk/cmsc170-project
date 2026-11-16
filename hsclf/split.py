"""Year-based split and chapter balancing (stdlib + numpy-free)."""
import random
from collections import defaultdict


def parse_years(spec):
    """'2012-2021' -> {2012..2021}; '2022' -> {2022}; '2012-2015,2018' also works."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cap_per_chapter(rows, cap, rng):
    """
    Evening-out step. For every chapter with more than `cap` rows, keep a uniform random
    subset of `cap` rows; chapters with <= cap rows are kept whole (no oversampling).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["hs_chapter"]].append(r)
    out = []
    for ch in sorted(groups):
        g = groups[ch]
        if len(g) > cap:
            g = rng.sample(g, cap)
        out.extend(g)
    return out


def balance_report(sets, population):
    """Per-chapter table: population share and the count that landed in each split."""
    chapters = sorted(set(population) | {r["hs_chapter"] for rows in sets.values() for r in rows})
    total = sum(population.values()) or 1
    table = []
    for ch in chapters:
        row = {"chapter": ch, "population": population.get(ch, 0),
               "population_share": round(population.get(ch, 0) / total, 6)}
        for name, rows in sets.items():
            row[name] = sum(1 for r in rows if r["hs_chapter"] == ch)
        table.append(row)
    return table
