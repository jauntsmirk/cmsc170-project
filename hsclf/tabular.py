"""Tiny CSV helpers (stdlib only): read a CSV into a list of dicts and write it back."""

import csv
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
