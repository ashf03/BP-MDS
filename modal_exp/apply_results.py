#!/usr/bin/env python3
"""Thin wrapper — prefer `record_sols.py` (canonical .sol → scoreboard runner)."""

from __future__ import annotations

import sys

from record_sols import main

if __name__ == "__main__":
    # Preserve old flag style: apply_results.py --solver ours --jsonl ...
    sys.exit(main())
