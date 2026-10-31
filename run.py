#!/usr/bin/env python3
"""Single entry point for the stadium crowd-flow and evacuation project.

Examples
--------
    python run.py --help
    python run.py list
    python run.py stadium football_bowl
    python run.py crowd football_bowl emergency --seed 7
    python run.py check-phase1
"""
import sys

from crowdsim.cli import main

if __name__ == "__main__":
    sys.exit(main())
