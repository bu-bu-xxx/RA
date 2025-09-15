"""Thin CLI shim. Usage:
    python -m cli <subcommand> ...
Run from inside the RABBI folder.
"""
import os
import sys

# Ensure this folder is on sys.path so 'framework' package resolves
REFAC_ROOT = os.path.dirname(__file__)
if REFAC_ROOT not in sys.path:
    sys.path.insert(0, REFAC_ROOT)


def main():
    from framework.cli import main as _main
    _main()


if __name__ == "__main__":
    main()
