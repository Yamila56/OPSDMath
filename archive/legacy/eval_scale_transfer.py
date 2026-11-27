"""Compatibility entry point for the packaged implementation."""

import sys

from archive.experiments.budget_efficiency import scale_transfer as implementation


if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
