#!/usr/bin/env python3
"""Download Arial.Unicode.ttf for training plot labels.

The font file is ~23MB and is excluded from git tracking.
This script downloads it to the project root on first use.
"""

import sys
import urllib.request
from pathlib import Path

FONT_URL = "https://github.com/matplotlib/matplotlib/raw/main/fonts/ttf/Arial Unicode MS.ttf"
TARGET_NAME = "Arial.Unicode.ttf"


def download_font():
    """Download the font file to the project root."""
    project_root = Path(__file__).parent.parent
    target = project_root / TARGET_NAME

    if target.exists():
        print(f"Font already exists: {target}")
        return True

    print(f"Downloading {TARGET_NAME}...")
    try:
        urllib.request.urlretrieve(FONT_URL, target)
        print(f"Downloaded to: {target}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually download the font and place it in the project root.")
        return False


if __name__ == "__main__":
    success = download_font()
    sys.exit(0 if success else 1)
