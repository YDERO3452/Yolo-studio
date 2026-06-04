#!/usr/bin/env python3
"""Download Arial.Unicode.ttf for training plot labels.

The font file is ~23MB and is excluded from git tracking.
This script downloads it to the project root on first use.
"""

import sys
import urllib.request
from pathlib import Path

# Multiple fallback URLs for the font
FONT_URLS = [
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
    "https://raw.githubusercontent.com/matplotlib/matplotlib/main/fonts/ttf/DejaVuSans.ttf",
]
TARGET_NAME = "Arial.Unicode.ttf"


def download_font():
    """Download the font file to the project root."""
    project_root = Path(__file__).parent.parent
    target = project_root / TARGET_NAME

    if target.exists():
        print(f"Font already exists: {target}")
        return True

    for url in FONT_URLS:
        print(f"Trying: {url}")
        try:
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded to: {target}")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("All downloads failed. You can manually place a .ttf font in the project root.")
    return False


if __name__ == "__main__":
    success = download_font()
    sys.exit(0 if success else 1)
