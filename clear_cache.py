#!/usr/bin/env python
"""
Clear cache directories for the MBO prediction project.
This script removes cached data to force fresh data loading.
"""

import shutil
from pathlib import Path

def clear_cache():
    """Clear all cache directories"""
    cache_dirs = [
        Path("cache_dir"),
        Path("__pycache__"),
        Path("scripts/__pycache__"),
        Path("scripts/models/__pycache__"),
        Path("scripts/prediction_methods/__pycache__"),
        Path("scripts/utils/__pycache__"),
    ]
    
    cleared = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                cleared.append(str(cache_dir))
                print(f"✓ Cleared: {cache_dir}")
            except Exception as e:
                print(f"✗ Failed to clear {cache_dir}: {e}")
    
    if cleared:
        print(f"\n✓ Successfully cleared {len(cleared)} cache director{'y' if len(cleared) == 1 else 'ies'}!")
    else:
        print("\n✓ No cache directories found to clear.")

if __name__ == "__main__":
    clear_cache()
