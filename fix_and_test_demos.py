#!/usr/bin/env python3
"""
Fix demo imports and test them.

This script:
1. Updates all demo files with correct imports
2. Tests each demo to ensure imports work
3. Reports what's broken and what works
"""

import re
from pathlib import Path
import subprocess
import sys

TARGET_DIR = Path("/Users/apa/ml_projects/constitutional-ai")


def fix_demo_imports(demo_path: Path) -> bool:
    """Fix imports in a demo file."""
    print(f"\n{'='*70}")
    print(f"Fixing: {demo_path.name}")
    print('='*70)

    with open(demo_path, 'r') as f:
        content = f.read()

    original = content

    # Fix src.safety.constitutional imports
    content = re.sub(
        r'from src\.safety\.constitutional import',
        'from constitutional_ai import',
        content
    )

    # Fix src.configs imports
    content = re.sub(
        r'from src\.configs\.constitutional_training_config import',
        'from constitutional_ai.config import',
        content
    )

    # Fix src.data imports
    content = re.sub(
        r'from src\.data import create_default_prompts',
        'from constitutional_ai.data_utils import create_default_prompts',
        content
    )
    content = re.sub(
        r'from src\.data import (.*PromptDataset)',
        r'from constitutional_ai.critique_revision import \1',
        content
    )

    # Remove sys.path.append lines
    content = re.sub(
        r'sys\.path\.append\(.*?\)',
        '# Path append removed - using installed package',
        content
    )

    if content != original:
        with open(demo_path, 'w') as f:
            f.write(content)
        print(f"✓ Updated {demo_path.name}")
        return True
    else:
        print(f"  No changes needed")
        return False


def test_demo_imports(demo_path: Path) -> bool:
    """Test if a demo's imports work."""
    print(f"\nTesting imports in {demo_path.name}...")

    # Create a test script that just imports the demo
    test_code = f"""
import sys
sys.path.insert(0, '{TARGET_DIR}')

# Try to import the demo (this will fail if imports are broken)
try:
    import {demo_path.stem}
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {{e}}")
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            cwd=demo_path.parent,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"✓ {demo_path.name} imports work!")
            return True
        else:
            print(f"✗ {demo_path.name} has import errors:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {demo_path.name} timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing {demo_path.name}: {e}")
        return False


def main():
    """Fix and test all demos."""
    print("="*70)
    print("Constitutional AI - Demo Fix & Test")
    print("="*70)

    # Find all demo files
    demos_dir = TARGET_DIR / "demos"
    demo_files = list(demos_dir.glob("*.py"))

    print(f"\nFound {len(demo_files)} demo files")

    # Fix imports
    print("\n" + "="*70)
    print("STEP 1: Fixing Imports")
    print("="*70)

    fixed_count = 0
    for demo_path in demo_files:
        if fix_demo_imports(demo_path):
            fixed_count += 1

    print(f"\n✓ Fixed {fixed_count}/{len(demo_files)} demo files")

    # Test imports
    print("\n" + "="*70)
    print("STEP 2: Testing Imports")
    print("="*70)

    working_count = 0
    for demo_path in demo_files:
        if test_demo_imports(demo_path):
            working_count += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total demos: {len(demo_files)}")
    print(f"Fixed: {fixed_count}")
    print(f"Working: {working_count}/{len(demo_files)}")

    if working_count == len(demo_files):
        print("\n✅ All demos working!")
        return 0
    else:
        print(f"\n⚠️  {len(demo_files) - working_count} demos still have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
