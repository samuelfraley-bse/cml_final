"""
Setup Checker - Run this first to verify everything is ready
"""

import sys
from pathlib import Path

print("="*60)
print("HEF KAGGLE SUBMISSION - SETUP CHECKER")
print("="*60)

errors = []
warnings = []

# 1. Check Python packages
print("\n[1/4] Checking Python packages...")
packages = {
    'pandas': 'pandas',
    'numpy': 'numpy', 
    'sklearn': 'scikit-learn'
}

for import_name, install_name in packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {install_name}")
    except ImportError:
        print(f"  ✗ {install_name} - NOT INSTALLED")
        errors.append(f"Missing package: {install_name}")
        print(f"    Install with: pip install {install_name}")

# 2. Check directory structure
print("\n[2/4] Checking directory structure...")
BASE_DIR = Path.cwd()
print(f"  Current directory: {BASE_DIR}")

expected_dirs = [
    BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF",
]

for dir_path in expected_dirs:
    if dir_path.exists():
        print(f"  ✓ {dir_path.relative_to(BASE_DIR)}")
    else:
        print(f"  ✗ {dir_path.relative_to(BASE_DIR)} - NOT FOUND")
        warnings.append(f"Missing directory: {dir_path.relative_to(BASE_DIR)}")

# 3. Check data files
print("\n[3/4] Checking data files...")
data_dir = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
required_files = [
    "mimic_train_HEF.csv",
    "mimic_test_HEF.csv"
]

for filename in required_files:
    filepath = data_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {filename} - NOT FOUND")
        errors.append(f"Missing data file: {filename}")

# 4. Check output directory
print("\n[4/4] Checking output directory...")
output_dir = BASE_DIR / "submissions"
if not output_dir.exists():
    print(f"  Creating: {output_dir.relative_to(BASE_DIR)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created successfully")
else:
    print(f"  ✓ {output_dir.relative_to(BASE_DIR)} exists")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if not errors and not warnings:
    print("\n✓ ALL CHECKS PASSED!")
    print("\nYou're ready to generate submissions:")
    print("  python quickstart_submission.py")
    
elif errors:
    print(f"\n✗ FOUND {len(errors)} ERROR(S):")
    for error in errors:
        print(f"  - {error}")
    print("\nPlease fix these before running the submission scripts.")
    
if warnings:
    print(f"\n⚠ FOUND {len(warnings)} WARNING(S):")
    for warning in warnings:
        print(f"  - {warning}")

print("="*60)