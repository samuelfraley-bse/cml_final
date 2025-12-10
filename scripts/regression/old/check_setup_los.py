"""
LOS Setup Checker
Verify everything is ready for LOS regression task
"""

import sys
from pathlib import Path

print("="*60)
print("LOS REGRESSION - SETUP CHECKER")
print("="*60)

# Determine base directory
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    print(f"\nRunning from: scripts/regression/")
else:
    BASE_DIR = Path.cwd()
    print(f"\nRunning from: {BASE_DIR}")

print(f"Base directory: {BASE_DIR}")

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
        errors.append(f"Missing: {install_name}")

# 2. Check directory structure
print("\n[2/4] Checking directories...")
required_dirs = [
    BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS",
    BASE_DIR / "scripts" / "regression",
    BASE_DIR / "submissions" / "regression",
]

for dir_path in required_dirs:
    if dir_path.exists():
        print(f"  ✓ {dir_path.relative_to(BASE_DIR)}")
    else:
        print(f"  ✗ {dir_path.relative_to(BASE_DIR)} - NOT FOUND")
        if "data/raw" in str(dir_path):
            errors.append(f"Missing data: {dir_path.relative_to(BASE_DIR)}")
        else:
            warnings.append(f"Missing: {dir_path.relative_to(BASE_DIR)}")

# 3. Check data files
print("\n[3/4] Checking LOS data files...")
data_dir = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
required_files = [
    "mimic_train_LOS.csv",
    "mimic_test_LOS.csv"
]

for filename in required_files:
    filepath = data_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {filename} - NOT FOUND")
        errors.append(f"Missing: {filename}")

# 4. Check los_prep.py module
print("\n[4/4] Checking los_prep.py module...")
los_prep_locations = [
    BASE_DIR / "scripts" / "regression" / "los_prep.py",
    BASE_DIR / "src" / "los_prep.py",
]

los_prep_found = False
for loc in los_prep_locations:
    if loc.exists():
        print(f"  ✓ Found: {loc.relative_to(BASE_DIR)}")
        los_prep_found = True
        break

if not los_prep_found:
    print(f"  ✗ los_prep.py not found!")
    print(f"    Expected in:")
    for loc in los_prep_locations:
        print(f"      - {loc.relative_to(BASE_DIR)}")
    errors.append("Missing los_prep.py module")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if not errors and not warnings:
    print("\n✓ ALL CHECKS PASSED!")
    print("\nYou're ready to generate LOS submissions:")
    print("  python quickstart_los.py")
    print("     OR")
    print("  python generate_los_submissions.py")
    
elif errors:
    print(f"\n✗ FOUND {len(errors)} ERROR(S):")
    for error in errors:
        print(f"  - {error}")
    print("\nPlease fix these before running LOS scripts.")
    
if warnings:
    print(f"\n⚠ FOUND {len(warnings)} WARNING(S):")
    for warning in warnings:
        print(f"  - {warning}")
    print("\nThese will be created automatically when needed.")

print("="*60)
