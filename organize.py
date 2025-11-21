"""
Project Organization Script
Cleans up your cml_final project structure for classification vs regression tasks
"""

import shutil
from pathlib import Path

print("="*70)
print("PROJECT ORGANIZATION SCRIPT")
print("="*70)

# Define base directory
BASE_DIR = Path.cwd()

print(f"\nWorking directory: {BASE_DIR}")
print(f"Project: {BASE_DIR.name}")

# Verify we're in the right place
if BASE_DIR.name != "cml_final":
    print("\n⚠️  WARNING: Not in cml_final directory!")
    print(f"Current directory: {BASE_DIR}")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        exit(0)

# Create new directory structure
print("\n[1/4] Creating new directory structure...")

new_dirs = [
    BASE_DIR / "scripts" / "classification",
    BASE_DIR / "scripts" / "regression",
    BASE_DIR / "submissions" / "classification",
    BASE_DIR / "submissions" / "regression",
]

for dir_path in new_dirs:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_path.relative_to(BASE_DIR)}")

# Define files to move
print("\n[2/4] Moving classification scripts...")

classification_scripts = {
    "both_formats.py": "scripts/classification",
    "check_setup.py": "scripts/classification",
    "finetune_gb.py": "scripts/classification",
    "generate_kaggle_submission.py": "scripts/classification",
    "quickstart_submission.py": "scripts/classification",
    "stacking_sub.py": "scripts/classification",
    "xgboost_sub.py": "scripts/classification",
}

moved_scripts = []
for filename, dest_folder in classification_scripts.items():
    src = BASE_DIR / filename
    if src.exists():
        dest = BASE_DIR / dest_folder / filename
        shutil.move(str(src), str(dest))
        print(f"  ✓ {filename} → {dest_folder}/")
        moved_scripts.append(filename)
    else:
        print(f"  ⊘ {filename} (not found, skipping)")

print(f"\n  Moved {len(moved_scripts)} scripts")

# Move submission files
print("\n[3/4] Moving classification submissions...")

submissions_dir = BASE_DIR / "submissions"
classification_subs_dest = BASE_DIR / "submissions" / "classification"

moved_subs = []
if submissions_dir.exists():
    for csv_file in submissions_dir.glob("*.csv"):
        # Skip if already in subdirectory
        if "classification" in str(csv_file.parent) or "regression" in str(csv_file.parent):
            continue
        
        dest = classification_subs_dest / csv_file.name
        shutil.move(str(csv_file), str(dest))
        print(f"  ✓ {csv_file.name}")
        moved_subs.append(csv_file.name)

print(f"\n  Moved {len(moved_subs)} submission files")

# Create README files
print("\n[4/4] Creating README files...")

# Classification README
classification_readme = BASE_DIR / "scripts" / "classification" / "README.md"
classification_readme.write_text("""# Classification Scripts (HEF - Hospital Expire Flag)

Scripts for predicting patient mortality (binary classification).

## Files:
- `quickstart_submission.py` - Quick single RF submission
- `generate_kaggle_submission.py` - Generate multiple submissions (RF, GB, ET, Ensemble)
- `finetune_gb.py` - Fine-tune GB hyperparameters
- `xgboost_sub.py` - XGBoost model
- `stacking_sub.py` - Stacking ensemble
- `both_formats.py` - Generate probability + binary versions
- `check_setup.py` - Verify setup before running

## Best Result:
- Gradient Boosting: 0.727226 AUC (Kaggle leaderboard)

## Quick Start:
```bash
python quickstart_submission.py
```
""")
print(f"  ✓ scripts/classification/README.md")

# Regression README placeholder
regression_readme = BASE_DIR / "scripts" / "regression" / "README.md"
regression_readme.write_text("""# Regression Scripts (LOS - Length of Stay)

Scripts for predicting ICU length of stay (regression).

## Status:
Work in progress - scripts will be added here.

## Structure (planned):
- `quickstart_los.py` - Quick single model submission
- `generate_los_submissions.py` - Multiple regression models
- Feature engineering and tuning scripts
""")
print(f"  ✓ scripts/regression/README.md")

# Summary
print("\n" + "="*70)
print("ORGANIZATION COMPLETE!")
print("="*70)

print(f"\n📁 New Structure:")
print(f"""
cml_final/
├── scripts/
│   ├── classification/        ({len(moved_scripts)} scripts)
│   │   └── README.md
│   └── regression/            (ready for LOS)
│       └── README.md
├── submissions/
│   ├── classification/        ({len(moved_subs)} CSV files)
│   └── regression/            (ready for LOS)
├── data/
│   ├── raw/
│   │   ├── MIMIC III dataset HEF/
│   │   └── MIMIC III dataset LOS/
│   └── output/
├── notebooks/
└── src/
""")

print("\n✅ Benefits:")
print("  - Clear separation of classification vs regression")
print("  - Easy to find scripts for each task")
print("  - Better for final submission/presentation")
print("  - Ready to add LOS regression scripts")

print("\n📝 Next Steps:")
print("  1. Test that scripts still work from new location")
print("  2. Update any import paths if needed")
print("  3. Ready to build LOS regression pipeline!")

print("\n" + "="*70)