# Dataset Splitting & Formatting Tools

## Overview

This repository contains tools for splitting paired images (e.g., segmentation input images and corresponding masks) into train / validation / test sets. All scripts run on standard library only and work on Linux / macOS / Windows with Python 3.7+.

Tools are consolidated under `util/datasets/random_split/`. The parent directory contains only backward-compatible wrappers.

---

## File List

| File | Role |
|------|------|
| `random_split/split_dataset.py` | Matches multiple image and mask directories, splits by specified ratios into `train` / `valid` / `test`. Generates logs (CSV / summary). |
| `random_split/format_dataset.py` | Copies `split_dataset.py` output to final submission format (`train_images/`, etc.) with sequential renaming. |
| `random_split/run_split_and_format.sh` | Shell wrapper that runs splitting and formatting consecutively. Removes intermediate directory and keeps only the formatted folder. |
| `random_split/rename_images.py` | Sequentially renames images in any directory. Supports `--recursive` for batch processing. |
| `random_split/README.md` | Quick reference for each script. |

---

## Requirements

- Python 3.7+
- Image and mask files must have matching base names (excluding extension) (e.g., `image1.png` and `image1.png`)
- Input directories are scanned non-recursively. To include subfolders, flatten them beforehand or use `rename_images.py --recursive`.

---

## Typical Workflow

### 1. Verify Split (dry-run)

```bash
python3 random_split/split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2 \
    --dry-run
```

- No copying or moving occurs; only logs are generated.

### 2. Execute Split

```bash
python3 random_split/split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2
```

- Default is copy. Use `--move` to move files instead.
- Output directory name automatically gets a timestamp suffix (`_yyyymmddHHMM`). Disable with `--no-append-timestamp`.
- On completion, `SPLIT_DATASET_OUTPUT_ROOT=/actual/path` is printed for use in subsequent steps.

### 3. Preserve Existing Test Set and Redistribute

```bash
python3 random_split/split_dataset.py \
    --images /path/to/train_images /path/to/test_images \
    --masks  /path/to/train_masks  /path/to/test_masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2 \
    --preserve-original-test
```

- The second input (index 1) pairs remain as `test`; the rest are redistributed into `train` / `valid`.
- Use `--preserve-test-index N` to specify any index.

### 4. Format to Final Structure

```bash
python3 random_split/format_dataset.py \
    --source /path/to/out_dir_202512171234 \
    --dest   /path/to/final_dataset
```

- Creates `train_images/`, `train_masks/`, `validation_images/`, `validation_masks/`, `test_images/`, `test_masks/` under `final_dataset/`.
- Files are renamed sequentially (e.g., `image1.png`, `mask1.png`).

### 5. Split and Format Together

```bash
bash random_split/run_split_and_format.sh \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2
```

- Use `--format-dest /your/dest` to specify formatted output destination. Default is `<split_output>_formatted`.
- Set `PYTHON_BIN` environment variable to use a different Python interpreter.
- Use `--dry-run` to run split dry-run only; formatting is skipped.
- On success, the intermediate split directory is removed and logs are moved to the formatted folder.

---

## Folder Structure Overview

```
split_out_YYYYMMDDHHMM/         # split_dataset.py output
├── train/
│   ├── images/
│   └── masks/
├── valid/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
├── split_log_*.csv
└── split_summary_*.txt

Dataset/                        # format_dataset.py output
├── train_images/
├── train_masks/
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

---

## `split_dataset.py` Options

| Option | Role |
|--------|------|
| `--images` (required) | Image directory (multiple allowed). Merges by matching base names. |
| `--masks` (required) | Mask directory (multiple allowed). |
| `--out` (required) | Output root. Creates `train/valid/test` with `images/` and `masks/` subdirectories. |
| `--train`, `--valid`, `--test` | Split ratios. Decimals or percentages; auto-normalized. |
| `--seed` | Random seed (default 42). |
| `--move` | Move instead of copy. |
| `--dry-run` | Generate logs only; no file operations. |
| `--preserve-test-index` | Keep specified input index as `test`. |
| `--preserve-original-test` | When 2 input sets exist, automatically keep index 1 as `test`. |
| `--no-append-timestamp` | Disable timestamp suffix on output root. |

---

## `format_dataset.py` Options

| Option | Role |
|--------|------|
| `--source` (required) | `split_dataset.py` output root. |
| `--dest` (required) | Formatted output destination. Created automatically if it doesn't exist. |
| `--dry-run` | Show actions without copying. |

---

## `rename_images.py` — Image Sequential Renaming Tool

Sequentially renames images in a single directory or recursively across multiple directories.

### Basic Command

```bash
python3 random_split/rename_images.py \
    --dir /path/to/images \
    --prefix sample \
    --start 1 \
    --zero-pad 4
```

| Option | Description |
|--------|-------------|
| `--dir` | Target directory (required). |
| `--prefix` | Filename prefix (default `gen_img`). |
| `--start` | Starting number for sequence (default 0). |
| `--zero-pad` | Zero-padding width. Use `--auto-pad` for automatic adjustment based on max index. |
| `--ext` | Target extension(s) (multiple allowed, default `.png`). |
| `--output` | Copy to different directory. Omit for in-place rename. |
| `--recursive` | Process all subdirectories under `--dir`. |
| `--skip-processed` | Skip directories with marker file. |
| `--write-marker` | Create marker file on successful completion. |
| `--marker-name` | Marker filename (default `.rename_images_done`). |
| `--overwrite` | Allow overwriting existing files. |
| `--dry-run` | Show plan without making changes. |

### Recursive Mode

```bash
python3 random_split/rename_images.py \
    --dir /datasets/raw_assets \
    --recursive \
    --prefix dataset \
    --start 0 \
    --auto-pad
```

- Auto-detects directories containing only files with target extensions.
- Combined with `--output`, preserves relative path structure from root to a separate tree.
- Sequence resets independently for each directory.

---

## Output Logs

Generated by `split_dataset.py` in `--out` directory:

- `split_log_<timestamp>.csv` — Records destination and source path for each file.
- `split_summary_<timestamp>.txt` — Total pairs, train/valid/test counts.

`format_dataset.py` doesn't generate log files but prints copy counts to stdout.

---

## Implementation Notes

- Errors and stops if duplicate base names are found across input directories.
- Remainder samples are assigned to `train`. Modify `compute_counts` for different rules.
- `format_dataset.py` auto-skips pairs with mismatched base names; only common pairs are formatted.

---

## Troubleshooting

| Symptom | Cause / Solution |
|---------|------------------|
| `No paired files found` | Base names don't match, or only one of image/mask exists. Check naming conventions and input paths. |
| `Duplicate basenames detected` | Same-named pairs exist in multiple input directories. Organize files. |
| Logs exist but files not copied | Possibly running with `--dry-run`. Remove the flag. |
| Split ratios don't match target | Rounding bias with small sample sizes. Increase samples or adjust `compute_counts`. |

---

## Future Enhancement Ideas

- Recursive subfolder scanning (for `split_dataset.py`)
- Random redistribution of remainder samples
- Hash integrity check after copy completion
- Progress bar (`tqdm`) or parallel copying

---

Feel free to share specific command examples or feature requests for your environment.

