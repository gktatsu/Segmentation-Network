# random_split Utilities

A collection of scripts for preparing paired image datasets for segmentation tasks.

## Included Files

| File | Role |
|------|------|
| `split_dataset.py` | Splits images and masks into `train/valid/test` by ratio, outputs CSV / summary logs |
| `format_dataset.py` | Copies split directories to final submission format (`train_images/`, etc.) with sequential renaming |
| `run_split_and_format.sh` | Shell wrapper that runs both scripts consecutively |
| `rename_images.py` | Sequentially renames images in any directory (supports recursive processing) |

> The README and scripts in the parent directory (`util/datasets/`) are for backward compatibility. Please reference this directory directly going forward.

---

## Requirements

- Python 3.7+ (standard library only)
- Image and mask files must have matching base names (excluding extension)
- Input directories are scanned non-recursively (flatten subfolders beforehand, or use `rename_images.py --recursive`)

---

## Quick Start

### Split Only

```bash
python3 split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /tmp/dataset_split \
    --train 0.6 --valid 0.2 --test 0.2
```

- Output directory name automatically gets a timestamp suffix (`_yyyymmddHHMM`)
- `SPLIT_DATASET_OUTPUT_ROOT=/actual/path` is printed after execution

### Format Only

```bash
python3 format_dataset.py \
    --source /tmp/dataset_split_202512171234 \
    --dest   /tmp/dataset_formatted
```

### Split + Format Together

```bash
bash run_split_and_format.sh \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /tmp/dataset_split \
    --train 0.6 --valid 0.2 --test 0.2
```

- Use `--format-dest /path/to/final` to specify formatted output destination
- Set `PYTHON_BIN` environment variable for different Python interpreter
- Use `--dry-run` to run split dry-run only; formatting is skipped
- On success, intermediate directory is removed and logs are moved to formatted folder

---

## `split_dataset.py` Options

| Option | Description |
|--------|-------------|
| `--images` (required) | Image directory (multiple allowed) |
| `--masks` (required) | Mask directory (multiple allowed) |
| `--out` (required) | Output root (creates `train/valid/test`) |
| `--train`, `--valid`, `--test` | Split ratios (decimals or percentages, auto-normalized) |
| `--seed` | Random seed (default 42) |
| `--move` | Move instead of copy |
| `--dry-run` | Generate logs only; no file operations |
| `--preserve-test-index N` | Keep specified input index as `test` |
| `--preserve-original-test` | When 2 input sets exist, automatically keep index 1 as `test` |
| `--no-append-timestamp` | Disable timestamp suffix on output root |

### Output Logs

- `split_log_<timestamp>.csv` — Records destination and source path for each file
- `split_summary_<timestamp>.txt` — Total pairs, train/valid/test counts

---

## `format_dataset.py` Options

| Option | Description |
|--------|-------------|
| `--source` (required) | `split_dataset.py` output root |
| `--dest` (required) | Formatted output destination (auto-created) |
| `--dry-run` | Show actions without copying |

### Output Structure

```
<dest>/
├── train_images/     (image1.png, image2.png, ...)
├── train_masks/      (mask1.png, mask2.png, ...)
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

---

## `rename_images.py` Options

| Option | Description |
|--------|-------------|
| `--dir` (required) | Target directory |
| `--prefix` | Filename prefix (default `gen_img`) |
| `--start` | Starting number for sequence (default 0) |
| `--zero-pad` | Zero-padding width |
| `--auto-pad` | Auto zero-pad based on max index |
| `--ext` | Target extension(s) (multiple allowed, default `.png`) |
| `--output` | Copy to different directory (omit for in-place rename) |
| `--recursive` | Process all subdirectories |
| `--skip-processed` | Skip if marker file exists |
| `--write-marker` | Create marker file on success |
| `--marker-name` | Marker filename (default `.rename_images_done`) |
| `--overwrite` | Allow overwriting existing files |
| `--dry-run` | Show plan without making changes |

### Usage Examples

```bash
# Single directory
python3 rename_images.py \
    --dir /datasets/raw \
    --prefix sample \
    --start 1 \
    --zero-pad 4

# Recursive processing
python3 rename_images.py \
    --dir /datasets/raw_assets \
    --recursive \
    --prefix dataset \
    --auto-pad \
    --write-marker
```

- Recursive mode auto-detects directories containing only target extension files
- Combined with `--output`, preserves relative path structure to a separate tree
- Sequence resets independently for each directory
- Files are sorted in natural order (0, 1, 2, ..., 9, 10, 11, ...)

---

## `run_split_and_format.sh` Details

### Processing Flow

1. Execute `split_dataset.py`
2. Retrieve `SPLIT_DATASET_OUTPUT_ROOT`
3. Execute `format_dataset.py`
4. Move log files to formatted folder
5. Remove intermediate split directory

### Script-Specific Options

| Option | Description |
|--------|-------------|
| `--format-dest` | Formatted output destination (default is `<split_output>_formatted`) |
| `-h`, `--help` | Show help |

All other arguments are passed directly to `split_dataset.py`.

---

## Notes

- All scripts use standard library only — zero dependencies
- Same behavior on Windows / macOS / Linux (`os.path` / `pathlib` based)
- For detailed usage, also see the parent directory's `util/datasets/README.md`
