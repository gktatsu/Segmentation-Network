#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT_SCRIPT="$SCRIPT_DIR/split_dataset.py"
FORMAT_SCRIPT="$SCRIPT_DIR/format_dataset.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: run_split_and_format.sh [split_dataset.py options] [--format-dest PATH]

This script runs split_dataset.py followed by format_dataset.py. Pass the same
arguments you would normally provide to split_dataset.py. Optionally supply
--format-dest to override the output directory for format_dataset.py. When not
provided, the formatted dataset is written next to the split output using the
suffix "_formatted".

If you include --dry-run, it is forwarded to split_dataset.py and format_dataset.py
is skipped so you can inspect the planned allocation safely.
EOF
}

FORMAT_DEST=""
RUN_DRY=false
declare -a SPLIT_ARGS=()

while (($#)); do
  case "$1" in
    --dry-run)
      RUN_DRY=true
      SPLIT_ARGS+=("$1")
      shift
      ;;
    --format-dest)
      if (($# < 2)); then
        echo "[run_split_and_format] Error: --format-dest requires a value" >&2
        exit 1
      fi
      FORMAT_DEST="$2"
      shift 2
      ;;
    --format-dest=*)
      FORMAT_DEST="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      SPLIT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SPLIT_ARGS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

OUT_VALUE=""
idx=0
while [[ $idx -lt ${#SPLIT_ARGS[@]} ]]; do
  arg="${SPLIT_ARGS[$idx]}"
  if [[ "$arg" == --out ]]; then
    next_idx=$((idx + 1))
    if [[ $next_idx -ge ${#SPLIT_ARGS[@]} ]]; then
      echo "[run_split_and_format] Error: --out requires a value" >&2
      exit 1
    fi
    OUT_VALUE="${SPLIT_ARGS[$next_idx]}"
    break
  elif [[ "$arg" == --out=* ]]; then
    OUT_VALUE="${arg#--out=}"
    break
  fi
  idx=$((idx + 1))
done

if [[ -z "$OUT_VALUE" ]]; then
  echo "[run_split_and_format] Error: please include --out so the formatted output can be routed." >&2
  exit 1
fi

tmp_log="$(mktemp)"
trap 'rm -f "$tmp_log"' EXIT

echo "[run_split_and_format] Running split_dataset.py..."
if ! "$PYTHON_BIN" "$SPLIT_SCRIPT" "${SPLIT_ARGS[@]}" |& tee "$tmp_log"; then
  echo "[run_split_and_format] split_dataset.py failed; aborting." >&2
  exit 1
fi

split_output_root="$(grep -E 'SPLIT_DATASET_OUTPUT_ROOT=' "$tmp_log" | tail -n1 | cut -d= -f2-)"
if [[ -z "$split_output_root" ]]; then
  echo "[run_split_and_format] Could not detect the final split output root. Ensure split_dataset.py is up to date." >&2
  exit 1
fi

if [[ "$RUN_DRY" == true ]]; then
  echo "[run_split_and_format] Dry-run requested; skipping format_dataset.py."
  exit 0
fi

if [[ -z "$FORMAT_DEST" ]]; then
  FORMAT_DEST="${split_output_root}_formatted"
fi

echo "[run_split_and_format] Running format_dataset.py..."
if ! "$PYTHON_BIN" "$FORMAT_SCRIPT" --source "$split_output_root" --dest "$FORMAT_DEST"; then
  echo "[run_split_and_format] format_dataset.py failed; aborting." >&2
  exit 1
fi

log_found=false
mkdir -p "$FORMAT_DEST"
while IFS= read -r -d '' log_file; do
  log_found=true
  base_name="$(basename "$log_file")"
  mv "$log_file" "$FORMAT_DEST/$base_name"
done < <(find "$split_output_root" -maxdepth 1 -type f \
  \( -name 'split_log_*.csv' -o -name 'split_summary_*.txt' \) -print0)

if [[ "$log_found" == true ]]; then
  echo "[run_split_and_format] Moved split logs into $FORMAT_DEST"
else
  echo "[run_split_and_format] No split logs found to move."
fi

rm -rf "$split_output_root"
echo "[run_split_and_format] Removed intermediate split directory: $split_output_root"

echo "[run_split_and_format] Done. Split output: $split_output_root"
echo "[run_split_and_format] Formatted dataset: $FORMAT_DEST"
