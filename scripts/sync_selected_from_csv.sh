
set -euo pipefail

REMOTE="${1:?Usage: sync_selected_from_csv.sh <rclone_remote_name> (e.g. ect_drive)}"
CSV="configs/province_links.csv"
SELECT="configs/provinces.txt"
DEST="data/raw"

if [[ ! -f "$CSV" ]]; then
  echo "Missing $CSV" >&2
  exit 1
fi
if [[ ! -f "$SELECT" ]]; then
  echo "Missing $SELECT" >&2
  exit 1
fi

mkdir -p "$DEST" data/logs

# Iterate CSV rows (skip header)
tail -n +2 "$CSV" | while IFS=, read -r province url; do
  # Trim quotes/spaces
  province="$(echo "$province" | sed 's/^"//; s/"$//; s/^[[:space:]]*//; s/[[:space:]]*$//')"
  url="$(echo "$url" | sed 's/^"//; s/"$//; s/^[[:space:]]*//; s/[[:space:]]*$//')"

  [[ -z "$province" || -z "$url" ]] && continue

  # Only proceed if province is in configs/provinces.txt (exact match)
  if ! grep -Fxq "$province" "$SELECT"; then
    continue
  fi

  # Extract Google Drive folder ID from common URL pattern
  folder_id="$(echo "$url" | sed -n 's|.*drive/folders/\([^?\/]*\).*|\1|p')"
  if [[ -z "$folder_id" ]]; then
    echo "Could not extract folder id for $province from: $url" >&2
    continue
  fi
echo "=== Syncing $province (folder_id=$folder_id) ==="

rclone copy "${REMOTE}:" "${DEST}/${province}" \
  --drive-root-folder-id "$folder_id" \
  --create-empty-src-dirs \
  --transfers 8 \
  --checkers 16 \
  --progress \
  --stats 10s \
  -vv \
  --log-file "data/logs/rclone_${province}.log" \
  --log-format "date,time"

done



