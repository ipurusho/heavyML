#!/usr/bin/env bash
# 00_download_data.sh — Download Kaggle MA dataset + MusicBrainz dump tables
# Idempotent: skips downloads if files already exist
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_DIR/data/raw"
MB_DIR="$PROJECT_DIR/data/musicbrainz"

MB_MIRROR="https://data.metabrainz.org/pub/musicbrainz/data/fullexport"

# --- Kaggle: Every Metal Archives Band ---
kaggle_download() {
    local target_file="$RAW_DIR/metal_bands.csv"
    if [[ -f "$target_file" ]]; then
        echo "[kaggle] metal_bands.csv already exists, skipping."
        return
    fi

    echo "[kaggle] Downloading 'Every Metal Archives Band' dataset..."
    if ! command -v kaggle &>/dev/null; then
        echo "[kaggle] ERROR: kaggle CLI not found. Install with: pip install kaggle"
        echo "[kaggle] Also ensure ~/.kaggle/kaggle.json exists with your API token."
        exit 1
    fi

    kaggle datasets download \
        -d guimacrlh/every-metal-archives-band-october-2024 \
        -p "$RAW_DIR" --unzip

    echo "[kaggle] Done."
}

# --- MusicBrainz: dump tables ---
mb_download() {
    local needed_files=("artist" "url" "l_artist_url" "artist_credit_name" "recording" "release_group" "release" "medium" "track")
    local all_present=true

    for f in "${needed_files[@]}"; do
        if [[ ! -f "$MB_DIR/$f" ]]; then
            all_present=false
            break
        fi
    done

    if $all_present; then
        echo "[musicbrainz] All ${#needed_files[@]} table files already exist, skipping."
        return
    fi

    # Get the latest dump date
    echo "[musicbrainz] Fetching latest dump date..."
    local latest
    latest=$(curl -sL "$MB_MIRROR/LATEST")
    echo "[musicbrainz] Latest dump: $latest"

    local dump_url="$MB_MIRROR/$latest/mbdump.tar.bz2"
    local tarball="$MB_DIR/mbdump.tar.bz2"

    # Download if tarball doesn't exist
    if [[ ! -f "$tarball" ]]; then
        echo "[musicbrainz] Downloading mbdump.tar.bz2 (~3-5GB)..."
        curl -L --progress-bar -o "$tarball" "$dump_url"
    else
        echo "[musicbrainz] Tarball already downloaded."
    fi

    # Extract all tables we need (linkage + album chain)
    echo "[musicbrainz] Extracting artist, url, l_artist_url, artist_credit_name, recording, release_group, release, medium, track..."
    tar -xjf "$tarball" -C "$MB_DIR" \
        --strip-components=1 \
        mbdump/artist \
        mbdump/url \
        mbdump/l_artist_url \
        mbdump/artist_credit_name \
        mbdump/recording \
        mbdump/release_group \
        mbdump/release \
        mbdump/medium \
        mbdump/track

    # Verify extraction
    for f in "${needed_files[@]}"; do
        if [[ ! -f "$MB_DIR/$f" ]]; then
            echo "[musicbrainz] ERROR: Failed to extract $f"
            exit 1
        fi
        local size
        size=$(du -h "$MB_DIR/$f" | cut -f1)
        echo "[musicbrainz] Extracted $f ($size)"
    done

    # Clean up tarball to save space
    echo "[musicbrainz] Removing tarball to save disk space..."
    rm -f "$tarball"

    echo "[musicbrainz] Done."
}

# --- Main ---
echo "=== heavyML Data Download ==="
echo "Project dir: $PROJECT_DIR"
echo ""

mkdir -p "$RAW_DIR" "$MB_DIR"

kaggle_download
echo ""
mb_download

echo ""
echo "=== All downloads complete ==="
