#!/bin/bash

# copy_voices.sh - Automate voice preset setup for VibeVoice-FastAPI
# Usage: ./scripts/copy_voices.sh [source_dir] [target_dir]

set -e

# Default directories
SOURCE_DIR="${1:-/home/bumpyclock/Projects/VibeVoice/demo/voices}"
TARGET_DIR="${2:-./voices}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up voice presets..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy WAV files
if [ -d "$SOURCE_DIR" ]; then
    echo "Copying WAV files..."
    cp "$SOURCE_DIR"/*.wav "$TARGET_DIR/"
    echo "Copied $(ls "$SOURCE_DIR"/*.wav | wc -l) voice files"
else
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Generate metadata.json by parsing filenames
echo "Generating metadata.json..."

# Build JSON content in memory
json_content='{\n  "voices": ['

# Parse each WAV file and extract metadata
first=true
for file in "$TARGET_DIR"/*.wav; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .wav)
        
        # Parse pattern: language-Name_gender[_bgm]
        if [[ $filename =~ ^([a-z]{2})-([A-Za-z]+)_([a-z]+)(_bgm)?$ ]]; then
            language="${BASH_REMATCH[1]}"
            name="${BASH_REMATCH[2]}"
            gender="${BASH_REMATCH[3]}"
            
            # Add comma separator for all but first entry
            if [ "$first" = true ]; then
                first=false
            else
                json_content="$json_content,"
            fi
            
            # Add voice entry
            json_content="$json_content\n    {\n      \"id\": \"$filename\",\n      \"name\": \"$name\",\n      \"language\": \"$language\",\n      \"gender\": \"$gender\"\n    }"
        else
            echo "Warning: Could not parse filename pattern for $filename"
        fi
    fi
done

json_content="$json_content\n  ]\n}"

# Write the complete JSON to file
echo -e "$json_content" > "$TARGET_DIR/metadata.json"

echo "Generated metadata for $(ls "$TARGET_DIR"/*.wav | wc -l) voices"
echo "Voice preset setup complete!"
echo "Directory: $TARGET_DIR"
echo "Files:"
ls -la "$TARGET_DIR"