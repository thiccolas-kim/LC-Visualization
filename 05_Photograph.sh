#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

# ============================================================
# SETTINGS
# ============================================================

ROOT="/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni/group_averages"
OUT="$ROOT/pictures"

# Approximate true 3x magnification in FSLeyes
ZOOM=1792

# World coordinates (same as before)
X=-4.248732
Y=-39.49332
Z=-26.03781

# Screenshot resolution
WIDTH=1600
HEIGHT=1000

# Reference image for centering calculation
REF="$ROOT/MTw/MTw_echo-01_N10_mean.nii.gz"

# ============================================================
# CHECK REFERENCE
# ============================================================

if [[ ! -f "$REF" ]]; then
    echo "ERROR: Missing reference image: $REF"
    exit 1
fi

mkdir -p "$OUT"

# ============================================================
# CALCULATE CENTERING
# ============================================================

centres=$(python3 - "$REF" "$X" "$Y" "$Z" <<'PY'
import sys
import subprocess
import itertools

image = sys.argv[1]
target = [float(v) for v in sys.argv[2:5]]

header = subprocess.check_output(
    ["fslhd", image], text=True
)

fields = {}
for line in header.splitlines():
    parts = line.split()
    if parts:
        fields[parts[0]] = parts[1:]

dims = [int(fields[f"dim{i}"][0]) for i in range(1, 4)]

prefix = "sto_xyz" if int(fields["sform_code"][0]) > 0 else "qto_xyz"

affine = [
    [float(v) for v in fields[f"{prefix}:{i}"][:4]]
    for i in range(1, 4)
]

lo = [float("inf")] * 3
hi = [float("-inf")] * 3

for corner in itertools.product(*[(-0.5, n - 0.5) for n in dims]):
    world = [
        sum(affine[i][j] * corner[j] for j in range(3)) + affine[i][3]
        for i in range(3)
    ]
    for i in range(3):
        lo[i] = min(lo[i], world[i])
        hi[i] = max(hi[i], world[i])

centre = [
    2 * (target[i] - lo[i]) / (hi[i] - lo[i]) - 1
    for i in range(3)
]

print(*[f"{v:.8f}" for v in centre])
PY
)

read -r CX CY CZ <<< "$centres"

# ============================================================
# COMMON FSLeyes VIEW SETTINGS
# ============================================================

VIEW_ARGS=(
    --scene ortho
    --hideCursor
    --displaySpace world
    --worldLoc "$X" "$Y" "$Z"
    --xzoom "$ZOOM"
    --yzoom "$ZOOM"
    --zzoom "$ZOOM"
    --xcentre "$CY" "$CZ"
    --ycentre "$CX" "$CZ"
    --zcentre "$CX" "$CY"
)

# ============================================================
# RENDER ALL MEAN IMAGES
# ============================================================

count=0

for contrast in MTw PDw T1w; do

    case "$contrast" in
        MTw)
            MIN=750
            MAX=950
            ;;
        PDw)
            MIN=889
            MAX=1089
            ;;
        T1w)
            MIN=511
            MAX=711
            ;;
        *)
            echo "ERROR: Unknown contrast $contrast"
            exit 1
            ;;
    esac

    echo ""
    echo "Processing $contrast images with display range $MIN-$MAX"

    for file in "$ROOT/$contrast/"*_mean.nii.gz; do
        filename=$(basename "$file" .nii.gz)
        output="$OUT/${filename}.png"

        echo "Rendering: $filename"

        fsleyes render \
            "${VIEW_ARGS[@]}" \
            --size "$WIDTH" "$HEIGHT" \
            --outfile "$output" \
            "$file" \
            -a 100 \
            -dr "$MIN" "$MAX"

        count=$((count + 1))
    done
done

echo ""
echo "Finished saving $count screenshots."
echo "Saved to: $OUT"
