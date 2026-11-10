#!/usr/bin/env bash
# qwindow1.sh - Compare one subject with the N05 and N10 group means in FSLeyes.
# Loads 15 AHEAD-space images: MTw E01, PDw E01, T1w E01,
# LC MT-on, LC MT-off for each of SUBJECT, N05, and N10.
# Usage: qwindow1.sh [SUBJECT] [--auto-range] [--dry-run]

set -euo pipefail
shopt -s nullglob

ROOT="${QVIEW_ROOT:-/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni}"
SESSION="ses-02"
SUBJECT="sub-004"
X=-4.248732
Y=-39.49332
Z=-26.03781
ZOOM=1792
LC_CMAP="brain_colours_1hot_iso"  # Hot [brain colours] (iso)
AUTO_RANGE=0
DRY_RUN=0
SUBJECT_PROVIDED=0

usage() {
    cat <<'EOF'
Usage: qwindow1.sh [SUBJECT] [--auto-range] [--dry-run]

SUBJECT        4, 004, sub-004, 11, sub-011, etc. (default: 004)
--auto-range   Use the automatic FSLeyes range instead of image presets
--dry-run      Check inputs and print the command instead of opening FSLeyes
-h, --help     Show help

Loads 15 AHEAD-aligned images, in this order:
  SUBJECT: MTw E01, PDw E01, T1w E01, LC MT-on, LC MT-off
  N05:     MTw E01, PDw E01, T1w E01, LC MT-on, LC MT-off
  N10:     MTw E01, PDw E01, T1w E01, LC MT-on, LC MT-off

Presets: MTw 750-950, PDw 900-1100, T1w 511-711,
         LC MT-on 200-400, LC MT-off 300-500.
The LC images use Hot [brain colours] (iso).
Only the first image is visible initially.
EOF
}

normalize_subject() {
    local value="${1#sub-}"
    if [[ ! "$value" =~ ^[0-9]{1,3}$ ]]; then
        printf 'ERROR: Invalid subject: %s\n' "$1" >&2
        exit 1
    fi
    printf -v SUBJECT 'sub-%03d' "$((10#$value))"
}

while (($# > 0)); do
    case "$1" in
        --auto-range) AUTO_RANGE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        sub-*|[0-9]|[0-9][0-9]|[0-9][0-9][0-9])
            if ((SUBJECT_PROVIDED)); then
                echo 'ERROR: Please specify only one subject.' >&2
                exit 1
            fi
            normalize_subject "$1"
            SUBJECT_PROVIDED=1
            shift ;;
        *) printf 'ERROR: Unknown argument: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
done

ANAT="$ROOT/$SUBJECT/$SESSION/anat/AHEAD"

subject_qmri() {
    local contrast="$1" echo_no="$2"
    local folder="$ANAT/$contrast"
    local -a matches=(
        "$folder"/"${SUBJECT}_${SESSION}_"*acq-"${contrast}"_*echo-"${echo_no}"_*part-mag*reorient_align-ahead_def-img.nii*
    )
    if ((${#matches[@]} != 1)); then
        printf 'ERROR: Expected one %s echo-%s image in %s; found %d.\n' \
            "$contrast" "$echo_no" "$folder" "${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

subject_lc() {
    local mt_state="$1"
    printf '%s\n' "$ANAT/LCslab/${SUBJECT}_${SESSION}_${mt_state}_chunk-LCslab_MTR_in-PDw_header_align-ahead_def-img.nii.gz"
}

mean_qmri() {
    local cohort="$1" contrast="$2"
    printf '%s\n' "$ROOT/group_averages/$contrast/${contrast}_echo-01_${cohort}_mean.nii.gz"
}

mean_lc() {
    local cohort="$1" mt_state="$2"
    printf '%s\n' "$ROOT/group_averages/LCslab/LCslab_${mt_state}_${cohort}_mean.nii.gz"
}

IMAGES=()
COUNT=0
REFERENCE=""

add_image() {
    local filename="$1" name="$2" min="$3" max="$4" cmap="${5:-}"
    if [[ ! -f "$filename" ]]; then
        printf 'ERROR: Image not found: %s\n' "$filename" >&2
        exit 1
    fi

    [[ -n "$REFERENCE" ]] || REFERENCE="$filename"
    IMAGES+=( "$filename" -n "$name" -a 100 )
    [[ -z "$cmap" ]] || IMAGES+=( -cm "$cmap" )
    if ((AUTO_RANGE == 0)); then IMAGES+=( -dr "$min" "$max" ); fi
    if ((COUNT > 0)); then IMAGES+=( -d ); fi
    ((COUNT += 1))
    printf '%2d. %-28s %s\n' "$COUNT" "$name" "$filename"
}

printf 'Window 1: %s + N05 + N10\n' "$SUBJECT"

# Five overlays for the individual subject.
add_image "$(subject_qmri MTw 01)"  "01 $SUBJECT / MTw E01"    750 950
add_image "$(subject_qmri PDw 01)"  "01 $SUBJECT / PDw E01"    900 1100
add_image "$(subject_qmri T1w 01)"  "01 $SUBJECT / T1w E01"    511 711
add_image "$(subject_lc mt-on)"      "01 $SUBJECT / LC MT-on"  200 400 "$LC_CMAP"
add_image "$(subject_lc mt-off)"     "01 $SUBJECT / LC MT-off" 300 500 "$LC_CMAP"

# Five overlays each for the N05 and N10 means.
for COHORT in N05 N10; do
    if [[ "$COHORT" == "N05" ]]; then PREFIX=02; else PREFIX=03; fi
    add_image "$(mean_qmri "$COHORT" MTw)" "$PREFIX $COHORT / MTw E01" 750 950
    add_image "$(mean_qmri "$COHORT" PDw)" "$PREFIX $COHORT / PDw E01" 900 1100
    add_image "$(mean_qmri "$COHORT" T1w)" "$PREFIX $COHORT / T1w E01" 511 711
    add_image "$(mean_lc "$COHORT" mt-on)"  "$PREFIX $COHORT / LC MT-on" 200 400 "$LC_CMAP"
    add_image "$(mean_lc "$COHORT" mt-off)" "$PREFIX $COHORT / LC MT-off" 300 500 "$LC_CMAP"
done

if ! command -v fslhd >/dev/null 2>&1; then
    echo 'ERROR: fslhd not available. Load FSL in this shell.' >&2
    exit 1
fi
if ((DRY_RUN == 0)) && ! command -v fsleyes >/dev/null 2>&1; then
    echo 'ERROR: fsleyes not available in this shell.' >&2
    exit 1
fi

# Convert the requested LC world location to normalised canvas centres.
CENTRES=$(python3 - "$REFERENCE" "$X" "$Y" "$Z" <<'PY'
import itertools
import subprocess
import sys

image = sys.argv[1]
target = [float(v) for v in sys.argv[2:5]]
header = subprocess.check_output(['fslhd', image], text=True)
fields = {}
for line in header.splitlines():
    parts = line.split()
    if parts:
        fields[parts[0]] = parts[1:]

dims = [int(fields[f'dim{i}'][0]) for i in range(1, 4)]
prefix = 'sto_xyz' if int(fields['sform_code'][0]) > 0 else 'qto_xyz'
affine = [[float(v) for v in fields[f'{prefix}:{i}'][:4]] for i in range(1, 4)]
lo = [float('inf')] * 3
hi = [float('-inf')] * 3
for corner in itertools.product(*[(-0.5, n - 0.5) for n in dims]):
    world = [sum(affine[i][j] * corner[j] for j in range(3)) + affine[i][3]
             for i in range(3)]
    for i in range(3):
        lo[i] = min(lo[i], world[i])
        hi[i] = max(hi[i], world[i])
center = [2 * (target[i] - lo[i]) / (hi[i] - lo[i]) - 1 for i in range(3)]
print(*[f'{v:.8f}' for v in center])
PY
)
read -r CX CY CZ <<< "$CENTRES"

COMMAND=(
    fsleyes
    --scene default
    --hideCursor
    --displaySpace world
    --worldLoc "$X" "$Y" "$Z"
    --selectedOverlay 0
    --xzoom "$ZOOM" --yzoom "$ZOOM" --zzoom "$ZOOM"
    --xcentre "$CY" "$CZ"
    --ycentre "$CX" "$CZ"
    --zcentre "$CX" "$CY"
    "${IMAGES[@]}"
)

printf '\nLoaded %d images; only %s MTw E01 is initially visible.\n' "$COUNT" "$SUBJECT"
printf 'LC position: %s, %s, %s | zoom %s\n' "$X" "$Y" "$Z" "$ZOOM"
if ((DRY_RUN)); then
    printf 'Would execute:\n'
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    "${COMMAND[@]}"
fi
