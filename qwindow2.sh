#!/usr/bin/env bash
# qwindow2.sh — AHEAD-space echo-series QC in FSLeyes.
# Loads 4 MTw + 6 PDw + 6 T1w echoes + LC MT-on/off (18 images total).
# Dataset: N05 mean, N10 mean, or an individual subject.
# Usage: ./qwindow2.sh [5|10|s SUBJECT|SUBJECT] [--auto-range] [--dry-run]
# Examples: ./qwindow2.sh 10
#           ./qwindow2.sh 5 --auto-range
#           ./qwindow2.sh 004
#           ./qwindow2.sh s 011
# When no dataset is supplied, the script asks interactively.

set -euo pipefail
shopt -s nullglob

ROOT="${QVIEW_ROOT:-/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni}"
SESSION="ses-02"
ZOOM=1792
# FSLeyes colormap: Hot [brain colours] (iso)
LC_CMAP="brain_colours_1hot_iso"
X=-4.248732
Y=-39.49332
Z=-26.03781

DATASET=""
SUBJECT=""
AUTO_RANGE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: qwindow2.sh [5|10|s SUBJECT|SUBJECT] [--auto-range] [--dry-run]

  5, 05, n5, n05       N05 mean images
  10, n10             N10 mean images
  4, 004, sub-004     An individual subject (numbers other than 5/10)
  s 5                 Individual sub-005 (distinguishes from N05)
  --auto-range        Let FSLeyes choose image ranges (presets are default)
  --dry-run           Validate paths and print command; do not start FSLeyes
  -h, --help          Print this help

Without a dataset argument, an interactive prompt will appear.
EOF
}

normalize_subject() {
    local input="${1#sub-}"
    if [[ ! "$input" =~ ^[0-9]{1,3}$ ]]; then
        echo "ERROR: Invalid subject number: $1" >&2
        exit 1
    fi
    printf -v SUBJECT 'sub-%03d' "$((10#$input))"
}

while (($#)); do
    case "$1" in
        --auto-range) AUTO_RANGE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        5|05|n5|N5|n05|N05)
            [[ -z "$DATASET" ]] || { echo 'ERROR: Multiple dataset selections.' >&2; exit 1; }
            DATASET="N05"; shift ;;
        10|n10|N10)
            [[ -z "$DATASET" ]] || { echo 'ERROR: Multiple dataset selections.' >&2; exit 1; }
            DATASET="N10"; shift ;;
        s|subject)
            [[ -z "$DATASET" ]] || { echo 'ERROR: Multiple dataset selections.' >&2; exit 1; }
            if (($# >= 2)) && [[ "$2" != --* ]]; then
                normalize_subject "$2"; shift 2
            else
                read -r -p 'Subject number (e.g. 004 or 011): ' answer
                normalize_subject "$answer"; shift
            fi
            DATASET="SUBJECT" ;;
        sub-*|[0-9]|[0-9][0-9]|[0-9][0-9][0-9])
            [[ -z "$DATASET" ]] || { echo 'ERROR: Multiple dataset selections.' >&2; exit 1; }
            normalize_subject "$1"
            DATASET="SUBJECT"; shift ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo 'Which images should Window 2 open?'
    echo '  5       N05 means (5 subjects)'
    echo '  10      N10 means (10 subjects)'
    echo '  s       Individual subject (asks for number)'
    read -r -p 'Choice [10]: ' answer
    answer="${answer:-10}"
    case "${answer,,}" in
        5|05|n5|n05) DATASET="N05" ;;
        10|n10) DATASET="N10" ;;
        s|subject)
            read -r -p 'Subject number (e.g. 004 or 011): ' number
            normalize_subject "$number"
            DATASET="SUBJECT" ;;
        sub-*|[0-9]|[0-9][0-9]|[0-9][0-9][0-9])
            normalize_subject "$answer"
            DATASET="SUBJECT" ;;
        *) echo "ERROR: Unknown choice: $answer" >&2; exit 1 ;;
    esac
fi

if [[ "$DATASET" == "SUBJECT" ]]; then
    LABEL="$SUBJECT"
    ANAT="$ROOT/$SUBJECT/$SESSION/anat/AHEAD"
else
    LABEL="$DATASET"
fi

# Find a subject-specific, AHEAD-aligned qMRI image unambiguously.
subject_qmri() {
    local contrast="$1" echo="$2"
    local dir="$ANAT/$contrast"
    local -a matches=(
        "$dir"/"${SUBJECT}_${SESSION}_"*acq-"${contrast}"_*echo-"${echo}"_*part-mag*reorient_align-ahead_def-img.nii*
    )
    if ((${#matches[@]} != 1)); then
        printf 'ERROR: Expected 1 %s echo-%s in %s; found %s.\n' \
            "$contrast" "$echo" "$dir" "${#matches[@]}" >&2
        for m in "${matches[@]}"; do printf '  %s\n' "$m" >&2; done
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

resolve_qmri() {
    local contrast="$1" echo="$2"
    if [[ "$DATASET" == "SUBJECT" ]]; then
        subject_qmri "$contrast" "$echo"
    else
        printf '%s\n' "$ROOT/group_averages/$contrast/${contrast}_echo-${echo}_${DATASET}_mean.nii.gz"
    fi
}

resolve_lc() {
    local state="$1"
    if [[ "$DATASET" == "SUBJECT" ]]; then
        printf '%s\n' "$ANAT/LCslab/${SUBJECT}_${SESSION}_${state}_chunk-LCslab_MTR_in-PDw_header_align-ahead_def-img.nii.gz"
    else
        printf '%s\n' "$ROOT/group_averages/LCslab/LCslab_${state}_${DATASET}_mean.nii.gz"
    fi
}

# Build the overlays in a consistent order, but display only the first one.
# Loading every echo visibly makes the last opaque image obscure the rest.
images=()
image_count=0
first_image=""

add_image() {
    local file="$1" label="$2" low="$3" high="$4" cmap="${5:-}"
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing input image: $file" >&2
        exit 1
    fi
    if [[ -z "$first_image" ]]; then first_image="$file"; fi

    images+=( "$file" -n "$label" -a 100 )
    if [[ -n "$cmap" ]]; then images+=( -cm "$cmap" ); fi
    if ((AUTO_RANGE == 0)); then images+=( -dr "$low" "$high" ); fi
    if ((image_count > 0)); then images+=( -d ); fi
    ((image_count += 1))
    printf '  %2d. %-28s %s\n' "$image_count" "$label" "$file"
}

printf 'Selected: %s | Preset intensity ranges: %s\n' \
    "$LABEL" "$([[ "$AUTO_RANGE" == 0 ]] && echo YES || echo NO)"

for echo in 01 02 03 04; do
    add_image "$(resolve_qmri MTw "$echo")" "MTw / E${echo} / ${LABEL}" 750 950
done
for echo in 01 02 03 04 05 06; do
    add_image "$(resolve_qmri PDw "$echo")" "PDw / E${echo} / ${LABEL}" 900 1100
done
for echo in 01 02 03 04 05 06; do
    add_image "$(resolve_qmri T1w "$echo")" "T1w / E${echo} / ${LABEL}" 511 711
done
add_image "$(resolve_lc mt-on)"  "LC / MT-on  / ${LABEL}" 200 400 "$LC_CMAP"
add_image "$(resolve_lc mt-off)" "LC / MT-off / ${LABEL}" 300 500 "$LC_CMAP"

if ! command -v fslhd >/dev/null 2>&1; then
    echo 'ERROR: fslhd not found. Load FSL first.' >&2
    exit 1
fi
if ((DRY_RUN == 0)) && ! command -v fsleyes >/dev/null 2>&1; then
    echo 'ERROR: fsleyes not found. Load FSLeyes first.' >&2
    exit 1
fi

# FSLeyes zoom is anchored to the display canvas. Convert your preferred
# world-coordinate location to each canvas' normalized centre using fslhd.
centres=$(python3 - "$first_image" "$X" "$Y" "$Z" <<'PY'
import sys, subprocess, itertools
image = sys.argv[1]
target = [float(x) for x in sys.argv[2:5]]
header = subprocess.check_output(['fslhd', image], text=True)
fields = {}
for line in header.splitlines():
    items = line.split()
    if items:
        fields[items[0]] = items[1:]
dims = [int(fields[f'dim{i}'][0]) for i in range(1, 4)]
prefix = 'sto_xyz' if int(fields['sform_code'][0]) > 0 else 'qto_xyz'
affine = [[float(x) for x in fields[f'{prefix}:{i}'][:4]] for i in range(1, 4)]
lo, hi = [float('inf')]*3, [float('-inf')]*3
for corner in itertools.product(*[(-0.5, n - 0.5) for n in dims]):
    world = [sum(affine[i][j]*corner[j] for j in range(3)) + affine[i][3]
             for i in range(3)]
    for i in range(3):
        lo[i] = min(lo[i], world[i]); hi[i] = max(hi[i], world[i])
centre = [2*(target[i]-lo[i])/(hi[i]-lo[i])-1 for i in range(3)]
print(*[f'{v:.8f}' for v in centre])
PY
)
read -r CX CY CZ <<< "$centres"

cmd=(
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
    "${images[@]}"
)

printf '\nReady: %d images. Initially visible: MTw echo 01.\n' "$image_count"
printf 'AHEAD coords: %s %s %s | Zoom: %s | Alpha: 100%%\n' "$X" "$Y" "$Z" "$ZOOM"
if ((DRY_RUN)); then
    printf '\nWould execute:\n'
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    "${cmd[@]}"
fi
