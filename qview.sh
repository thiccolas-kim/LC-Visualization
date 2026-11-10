#!/usr/bin/env bash
# qview: Interactive FSLeyes launcher for LC / qMRI images (ses-02)
# Range menu: use previous preset, enter custom, or let FSLeyes auto-range.
# Uses existing files only; never modifies images.
set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/data/pt_03187/data/in_vivo/bids}"
LORAKS_ROOT="$BIDS_ROOT/derivatives/LORAKS"
ECHO_ROOT="${ECHO_ROOT:-$LORAKS_ROOT/derivatives/nighres/echoes_to_mni}"
LCPCA_ROOT="${LCPCA_ROOT:-$LORAKS_ROOT/derivatives/LCPCA_distCorr}"
SESSION="ses-02"
ZOOM=1792  # same approximate 3x zoom used in your AHEAD comparisons

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ask() { read -r -p "$1" REPLY || exit 1; }

command -v fsleyes >/dev/null 2>&1 || fail 'fsleyes is not on PATH (load FSL/FSLeyes first).'

printf '\n===== qMRI / LC FSLeyes launcher =====\n'
printf 'Image spaces:\n  1) AHEAD (echoes_to_mni / group means)\n  2) Native PDw (LC slabs aligned to PDw; native PDw echoes)\n  3) Original acquisitions (LCPCA_distCorr / raw LC slabs)\n'
while :; do
    ask 'Space [1/2/3]: '
    case "$REPLY" in 1) SPACE=ahead; break;; 2) SPACE=pdw; break;; 3) SPACE=original; break;; esac
    echo 'Please enter 1, 2, or 3.'
done

while :; do
    ask 'Average or subject? [a/s]: '
    case "${REPLY,,}" in a) MODE=average; break;; s) MODE=subject; break;; esac
    echo 'Please enter a or s.'
done

if [[ "$MODE" == average && "$SPACE" != ahead ]]; then
    echo 'N05/N10 averages are available only in AHEAD space.'
    ask 'Switch to AHEAD space? [Y/n]: '
    case "${REPLY,,}" in n|no) exit 0;; *) SPACE=ahead;; esac
fi

if [[ "$MODE" == subject ]]; then
    while :; do
        ask 'Subject number (e.g., 4, 011, sub-011): '
        input="${REPLY#sub-}"
        if [[ "$input" =~ ^[0-9]{1,3}$ ]]; then
            printf -v SUBJECT 'sub-%03d' "$((10#$input))"
            break
        fi
        echo 'Enter a numeric subject ID, e.g. 004 or sub-011.'
    done
fi

printf '\nImaging types:\n  1) MTw\n  2) PDw\n  3) T1w\n  4) LC slab MT-on\n  5) LC slab MT-off\n'
while :; do
    ask 'Imaging type [1-5]: '
    case "${REPLY,,}" in
        1|mtw) TYPE=MTw; LC=no; break;;
        2|pdw) TYPE=PDw; LC=no; break;;
        3|t1w) TYPE=T1w; LC=no; break;;
        4|on|mt-on|lc-on) TYPE=LCslab; LC=yes; SIDE=on; break;;
        5|off|mt-off|lc-off) TYPE=LCslab; LC=yes; SIDE=off; break;;
    esac
    echo 'Please choose 1-5.'
done

if [[ "$LC" == no ]]; then
    if [[ "$TYPE" == MTw ]]; then MAX_ECHO=4; else MAX_ECHO=6; fi
    while :; do
        ask "Echo number [1-$MAX_ECHO]: "
        if [[ "$REPLY" =~ ^0*[1-9][0-9]*$ ]]; then
            number=$((10#$REPLY))
            if (( number >= 1 && number <= MAX_ECHO )); then
                printf -v ECHO '%02d' "$number"
                break
            fi
        fi
        printf 'Enter an echo between 1 and %d.\n' "$MAX_ECHO"
    done
    case "$TYPE" in
        MTw) LO=750; HI=950;;
        PDw) LO=889; HI=1089;;
        T1w) LO=511; HI=711;;
    esac
else
    if [[ "$SIDE" == on ]]; then LO=0; HI=500; else LO=220; HI=500; fi
fi

if [[ "$MODE" == average ]]; then
    while :; do
        ask 'Number of subjects [5/10]: '
        case "$REPLY" in 5|05) N=05; break;; 10) N=10; break;; esac
        echo 'Enter 5 or 10.'
    done
fi

# Find a single matching input. Never silently select an arbitrary file.
find_image() {
    local directory="$1" pattern="$2"
    [[ -d "$directory" ]] || fail "Directory does not exist: $directory"
    mapfile -d '' -t matches < <(find "$directory" -type f -name "$pattern" -print0 | sort -z)
    (( ${#matches[@]} > 0 )) || fail "No image matched '$pattern' under $directory"
    if (( ${#matches[@]} == 1 )); then
        IMAGE="${matches[0]}"
        return
    fi
    printf '\nMultiple matching images found. Choose one:\n'
    local i pick
    for i in "${!matches[@]}"; do printf '  %d) %s\n' "$((i+1))" "${matches[$i]}"; done
    while :; do
        ask "Image number [1-${#matches[@]}]: "
        pick="$REPLY"
        if [[ "$pick" =~ ^[0-9]+$ ]] && (( 10#$pick >= 1 && 10#$pick <= ${#matches[@]} )); then
            IMAGE="${matches[$((10#$pick-1))]}"
            return
        fi
        echo 'Invalid image number.'
    done
}

if [[ "$MODE" == average ]]; then
    if [[ "$LC" == yes ]]; then
        IMAGE="$ECHO_ROOT/group_averages/LCslab/LCslab_mt-${SIDE}_N${N}_mean.nii.gz"
        LABEL="LC MT-${SIDE} N${N}"
    else
        IMAGE="$ECHO_ROOT/group_averages/$TYPE/${TYPE}_echo-${ECHO}_N${N}_mean.nii.gz"
        LABEL="$TYPE E${ECHO} N${N}"
    fi
    [[ -f "$IMAGE" ]] || fail "Missing group mean: $IMAGE"
else
    ANAT="$ECHO_ROOT/$SUBJECT/$SESSION/anat"
    if [[ "$LC" == yes ]]; then
        LABEL="$SUBJECT LC MT-${SIDE} ($SPACE)"
        case "$SPACE" in
            ahead)
                find_image "$ANAT/AHEAD/LCslab" \
                    "${SUBJECT}_${SESSION}_mt-${SIDE}_chunk-LCslab_MTR_*align-ahead_def-img.nii*";;
            pdw)
                find_image "$ANAT/LCslab_PDw_header" \
                    "${SUBJECT}_${SESSION}_mt-${SIDE}_chunk-LCslab_MTR_in-PDw*.nii*";;
            original)
                find_image "$BIDS_ROOT/$SUBJECT/$SESSION/anat" \
                    "${SUBJECT}_${SESSION}_mt-${SIDE}_chunk-LCslab_MTR.nii*";;
        esac
    else
        LABEL="$SUBJECT $TYPE E${ECHO} ($SPACE)"
        case "$SPACE" in
            ahead)
                find_image "$ANAT/AHEAD/$TYPE" \
                    "*acq-${TYPE}_*echo-${ECHO}_*part-mag*reorient_align-ahead_def-img.nii*";;
            original)
                find_image "$LCPCA_ROOT/$SUBJECT/$SESSION/anat" \
                    "*acq-${TYPE}_*echo-${ECHO}_*part-mag*_MPM.nii*";;
            pdw)
                if [[ "$TYPE" != PDw ]]; then
                    fail "$TYPE reoriented echoes are not necessarily registered to the PDw grid. Only PDw source images are supported here; choose AHEAD or original for $TYPE."
                fi
                find_image "$ANAT/Reoriented" \
                    "*acq-PDw_*echo-${ECHO}_*part-mag*reorient.nii*";;
        esac
    fi
fi

# Choose intensity display range for any image.
# The default option intentionally passes no -dr arguments to FSLeyes.
RANGE_ARGS=()
RANGE_LABEL='FSLeyes automatic (no range override)'
ask "Use the preset display range ($LO-$HI)? [y/N]: "
case "${REPLY,,}" in
    y|yes)
        RANGE_ARGS=(-dr "$LO" "$HI")
        RANGE_LABEL="Preset $LO-$HI"
        ;;
    *)
        ask 'Set a custom display range? [y/N]: '
        case "${REPLY,,}" in
            y|yes)
                while :; do
                    ask 'Enter minimum and maximum (e.g. 0 500): '
                    read -r RANGE_MIN RANGE_MAX RANGE_EXTRA <<< "$REPLY"
                    if [[ -n "${RANGE_EXTRA:-}" || -z "${RANGE_MIN:-}" || -z "${RANGE_MAX:-}" ]]; then
                        echo 'Enter exactly two numbers.'
                        continue
                    fi
                    if python3 - "$RANGE_MIN" "$RANGE_MAX" <<'PYRANGE'
import math, sys
try:
    low, high = map(float, sys.argv[1:])
    assert math.isfinite(low) and math.isfinite(high) and low < high
except (ValueError, AssertionError):
    sys.exit(1)
PYRANGE
                    then
                        RANGE_ARGS=(-dr "$RANGE_MIN" "$RANGE_MAX")
                        RANGE_LABEL="Custom $RANGE_MIN-$RANGE_MAX"
                        break
                    fi
                    echo 'Minimum and maximum must be finite numbers, with minimum < maximum.'
                done
                ;;
            *) ;;  # No -dr: let FSLeyes determine the image range.
        esac
        ;;
esac

# AHEAD location preset; native original LC slab location preset.
COORDS=()
if [[ "$SPACE" == ahead ]]; then
    ask 'Use preset AHEAD LC coordinates (-4.248732, -39.49332, -26.03781)? [Y/n]: '
    case "${REPLY,,}" in n|no) ;; *) COORDS=(-4.248732 -39.49332 -26.03781);; esac
elif [[ "$SPACE" == original && "$LC" == yes ]]; then
    ask 'Use preset original LC slab coordinates (4.567116, -16.09998, -17.62556)? [Y/n]: '
    case "${REPLY,,}" in n|no) ;; *) COORDS=(4.567116 -16.09998 -17.62556);; esac
fi

if (( ${#COORDS[@]} == 0 )); then
    ask 'Set custom world coordinates instead of the default image center? [y/N]: '
    case "${REPLY,,}" in
        y|yes)
            ask 'Enter X Y Z (space-separated): '
            read -r X Y Z extra <<< "$REPLY"
            if [[ -n "${extra:-}" || -z "${X:-}" || -z "${Y:-}" || -z "${Z:-}" ]]; then
                fail 'Please enter exactly three coordinate values.'
            fi
            for value in "$X" "$Y" "$Z"; do
                [[ "$value" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] \
                    || fail "Invalid coordinate: $value"
            done
            COORDS=("$X" "$Y" "$Z")
            ;;
    esac
fi

# If a target location is supplied, center each orthographic canvas at that
# point. Uses fslhd so the user's host Python does not need nibabel.
VIEW_ARGS=()
if (( ${#COORDS[@]} == 3 )); then
    command -v fslhd >/dev/null 2>&1 || fail 'fslhd is required for precise coordinate centering. Load FSL first.'
    centres=$(python3 - "$IMAGE" "${COORDS[@]}" <<'PY'
import sys, subprocess, itertools
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
lo, hi = [float('inf')]*3, [float('-inf')]*3
for corner in itertools.product(*[(-0.5, n-0.5) for n in dims]):
    world = [sum(affine[i][j]*corner[j] for j in range(3)) + affine[i][3]
             for i in range(3)]
    for i in range(3):
        lo[i], hi[i] = min(lo[i], world[i]), max(hi[i], world[i])
centre = [2*(target[i]-lo[i])/(hi[i]-lo[i])-1 for i in range(3)]
print(*[f'{v:.8f}' for v in centre])
PY
    ) || fail 'Could not calculate centering from NIfTI header.'
    read -r CX CY CZ <<< "$centres"
    VIEW_ARGS=(
        --worldLoc "${COORDS[@]}"
        --xcentre "$CY" "$CZ"
        --ycentre "$CX" "$CZ"
        --zcentre "$CX" "$CY"
    )
fi

printf '\nSelected: %s\nImage: %s\nDisplay range: %s | opacity: 100%%\n' "$LABEL" "$IMAGE" "$RANGE_LABEL"
if (( ${#COORDS[@]} == 3 )); then
    printf 'World coordinates: %s %s %s\n' "${COORDS[@]}"
else
    echo 'World coordinates: FSLeyes default'
fi

echo 'Opening FSLeyes (default layout with Overlay List)...'
fsleyes \
    --scene default \
    --hideCursor \
    --displaySpace world \
    --selectedOverlay 0 \
    --xzoom "$ZOOM" --yzoom "$ZOOM" --zzoom "$ZOOM" \
    "${VIEW_ARGS[@]}" \
    "$IMAGE" -n "$LABEL" -a 100 "${RANGE_ARGS[@]}"
