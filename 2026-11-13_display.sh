#!/usr/bin/env bash
# Launch FSLeyes for LC/qMRI group means, sub-004, and brainstem atlases.
# Install at: /data/u_kimn_software/Projects/LC-Averaging/2026-11-13_display.sh
set -euo pipefail

ROOT="${LC_DISPLAY_ROOT:-/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni}"
AVG="$ROOT/group_averages"
ATLAS="$ROOT/atlases"
SUB004="$ROOT/sub-004/ses-02/anat/AHEAD"
FSLEYES="${FSLEYES_BIN:-fsleyes}"

# Shared location and orthographic zoom for N10, N5, 004, and atlases.
XYZ=(4.086349 -38.37062 -24.5)
ZOOM=2415

usage() {
  cat <<'HELP'
Usage: ./2026-11-13_display.sh MODE [SHI_LOW]

Modes:
  n10        N10 mean: MTw echo 1, LC slabs, PD, MTsat, R2*, R1
  n5         N5 mean: same overlays (accepts N5/N05 naming)
  004        Subject 004: registered AHEAD MTw, LC slab(s), MPM maps
  atlases    MTw N10 + combined Ye, Biancardi, Edlow, Shi, LC atlases
  shi        MTw N10 + continuous bilateral Shi tract maps

Optional:
  SHI_LOW    Lower tract-density threshold for 'shi' (default: 0.05).
             Example: ./2026-11-13_display.sh shi 0.10

Aliases: mpm-n10, mpm-n5, sub-004, atlas, shi-gradient

N10/N5/004/atlases: world coordinates (4.086349, -38.37062, -24.5), zoom 2415.
Shi gradient preserves your earlier coordinates (1.939358, -37.965, -20.10426),
zoom 2452.
Group/sub-004 display ranges: MTw 750-950; LC 220-500 (hidden);
PD 0-170; MTsat 0-0.8; R2* 0-50; R1 0.1-0.75.
HELP
}

warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Find the first existing path/pattern. Wildcards are supported.
first_match() {
  local pattern match
  for pattern in "$@"; do
    while IFS= read -r match; do
      if [[ -f "$match" ]]; then
        printf '%s\n' "$match"
        return 0
      fi
    done < <(compgen -G "$pattern" || true)
  done
  return 1
}

# FSLeyes accepts per-image options directly after their image pathname.
args=(-s default -hc -wl "${XYZ[@]}" -xz "$ZOOM" -yz "$ZOOM" -zz "$ZOOM")
COUNT=0
MISSING=0

add_volume() {
  local name="$1" lo="$2" hi="$3" visibility="$4" requirement="$5"
  shift 5
  local found
  found="$(first_match "$@" || true)"
  if [[ -z "$found" ]]; then
    if [[ "$requirement" == optional ]]; then
      warn "$name not found; skipping."
    else
      warn "$name not found; check its file location."
      MISSING=$((MISSING + 1))
    fi
    return 0
  fi
  args+=("$found" -n "$name" -ot volume -dr "$lo" "$hi")
  [[ "$visibility" == hidden ]] && args+=(-d)
  COUNT=$((COUNT + 1))
}

# Build candidates for both modern .../N5/... and earlier flat layouts,
# and for both N05 and N5 filename suffixes.
add_group_volume() {
  local cohort="$1" subdir="$2" stem="$3" display_name="$4"
  local lo="$5" hi="$6" visibility="$7" requirement="$8"
  local folder tag
  local -a folders tags candidates=()
  if [[ "$cohort" == N10 ]]; then
    folders=(N10)
    tags=(N10)
  else
    folders=(N5 N05)
    tags=(N05 N5)
  fi
  for folder in "${folders[@]}"; do
    for tag in "${tags[@]}"; do
      candidates+=("$AVG/$subdir/$folder/${stem}_${tag}_mean.nii.gz")
    done
  done
  for tag in "${tags[@]}"; do
    candidates+=("$AVG/$subdir/${stem}_${tag}_mean.nii.gz")
  done
  add_volume "$display_name" "$lo" "$hi" "$visibility" "$requirement" "${candidates[@]}"
}

add_group() {
  local cohort="$1"
  add_group_volume "$cohort" SRC/MTw MTw_echo-01 "MTw $cohort Echo 1" 750 950 visible required
  add_group_volume "$cohort" LCslab LCslab_mt-off "LC mt-off $cohort" 220 500 hidden required
  add_group_volume "$cohort" LCslab LCslab_mt-on "LC mt-on $cohort" 220 500 hidden required
  add_group_volume "$cohort" MPM/PDmap PDmap "PD $cohort" 0 170 visible required
  add_group_volume "$cohort" MPM/MTsat MTsat "MTsat $cohort" 0 0.8 visible required
  add_group_volume "$cohort" MPM/R2starmap R2starmap "R2* $cohort" 0 50 visible required
  add_group_volume "$cohort" MPM/R1map R1map "R1 $cohort" 0.1 0.75 visible required
}

add_subject004() {
  add_volume 'MTw sub-004 Echo 1' 750 950 visible required \
    "$SUB004/MTw/sub-004_ses-02_acq-MTw_rec-loraksRsos_echo-01_flip-1_mt-on_part-mag_desc-LCPCAout_desc-undistortedJac_MPM_reorient_align-ahead_def-img.nii" \
    "$SUB004/MTw/*echo-01*align-ahead_def-img.nii" \
    "$SUB004/MTw/*echo-01*align-ahead_def-img.nii.gz"

  add_volume 'LC mt-off sub-004' 220 500 hidden required \
    "$SUB004/LCslab/sub-004_ses-02_mt-off_chunk-LCslab_MTR_in-PDw_header_align-ahead_def-img.nii.gz" \
    "$SUB004/LCslab/*mt-off*align-ahead_def-img.nii*"

  # In the provided sub-004 listing, only mt-off exists. If an mt-on
  # registered AHEAD image is added later, it will be loaded automatically.
  add_volume 'LC mt-on sub-004' 220 500 hidden optional \
    "$SUB004/LCslab/*mt-on*align-ahead_def-img.nii*"

  add_volume 'PD sub-004' 0 170 visible required \
    "$SUB004/MPM/PDmap/sub-004_ses-02_PDmap_align-ahead_def-img.nii.gz" \
    "$SUB004/MPM/PDmap/*PDmap*align-ahead_def-img.nii*"
  add_volume 'MTsat sub-004' 0 0.8 visible required \
    "$SUB004/MPM/MTsat/sub-004_ses-02_MTsat_align-ahead_def-img.nii.gz" \
    "$SUB004/MPM/MTsat/*MTsat*align-ahead_def-img.nii*"
  add_volume 'R2* sub-004' 0 50 visible required \
    "$SUB004/MPM/R2starmap/sub-004_ses-02_R2starmap_align-ahead_def-img.nii.gz" \
    "$SUB004/MPM/R2starmap/*R2starmap*align-ahead_def-img.nii*"
  add_volume 'R1 sub-004' 0.1 0.75 visible required \
    "$SUB004/MPM/R1map/sub-004_ses-02_R1map_align-ahead_def-img.nii.gz" \
    "$SUB004/MPM/R1map/*R1map*align-ahead_def-img.nii*"
}

add_atlas() {
  local relpath="$1" name="$2" visibility="$3"
  local path="$ATLAS/0_Display/$relpath"
  if [[ ! -f "$path" ]]; then
    warn "$name not found: $path"
    MISSING=$((MISSING + 1))
    return 0
  fi
  args+=("$path" -n "$name" -ot label -l "$ATLAS/0_Display/common_brainstem.lut" -a 65)
  [[ "$visibility" == hidden ]] && args+=(-d)
  COUNT=$((COUNT + 1))
}

add_atlases() {
  add_group_volume N10 SRC/MTw MTw_echo-01 'MTw N10 Average' 750 950 visible required
  [[ -f "$ATLAS/0_Display/common_brainstem.lut" ]] || \
    die "Missing atlas LUT: $ATLAS/0_Display/common_brainstem.lut"
  add_atlas full/Ye_combined.nii.gz 'Ye Full' visible
  add_atlas full/Biancardi_0.25_combined.nii.gz 'Biancardi 0.25 Full' hidden
  add_atlas full/Biancardi_0.35_combined.nii.gz 'Biancardi 0.35 Full' hidden
  add_atlas full/Edlow_combined.nii.gz 'Edlow Full' hidden
  add_atlas full/Shi_0.35_combined.nii.gz 'Shi 0.35 Full' hidden
  add_atlas LC/Ye_LC_probabilities.nii.gz 'Ye LC' hidden
  add_atlas LC/Biancardi_0.25_LC.nii.gz 'Biancardi 0.25 LC' hidden
  add_atlas LC/Biancardi_0.35_LC.nii.gz 'Biancardi 0.35 LC' hidden
  add_atlas LC/Edlow_LC.nii.gz 'Edlow LC' hidden
}

add_shi() {
  local low="$1" roi side cmap file
  local prob="$ATLAS/Shi/registered_AHEAD/probabilistic"
  local site="$ATLAS/Shi/fsleyes_site"
  [[ -d "$prob" ]] || die "Shi probabilistic directory missing: $prob"
  [[ -d "$site" ]] || die "Custom Shi colour-map configuration missing: $site"
  export FSLEYES_SITE_CONFIG_DIR="$site"
  add_group_volume N10 SRC/MTw MTw_echo-01 'MTw N10 Average' 750 950 visible required
  for roi in LL SCPCR SCPCT SCPSC ML; do
    case "$roi" in
      LL)    cmap=shi_ll ;;
      SCPCR) cmap=shi_scpcr ;;
      SCPCT) cmap=shi_scpct ;;
      SCPSC) cmap=shi_scpsc ;;
      ML)    cmap=shi_ml ;;
    esac
    for side in L R; do
      file="$prob/${roi}${side}_AHEAD.nii.gz"
      if [[ ! -f "$file" ]]; then
        warn "Shi $roi $side missing: $file"
        MISSING=$((MISSING + 1))
        continue
      fi
      args+=("$file" -n "${roi}_${side} (Shi Continuous)" \
        -ot volume -cm "$cmap" -dr "$low" 1.0 -cr "$low" 1.0 -a 90)
      COUNT=$((COUNT + 1))
    done
  done
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
  n10|mpm-n10) mode=n10 ;;
  n5|mpm-n5) mode=n5 ;;
  004|sub-004|subject-004) mode=004 ;;
  atlases|atlas) mode=atlases ;;
  shi|shi-gradient) mode=shi ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$mode" != shi && $# -gt 1 ]] || [[ $# -gt 2 ]]; then
  die "Unexpected argument(s). Run './2026-11-13_display.sh --help'."
fi

case "$mode" in
  n10) add_group N10 ;;
  n5) add_group N5 ;;
  004) add_subject004 ;;
  atlases) add_atlases ;;
  shi)
    # Preserve coordinates and zoom from your existing Shi-gradient script.
    args=(-s default -hc -wl 1.939358 -37.965 -20.10426 \
          -xz 2452 -yz 2452 -zz 2452)
    low="${2:-0.05}"
    [[ "$low" =~ ^(0(\.[0-9]+)?|\.[0-9]+)$ ]] || die 'SHI_LOW must be at least 0 and less than 1.'
    add_shi "$low"
    ;;
esac

[[ "$COUNT" -gt 0 ]] || die "No matching images found for mode '$mode'."
if [[ "$MISSING" -gt 0 ]]; then
  warn "$MISSING requested image(s) could not be found; opening $COUNT available image(s)."
fi
command -v "$FSLEYES" >/dev/null 2>&1 || die "Cannot find '$FSLEYES' on PATH. Activate FSL/FSLeyes first."
printf 'Opening %s images in FSLeyes (mode: %s).\n' "$COUNT" "$mode"
exec "$FSLEYES" "${args[@]}"
