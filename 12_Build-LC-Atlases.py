#!/usr/bin/env python3
"""Organize existing combined atlas outputs into full/ and LC/ for FSLeyes.

Run AFTER build_display_atlases.py (does not re-run that script).

Output structure under echoes_to_mni/atlases/0_Display/:
  common_brainstem.lut                 (kept from original script)
  color_legend.tsv                     (kept from original script)
  full/
    Ye_combined.nii.gz
    Biancardi_0.25_combined.nii.gz
    Biancardi_0.35_combined.nii.gz
    Edlow_combined.nii.gz
    Shi_0.35_combined.nii.gz
    Shi_0.50_combined.nii.gz
    open_fsleyes.sh
  LC/
    Ye_LC_probabilities.nii.gz        (same 3 Ye probability classes)
    Biancardi_0.25_LC.nii.gz          (only integer label 1 = LC)
    Biancardi_0.35_LC.nii.gz
    Edlow_LC.nii.gz
    open_fsleyes.sh
  open_fsleyes.sh                      (shortcut to full/ launcher)
  open_LC_fsleyes.sh                   (shortcut to LC/ launcher)

Shi maps tracks/tracts and has no LC label, so no Shi LC-only file is made.
No image registrations or resampling are performed.

Dependencies: numpy and nibabel (already required by build_display_atlases.py).

Usage:
  python3 organize_lc_display_atlases.py
  bash /data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/\
    nighres/echoes_to_mni/atlases/0_Display/open_LC_fsleyes.sh
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import stat
import sys
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy or nibabel. Use the Python environment "
        "that successfully ran build_display_atlases.py."
    ) from exc

DEFAULT_ROOT = Path(
    "/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/"
    "nighres/echoes_to_mni"
)

FULL_FILES = [
    "Ye_combined.nii.gz",
    "Biancardi_0.25_combined.nii.gz",
    "Biancardi_0.35_combined.nii.gz",
    "Edlow_combined.nii.gz",
    "Shi_0.35_combined.nii.gz",
    "Shi_0.50_combined.nii.gz",
]

# Original label value for LC from the first script (same blue color in the LUT).
LC_LABEL_VALUE = 1

LC_EXTRACTIONS = {
    "Biancardi_0.25_combined.nii.gz": "Biancardi_0.25_LC.nii.gz",
    "Biancardi_0.35_combined.nii.gz": "Biancardi_0.35_LC.nii.gz",
    "Edlow_combined.nii.gz": "Edlow_LC.nii.gz",
}

# (Display label, file name) - one single FSLeyes checkbox per entry.
FULL_DISPLAYS = [
    ("Ye | probabilities", "Ye_combined.nii.gz"),
    ("Biancardi | 0.25", "Biancardi_0.25_combined.nii.gz"),
    ("Biancardi | 0.35", "Biancardi_0.35_combined.nii.gz"),
    ("Edlow | full", "Edlow_combined.nii.gz"),
    ("Shi | 0.35", "Shi_0.35_combined.nii.gz"),
    ("Shi | 0.50", "Shi_0.50_combined.nii.gz"),
]
LC_DISPLAYS = [
    ("Ye | LC probabilities", "Ye_LC_probabilities.nii.gz"),
    ("Biancardi | LC 0.25", "Biancardi_0.25_LC.nii.gz"),
    ("Biancardi | LC 0.35", "Biancardi_0.35_LC.nii.gz"),
    ("Edlow | LC", "Edlow_LC.nii.gz"),
]


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy before replacing the destination so interrupted copies don't corrupt it."""
    temp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(src, temp)
        os.replace(temp, dst)
    finally:
        if temp.exists():
            temp.unlink()


def move_full_overlays(out: Path, full_dir: Path) -> None:
    """Move original outputs without recomputing or replacing existing full files."""
    missing = [
        filename for filename in FULL_FILES
        if not (out / filename).is_file() and not (full_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Can't find the following outputs from build_display_atlases.py "
            "in either 0_Display/ or 0_Display/full/:\n  "
            + "\n  ".join(missing)
            + "\nCheck that the original script completed successfully."
        )

    print("\n[1/3] Organizing existing full atlas files")
    for filename in FULL_FILES:
        src = out / filename
        dest = full_dir / filename
        if dest.exists():
            if src.exists():
                print(f"  WARNING: Both locations have {filename}. Keeping full/ "
                      "version; root copy was NOT removed or overwritten.")
            else:
                print(f"  Already organized: full/{filename}")
            continue
        shutil.move(str(src), str(dest))
        print(f"  Moved: {filename} -> full/{filename}")


def extract_lc(source: Path, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        print(f"  Already exists (skipped): LC/{destination.name}")
        return

    image = nib.load(str(source))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D label atlas: {source}, got {image.shape}")

    # The first script writes uint8 integer labels; all non-LC anatomy becomes 0.
    input_labels = np.asanyarray(image.dataobj)
    if not np.issubdtype(input_labels.dtype, np.integer):
        if not np.all(np.isfinite(input_labels)) or not np.all(
            input_labels == np.rint(input_labels)
        ):
            raise ValueError(f"Input is not a discrete label image: {source}")

    lc_labels = np.asarray(input_labels == LC_LABEL_VALUE, dtype=np.uint8)
    voxel_count = int(np.count_nonzero(lc_labels))
    if voxel_count == 0:
        raise ValueError(
            f"LC label {LC_LABEL_VALUE} not present in {source.name}. "
            "Please check whether the previous atlas-building script succeeded."
        )

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 1
    header["descrip"] = b"LC-only label=1; shared FSLeyes LUT"

    result = nib.Nifti1Image(lc_labels, image.affine, header=header)
    for getter, setter in (
        (image.get_qform, result.set_qform),
        (image.get_sform, result.set_sform),
    ):
        matrix, code = getter(coded=True)
        if matrix is not None and code:
            setter(matrix, int(code))

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp.nii.gz"
    )
    try:
        nib.save(result, str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"  Created: LC/{destination.name} | LC voxels: {voxel_count:,}")


def create_lc_overlays(full_dir: Path, lc_dir: Path, overwrite: bool) -> None:
    print("\n[2/3] Creating LC-only overlays")

    # Ye is already entirely an LC probability atlas. Keep all three colored
    # probability bins in its LC copy, without altering any voxel or coordinates.
    ye_src = full_dir / "Ye_combined.nii.gz"
    ye_dest = lc_dir / "Ye_LC_probabilities.nii.gz"
    if ye_dest.exists() and not overwrite:
        print("  Already exists (skipped): LC/Ye_LC_probabilities.nii.gz")
    else:
        atomic_copy(ye_src, ye_dest)
        print("  Copied: LC/Ye_LC_probabilities.nii.gz "
              "(three probability bins retained)")

    for full_name, lc_name in LC_EXTRACTIONS.items():
        extract_lc(full_dir / full_name, lc_dir / lc_name, overwrite)

    print("  Shi excluded from LC/ because none of its labels are LC.")


def write_executable(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_launcher(
    target_dir: Path,
    shared_lut: Path,
    default_mtw: Path,
    overlays: list[tuple[str, str]],
    visible_index: int,
) -> None:
    path = target_dir / "open_fsleyes.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"DIR={shlex.quote(str(target_dir))}",
        f"LUT={shlex.quote(str(shared_lut))}",
        f"DEFAULT_MTW={shlex.quote(str(default_mtw))}",
        'MTW="${1:-$DEFAULT_MTW}"',
        'if [[ ! -f "$MTW" ]]; then',
        '  echo "MTw image not found: $MTW" >&2',
        '  echo "Pass an AHEAD-registered MTw path as the first argument." >&2',
        '  exit 1',
        'fi',
        'if ! command -v fsleyes >/dev/null 2>&1; then',
        '  echo "fsleyes not in PATH; load your FSL environment." >&2',
        '  exit 1',
        'fi',
        'exec fsleyes -wl -0.06064215 -37.965 -20.10426 -xz 350 -yz 350 -zz 350 \\',
        '  "$MTW" -n "MTw background" \\',
    ]
    for i, (label, filename) in enumerate(overlays):
        hidden = " -d" if i != visible_index else ""
        final_slash = " \\" if i < len(overlays) - 1 else ""
        lines.append(
            f'  "$DIR/{filename}" -n {shlex.quote(label)} '
            f'-ot label -l "$LUT" -a 65{hidden}{final_slash}'
        )
    write_executable(path, lines)
    print(f"  Wrote {path}")


def write_shortcut(path: Path, launcher: Path) -> None:
    write_executable(path, [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"exec bash {shlex.quote(str(launcher))} \"$@\"",
    ])
    print(f"  Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Path to echoes_to_mni")
    parser.add_argument("--out", type=Path,
                        help="Existing 0_Display directory (default ROOT/atlases/0_Display)")
    parser.add_argument("--overwrite-lc", action="store_true",
                        help="Regenerate existing LC/ outputs from full/ volumes")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out or root / "atlases" / "0_Display").expanduser().resolve()
    if not out.is_dir():
        print(f"ERROR: Original output directory does not exist: {out}", file=sys.stderr)
        return 1

    shared_lut = out / "common_brainstem.lut"
    if not shared_lut.is_file():
        print(f"ERROR: Original LUT not found: {shared_lut}\n"
              "Run/check build_display_atlases.py first.", file=sys.stderr)
        return 1

    full_dir = out / "full"
    lc_dir = out / "LC"
    full_dir.mkdir(exist_ok=True)
    lc_dir.mkdir(exist_ok=True)

    try:
        move_full_overlays(out, full_dir)
        create_lc_overlays(full_dir, lc_dir, args.overwrite_lc)
        print("\n[3/3] Writing FSLeyes launchers")
        mtw = (root / "group_averages" / "SRC" / "MTw" / "N10" /
               "MTw_echo-01_N10_mean.nii.gz")
        write_launcher(full_dir, shared_lut, mtw, FULL_DISPLAYS, visible_index=1)
        write_launcher(lc_dir, shared_lut, mtw, LC_DISPLAYS, visible_index=1)
        # Keep the launcher's original name working after the .nii.gz moves.
        write_shortcut(out / "open_fsleyes.sh", full_dir / "open_fsleyes.sh")
        write_shortcut(out / "open_LC_fsleyes.sh", lc_dir / "open_fsleyes.sh")
    except (OSError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nSUCCESS: existing full atlases moved to full/; LC-only files in LC/.")
    print("Use:")
    print(f"  bash {out / 'open_LC_fsleyes.sh'}")
    print(f"  bash {out / 'open_fsleyes.sh'}  # full atlases")
    print("LC and full overlays share the same original common_brainstem.lut.")
    print("Ye retains its original spatial coordinates: verify alignment with MTw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
