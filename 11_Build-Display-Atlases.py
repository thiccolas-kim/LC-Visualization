#!/usr/bin/env python3
"""Build six one-checkbox FSLeyes atlas overlays with a shared anatomical LUT.

Requirements: numpy and nibabel (python3 -m pip install numpy nibabel)

Example on the cluster:
  python3 build_display_atlases.py
  bash /data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni/atlases/0_Display/open_fsleyes.sh

Note: This script merges labels but DOES NOT perform image registration.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import stat
import sys
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependency. Install with: python3 -m pip install numpy nibabel") from exc

DEFAULT_ROOT = Path(
    "/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/"
    "nighres/echoes_to_mni"
)

# 0 = background (not included in LUT); anatomical IDs are stable across atlases.
# Each tuple is: output integer, descriptive label, HTML-style RGB hex.
STRUCTURES = {
    "LC":        (1,  "LC | Locus coeruleus",                        "#2986DE"),
    "LPB":       (2,  "LPB | Lateral parabrachial",                  "#F08436"),
    "LDTg-CGPn": (3,  "LDTg-CGPn | Laterodorsal tegmental / CG",    "#8753C7"),
    "PTg":       (4,  "PTg | Pedunculotegmental",                    "#23B59C"),
    "SubC":      (5,  "SubC | Subcoeruleus",                         "#DC5A96"),
    "MPB":       (6,  "MPB | Medial parabrachial",                   "#E7B21F"),
    "PnO-PnC":   (7,  "PnO-PnC | Pontine reticular nuclei",         "#986A39"),
    "PBC":       (8,  "PBC | Parabrachial complex",                  "#E5A453"),
    "LDTg":      (9,  "LDTg | Laterodorsal tegmental",               "#A17BE1"),
    "MnR":       (10, "MnR | Median raphe",                          "#F0705B"),
    "DR":        (11, "DR | Dorsal raphe",                           "#E64B62"),
    "VTA":       (12, "VTA | Ventral tegmental area",                "#2CB66E"),
    "PnO":       (13, "PnO | Oral pontine reticular",                "#B48058"),
    "PAG":       (14, "PAG | Periaqueductal gray",                  "#9CAE2D"),
    "mRt":       (15, "mRt | Mesencephalic reticular formation",    "#21A6C3"),
    "LL":        (16, "LL | Lateral lemniscus",                      "#4E66D9"),
    "SCPCR":     (17, "SCPCR | Cerebellorubral tract",               "#B048B3"),
    "SCPCT":     (18, "SCPCT | Cerebellothalamic tract",             "#ED8231"),
    "SCPSC":     (19, "SCPSC | Anterior spinocerebellar tract",     "#268D8A"),
    "ML":        (20, "ML | Medial lemniscus",                       "#A97545"),
    "Ye_0":     (101, "Ye | 0-5% probability (green)",              "#26B447"),
    "Ye_5":     (102, "Ye | 5-25% probability (yellow)",            "#FFDB26"),
    "Ye_25":    (103, "Ye | >25% probability (red)",                "#F33F34"),
}

BIANCARDI_CODES = {
    "PnO-PnC": "PnO_PnC", "MPB": "MPB", "SubC": "SubC",
    "LPB": "LPB", "LDTg-CGPn": "LDTg_CGPn", "PTg": "PTg", "LC": "LC",
}

SHI_CODES = {
    "LL": "LL", "SCPCR": "SCPCR", "SCPCT": "SCPCT",
    "SCPSC": "SCPSC", "ML": "ML",
}

OUTPUTS = [
    ("Ye", "Ye_combined.nii.gz"),
    ("Biancardi_0.25", "Biancardi_0.25_combined.nii.gz"),
    ("Biancardi_0.35", "Biancardi_0.35_combined.nii.gz"),
    ("Edlow", "Edlow_combined.nii.gz"),
    ("Shi_0.35", "Shi_0.35_combined.nii.gz"),
    ("Shi_0.50", "Shi_0.50_combined.nii.gz"),
]


def label_id(name: str) -> int:
    return STRUCTURES[name][0]


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D atlas, got shape {img.shape}: {path}")
    return img


def same_grid(a, b, source: Path):
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4, rtol=0):
        raise ValueError(
            f"Different voxel grid/affine: {source}\n"
            "Masks within ONE combined atlas must already share a registered grid. "
            "Do not combine unregistered images by voxel index."
        )


def mask_from(path: Path, first, threshold: float = 0.5):
    img = load(path)
    same_grid(first, img, path)
    data = np.asanyarray(img.dataobj)
    # binary_* masks use >0.5; Ye probability-thresholded maps use >0.
    # This avoids dropping low (but nonzero) values from the Ye >0% map.
    mask = data > threshold
    if np.issubdtype(data.dtype, np.floating):
        np.logical_and(mask, np.isfinite(data), out=mask)
    count = int(np.count_nonzero(mask))
    if count == 0:
        print(f"    WARNING: empty mask: {path.name}")
    return mask


def save_label_volume(outpath: Path, labels, reference, description: str):
    if labels.dtype != np.uint8:
        raise TypeError("Expected uint8 label array")
    hdr = reference.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["cal_min"] = 0
    hdr["cal_max"] = int(labels.max())
    hdr["descrip"] = description[:79].encode("ascii", errors="replace")
    result = nib.Nifti1Image(labels, reference.affine, header=hdr)
    # Retain qform/sform when set in the source (and thus retain spatial metadata).
    for getter, setter in ((reference.get_qform, result.set_qform),
                           (reference.get_sform, result.set_sform)):
        transform, code = getter(coded=True)
        if transform is not None and code:
            setter(transform, int(code))
    tmp = outpath.with_name(outpath.stem + ".tmp.nii.gz")
    nib.save(result, str(tmp))
    os.replace(tmp, outpath)
    print(f"  Saved: {outpath}")


def combine_binary_masks(outpath: Path, mask_files: list[tuple[str, Path]], name: str):
    missing = [str(p) for _, p in mask_files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required masks:\n  " + "\n  ".join(missing))
    reference = load(mask_files[0][1])
    labels = np.zeros(reference.shape, dtype=np.uint8)
    print(f"\n[{name}] grid = {reference.shape}")
    total_overwritten = 0
    # The order intentionally puts high-priority small nuclei (especially LC) last.
    for struct, path in mask_files:
        mask = mask_from(path, reference)
        overlap = int(np.count_nonzero(mask & (labels != 0)))
        labels[mask] = label_id(struct)
        total_overwritten += overlap
        print(f"  {struct:<11} {path.name:<34} "
              f"voxels={int(mask.sum()):>8,}; previously labeled overlap={overlap:,}")
        del mask
    if total_overwritten:
        print(f"  NOTE: {total_overwritten:,} repeated/overlapping assignments; later "
              "structures take precedence, especially LC.")
    save_label_volume(outpath, labels, reference, name)


def biancardi_files(root: Path, cutoff: str):
    folder = root / "atlases" / "Biancardi" / "registered_AHEAD" / f"binary_{cutoff}"
    result = []
    for struct, base in BIANCARDI_CODES.items():
        for side in ("l", "r"):
            result.append((struct, folder / f"{base}_{side}_AHEAD.nii.gz"))
    return result


def shi_files(root: Path, cutoff: str):
    folder = root / "atlases" / "Shi" / "registered_AHEAD" / f"binary_{cutoff}"
    result = []
    for struct, base in SHI_CODES.items():
        for side in ("L", "R"):
            result.append((struct, folder / f"{base}{side}_AHEAD.nii.gz"))
    return result


def build_ye(root: Path, out: Path):
    folder = root / "atlases" / "Ye" / "LC_7T_prob"
    files = [folder / f"LCTMP_n53_5SD_prob{x}.nii.gz" for x in (0, 5, 25)]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Missing Ye probability masks: " + ", ".join(str(p) for p in files if not p.is_file()))
    first = load(files[0])
    labels = np.zeros(first.shape, dtype=np.uint8)
    masks = []
    print(f"\n[Ye] grid = {first.shape}; retaining original Ye spatial coordinates")
    # Each file is presumed to be a separately thresholded >0, >5, >25 mask,
    # not an unthresholded floating-point probability image.
    for threshold, source in zip((0, 5, 25), files):
        mask = mask_from(source, first, threshold=0.0)
        print(f"  >{threshold:>2}% file: {source.name}: {int(mask.sum()):,} voxels")
        labels[mask] = label_id(f"Ye_{threshold}")
        masks.append(mask)
    if np.any(masks[2] & ~masks[1]) or np.any(masks[1] & ~masks[0]):
        print("  WARNING: Ye masks are not perfectly nested. Verify their source "
              "thresholding and alignment before interpreting the categories.")
    save_label_volume(out / "Ye_combined.nii.gz", labels, first, "Ye probabilities: G 0-5, Y 5-25, R >25")


def normalize_edlow_name(name: str) -> str | None:
    """Map human-readable AAN LUT label names (and side suffixes) to our keys."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    tokens = [t for t in tokens if t not in {"aan", "left", "right", "l", "r", "v2p0", "v2"}]
    flat = "".join(tokens)
    if not flat or flat in {"background", "unknown", "none"}:
        return None
    # Compare full normalized names, not loose substrings: e.g. DR != LDTg.
    aliases = {
        "LC": {"lc", "locuscoeruleus", "locuscaeruleus"},
        "PBC": {"pbc", "parabrachialcomplex", "parabrachialnucleus", "parabrachialnuclei"},
        "LDTg": {"ldtg", "laterodorsaltegmental", "laterodorsaltegmentalnucleus"},
        "MnR": {"mnr", "medianraphe", "medianraphenucleus"},
        "DR": {"dr", "dorsalraphe", "dorsalraphenucleus"},
        "VTA": {"vta", "ventraltegmentalarea", "ventraltegmental"},
        "PnO": {"pno", "oralpontinereticular", "oralpontinereticularnucleus", "pontinereticularoral"},
        "PAG": {"pag", "periaqueductalgray", "periaqueductalgrey"},
        "mRt": {"mrt", "mesencephalicreticularformation", "mesencephalicreticular"},
        "PTg": {"ptg", "pedunculotegmental", "pedunculopontine", "pedunculopontinenucleus", "pedunculopontinetegmental"},
    }
    for key, candidates in aliases.items():
        if flat in candidates:
            return key
    return None


def read_edlow_lut(path: Path):
    """Support both FSLeyes: id r g b name, and FreeSurfer: id name r g b a."""
    if not path.is_file():
        raise FileNotFoundError(f"Edlow label dictionary missing: {path}")
    mapping = {}
    raw = {}
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            old_id = int(parts[0])
        except (IndexError, ValueError):
            continue  # header/comment
        if old_id == 0 or len(parts) < 5:
            continue
        try:
            float(parts[1]); float(parts[2]); float(parts[3])
            # FSLeyes format: numeric RGB precedes the human-readable name
            name = " ".join(parts[4:])
        except ValueError:
            # FreeSurfer format: name before numeric RGB(A)
            name = parts[1]
        if not name:
            raise ValueError(f"Empty Edlow label name on line {n} of {path}")
        if old_id in raw:
            raise ValueError(f"Duplicate Edlow label ID {old_id} in {path}")
        raw[old_id] = name
        mapping[old_id] = normalize_edlow_name(name)
    if not raw:
        raise ValueError(f"Could not read any labels from Edlow LUT: {path}")
    return raw, mapping


def labels_present(data):
    """Discover integer label codes in bounded axial chunks (low extra RAM)."""
    seen = set()
    for z in range(0, data.shape[2], 8):
        slab = data[:, :, z : z + 8]
        if np.issubdtype(slab.dtype, np.floating):
            if not np.all(np.isfinite(slab)):
                raise ValueError("Edlow atlas contains NaN/Inf label values")
            if np.any(np.abs(slab - np.rint(slab)) > 1e-3):
                raise ValueError(
                    "Edlow AAN_Brainstem is not a discrete integer-label image. "
                    "Check that nearest-neighbor interpolation was used during registration."
                )
        seen.update(int(x) for x in np.unique(slab))
    return seen


def build_edlow(root: Path, out: Path):
    folder = root / "atlases" / "Edlow" / "registered_AHEAD"
    brainstem_path = folder / "AAN_Brainstem_AHEAD.nii.gz"
    lc_path = folder / "AAN_LC_AHEAD.nii.gz"
    lut = folder / "Edlow_AAN.lut"
    first = load(brainstem_path)
    raw, mapping = read_edlow_lut(lut)
    data = np.asanyarray(first.dataobj)
    present = labels_present(data) - {0}
    unknown = {i: raw.get(i, "NOT IN LUT") for i in sorted(present)
               if i not in mapping or mapping[i] is None}
    if unknown:
        raise ValueError(
            "Edlow codes cannot be reliably identified from Edlow_AAN.lut: "
            f"{unknown}. Please inspect that LUT and add an explicit alias to "
            "normalize_edlow_name() rather than guessing labels."
        )
    keys_present = {mapping[i] for i in present}
    mandatory = {"PBC", "LDTg", "MnR", "DR", "VTA", "PnO", "PAG", "mRt", "PTg"}
    missing = mandatory - keys_present
    if missing:
        raise ValueError(
            f"Edlow label volume/LUT is missing requested structures: {sorted(missing)}. "
            f"Codes actually present: { {i: raw[i] for i in sorted(present)} }. "
            "Inspect Edlow_AAN.lut and the original AAN label image."
        )
    labels = np.zeros(first.shape, dtype=np.uint8)
    print(f"\n[Edlow] grid = {first.shape}; using named IDs from Edlow_AAN.lut")
    for orig in sorted(present):
        key = mapping[orig]
        if key == "LC":
            # LC is applied last using the separate LC file (more specific mask).
            continue
        mask = (data == orig)
        labels[mask] = label_id(key)
        print(f"  input={orig:<4} {raw[orig]:<28} => {key:<6} voxels={int(mask.sum()):,}")
    del data
    lc = mask_from(lc_path, first)
    print(f"  Separate LC mask: {int(lc.sum()):,} voxels; overwrites overlapping labels")
    labels[lc] = label_id("LC")
    save_label_volume(out / "Edlow_combined.nii.gz", labels, first, "Edlow AAN with shared brainstem colors")


def write_lut(out: Path):
    lines = ["# Shared FSLeyes lookup table: label_value R G B anatomy"]
    for _, (value, name, rgb) in sorted(STRUCTURES.items(), key=lambda x: x[1][0]):
        rgb = rgb.lstrip("#")
        comps = [int(rgb[i:i+2], 16) / 255 for i in (0, 2, 4)]
        lines.append(f"{value:<4} {comps[0]:.6f} {comps[1]:.6f} {comps[2]:.6f} {name}")
    path = out / "common_brainstem.lut"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved shared LUT: {path}")


def write_legend(out: Path):
    lines = ["value\tstructure\thex_color"]
    for _, (value, name, rgb) in sorted(STRUCTURES.items(), key=lambda x: x[1][0]):
        lines.append(f"{value}\t{name}\t{rgb}")
    (out / "color_legend.tsv").write_text("\n".join(lines) + "\n")


def write_launcher(root: Path, out: Path):
    mtw = root / "group_averages" / "SRC" / "MTw" / "N10" / "MTw_echo-01_N10_mean.nii.gz"
    path = out / "open_fsleyes.sh"
    # First input is an optional alternate MTw image, so the same combined
    # atlases can be viewed on an individual subject AHEAD-registered MTw.
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"OUT={shlex.quote(str(out))}",
        f"DEFAULT_MTW={shlex.quote(str(mtw))}",
        'MTW="${1:-$DEFAULT_MTW}"',
        'if [[ ! -f "$MTW" ]]; then',
        '  echo "MTw image not found: $MTW" >&2',
        '  echo "Use: bash open_fsleyes.sh /path/to/your/AHEAD_registered_MTw.nii.gz" >&2',
        '  exit 1',
        'fi',
        'if ! command -v fsleyes >/dev/null 2>&1; then',
        '  echo "fsleyes was not found in PATH; load your FSL/FSLeyes environment." >&2',
        '  exit 1',
        'fi',
        '# Load all 6 as a SINGLE checkbox each. Only Biancardi 0.25 starts visible.',
        'exec fsleyes -wl -0.06064215 -37.965 -20.10426 -xz 350 -yz 350 -zz 350 \\',
        '  "$MTW" -n "MTw background" \\',
    ]
    for i, (key, filename) in enumerate(OUTPUTS):
        hidden = " -d" if key != "Biancardi_0.25" else ""
        more = " \\" if i < len(OUTPUTS) - 1 else ""
        lines.append(
            f'  "$OUT/{filename}" -n "{key}" -ot label -l "$OUT/common_brainstem.lut" '
            f'-a 65{hidden}{more}'
        )
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Saved FSLeyes launcher: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Path to echoes_to_mni")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder (default: ROOT/atlases/0_Display)")
    parser.add_argument("--only", choices=[key for key, _ in OUTPUTS], nargs="*",
                        help="Build only selected atlas variants")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out or (root / "atlases" / "0_Display")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_lut(out)
    write_legend(out)
    write_launcher(root, out)

    tasks = [
        ("Ye", lambda: build_ye(root, out)),
        ("Biancardi_0.25", lambda: combine_binary_masks(
            out / "Biancardi_0.25_combined.nii.gz", biancardi_files(root, "0.25"), "Biancardi 0.25")),
        ("Biancardi_0.35", lambda: combine_binary_masks(
            out / "Biancardi_0.35_combined.nii.gz", biancardi_files(root, "0.35"), "Biancardi 0.35")),
        ("Edlow", lambda: build_edlow(root, out)),
        ("Shi_0.35", lambda: combine_binary_masks(
            out / "Shi_0.35_combined.nii.gz", shi_files(root, "0.35"), "Shi 0.35")),
        ("Shi_0.50", lambda: combine_binary_masks(
            out / "Shi_0.50_combined.nii.gz", shi_files(root, "0.50"), "Shi 0.50")),
    ]
    errors = []
    selected = set(args.only) if args.only else {key for key, _ in tasks}
    for name, task in tasks:
        if name not in selected:
            continue
        try:
            task()
        except (ValueError, OSError) as exc:
            errors.append(name)
            print(f"\nERROR building {name}:\n  {exc}", file=sys.stderr)

    if errors:
        print(f"\nSome atlas files were NOT created: {', '.join(errors)}", file=sys.stderr)
        print("Correct the reported issue(s) and rerun the script.", file=sys.stderr)
        return 1
    print("\nSUCCESS. Open FSLeyes with:")
    print(f"  bash {out / 'open_fsleyes.sh'}")
    print("Toggle a single FSLeyes overlay checkbox per atlas variant.")
    print("NOTE: Ye remains in its native spatial frame; ensure it really aligns with MTw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
