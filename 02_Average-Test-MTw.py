#!/usr/bin/env python3
"""Average AHEAD-aligned MTw echo-01 images for the first 5, 10, and 15 eligible subjects.

Eligibility here means: exactly one expected file exists, it can be read,
its 3D voxel grid/affine match the first accepted image, and it has some
finite nonzero voxels. This does NOT establish anatomical registration QC.
Optionally provide a text file listing subjects that have passed visual QC.
"""

import csv
import re
import time
from datetime import timedelta
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(
    "/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/"
    "derivatives/nighres/echoes_to_mni"
)
SESSION = "ses-02"
CONTRAST = "MTw"
ECHO = 1
GROUP_SIZES = (5, 10, 15)
OUTPUT_DIR = ROOT / "group_averages" / "MTw_echo-01"

# OPTIONAL: list one visually QC-approved subject per line, e.g. sub-004.
# Leave as None to use all subjects with an existing, geometrically compatible image.
QC_PASSED_FILE = None  # e.g. Path('/data/u_kimn_software/LC_qc_passed.txt')

# Native resampling used padding='zero'. Excluding zero-valued and nonfinite
# voxels prevents zero padding at image edges from lowering the mean.
IGNORE_ZERO_PADDING = True

# Tolerance for the NIfTI affine matrix, measured approximately in millimeters.
AFFINE_ATOL = 1e-4


def subject_sort_key(path):
    """Natural order: sub-002 before sub-010."""
    name = path.name if isinstance(path, Path) else str(path)
    match = re.fullmatch(r"sub-(\d+)", name)
    return (0, int(match.group(1))) if match else (1, name)


def discover_images(root, session, contrast, echo, allowed_subjects):
    """Find exactly one AHEAD-aligned image for each subject."""
    candidates = []
    exclusions = []
    subject_dirs = sorted(
        (p for p in root.glob('sub-*') if p.is_dir()),
        key=subject_sort_key,
    )

    for sub_dir in subject_dirs:
        subject = sub_dir.name

        if allowed_subjects is not None and subject not in allowed_subjects:
            continue

        anat_dir = sub_dir / session / 'anat'
        # Match exact echo number via regex after globbing: echo-01, not echo-010.
        files = sorted(
            p for p in anat_dir.glob(
                f'*acq-{contrast}*echo-*part-mag*reorient_align-ahead_def-img.nii*'
            )
            if re.search(rf'_echo-0*{echo}_', p.name)
            and p.name.endswith(('.nii', '.nii.gz'))
        )

        if len(files) == 0:
            exclusions.append((subject, 'MISSING_ALIGNED_IMAGE', str(anat_dir)))
        elif len(files) > 1:
            exclusions.append((subject, 'AMBIGUOUS_ALIGNED_IMAGES', ' | '.join(map(str, files))))
        else:
            candidates.append((subject, files[0]))

    return candidates, exclusions, len(subject_dirs)


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_group_mean(n, sums, counts, reference_image, selected):
    """Save voxelwise mean, valid-subject count, and the subject manifest."""
    valid = counts > 0
    mean = np.zeros(sums.shape, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=valid)

    stem = f'MTw_echo-01_N{n:02d}'
    mean_path = OUTPUT_DIR / f'{stem}_mean.nii.gz'
    count_path = OUTPUT_DIR / f'{stem}_coverage.nii.gz'
    subjects_path = OUTPUT_DIR / f'{stem}_subjects.csv'

    mean_img = nib.Nifti1Image(
        mean, reference_image.affine, header=reference_image.header.copy()
    )
    mean_img.set_data_dtype(np.float32)
    nib.save(mean_img, str(mean_path))

    coverage_img = nib.Nifti1Image(
        counts.copy(), reference_image.affine,
        header=reference_image.header.copy()
    )
    coverage_img.set_data_dtype(np.uint8)
    nib.save(coverage_img, str(count_path))

    write_csv(
        subjects_path,
        ['subject', 'image_path'],
        [(subject, str(path)) for subject, path in selected],
    )

    print(f'  Saved mean:     {mean_path}', flush=True)
    print(f'  Saved coverage: {count_path}', flush=True)
    print(f'  Saved subjects: {subjects_path}', flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    allowed_subjects = None
    if QC_PASSED_FILE is not None:
        qc_path = Path(QC_PASSED_FILE)
        if not qc_path.is_file():
            raise FileNotFoundError(f'QC approved-subject file does not exist: {qc_path}')
        allowed_subjects = {
            line.strip()
            for line in qc_path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        }
        print(f'Using visual-QC approved list: {qc_path} ({len(allowed_subjects)} subjects)')
    else:
        print('No QC whitelist: checking file presence and image geometry only.')
        print('IMPORTANT: this does not verify anatomical alignment.')

    candidates, exclusions, discovered = discover_images(
        ROOT, SESSION, CONTRAST, ECHO, allowed_subjects
    )
    print(f'Found {discovered} subject directories; {len(candidates)} candidate images.')
    print(f'Will save averages for N={GROUP_SIZES}.\n', flush=True)

    reference_img = None
    sums = None
    counts = None
    selected = []

    for subject, img_path in candidates:
        if len(selected) >= max(GROUP_SIZES):
            break

        try:
            img = nib.load(str(img_path))

            if len(img.shape) != 3:
                raise ValueError(f'Expected 3D NIfTI, got shape {img.shape}')

            if reference_img is not None:
                if img.shape != reference_img.shape:
                    raise ValueError(
                        f'Grid mismatch: {img.shape} != reference {reference_img.shape}'
                    )
                if not np.allclose(
                    img.affine, reference_img.affine, rtol=0, atol=AFFINE_ATOL
                ):
                    raise ValueError('Affine mismatch from reference image')

            # One volume at a time; never stack all subjects in memory.
            data = np.asarray(img.dataobj, dtype=np.float32)
            if IGNORE_ZERO_PADDING:
                valid = np.isfinite(data) & (data != 0)
            else:
                valid = np.isfinite(data)

            if not np.any(valid):
                raise ValueError('Image contains no valid voxels')

        except Exception as exc:
            exclusions.append((
                subject, 'READ_OR_GEOMETRY_ERROR',
                f'{img_path} | {type(exc).__name__}: {exc}'
            ))
            print(f'WARNING: skipping {subject}: {type(exc).__name__}: {exc}', flush=True)
            continue

        if reference_img is None:
            reference_img = img
            sums = np.zeros(img.shape, dtype=np.float32)
            counts = np.zeros(img.shape, dtype=np.uint8)  # safe for N<=255

        np.add(sums, data, out=sums, where=valid)
        np.add(counts, 1, out=counts, where=valid)
        selected.append((subject, img_path))

        elapsed = time.monotonic() - started
        n_done = len(selected)
        estimated_remaining = elapsed / n_done * (max(GROUP_SIZES) - n_done)
        print(
            f'[{n_done:02d}/{max(GROUP_SIZES):02d}] {subject} accepted | '
            f'elapsed {timedelta(seconds=int(elapsed))} | '
            f'approx. ETA {timedelta(seconds=int(estimated_remaining))}',
            flush=True,
        )

        if n_done in GROUP_SIZES:
            print(f'\nSaving {n_done}-subject average...', flush=True)
            save_group_mean(n_done, sums, counts, reference_img, selected)
            print('', flush=True)

        del data, valid

    exclusions_path = OUTPUT_DIR / 'excluded_subjects.csv'
    write_csv(exclusions_path, ['subject', 'reason', 'details'], exclusions)

    print('\n================ SUMMARY ================')
    print(f'Accepted: {len(selected)} subjects')
    print('Accepted subjects: ' + (', '.join(s for s, _ in selected) or '(none)'))
    print(f'Excluded/flagged: {len(exclusions)} (see {exclusions_path})')
    for n in GROUP_SIZES:
        print(f'  N={n}: ' + ('SAVED' if len(selected) >= n else 'NOT SAVED (insufficient eligible subjects)'))
    print(f'Total runtime: {timedelta(seconds=int(time.monotonic() - started))}')
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

