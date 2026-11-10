#!/usr/bin/env python3
"""Compute AHEAD-space mean magnitude images for fixed 5- and 10-subject cohorts.

Inputs:  previously registered MTw/PDw/T1w echoes in echoes_to_mni/sub-*/ses-02/anat.
Outputs: group_averages/{MTw,PDw,T1w}/<contrast>_echo-XX_N05/N10_*.nii.gz

No registration is performed. Files must already be aligned to one common voxel grid.
"""

import csv
import time
from datetime import timedelta
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# SETTINGS: EXACT SUBJECTS TO USE FOR EVERY CONTRAST AND ECHO
# ============================================================

ROOT = Path(
    '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/'
    'derivatives/nighres/echoes_to_mni'
)
OUTPUT_ROOT = ROOT / 'group_averages'
SESSION = 'ses-02'

FIVE_SUBJECTS = [
    'sub-004', 'sub-011', 'sub-012', 'sub-015', 'sub-017'
]
TEN_SUBJECTS = [
    'sub-004', 'sub-009', 'sub-011', 'sub-012', 'sub-013',
    'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-019'
]

# Use the same echoes as the original MPM pipeline.
ECHOES_BY_CONTRAST = {
    'MTw': range(1, 5),  # echoes 01-04
    'PDw': range(1, 7),  # echoes 01-06
    'T1w': range(1, 7),  # echoes 01-06
}

# Match previous averaging behavior: don't treat zero padding as real signal.
EXCLUDE_ZERO_VOXELS = True
SAVE_COVERAGE_MAPS = True

# Recreate output files if this script is run again with these fixed groups.
OVERWRITE_OUTPUTS = True


# ============================================================
# HELPERS
# ============================================================

def validate_cohorts():
    if len(FIVE_SUBJECTS) != 5 or len(set(FIVE_SUBJECTS)) != 5:
        raise ValueError('FIVE_SUBJECTS must have exactly 5 unique subjects')
    if len(TEN_SUBJECTS) != 10 or len(set(TEN_SUBJECTS)) != 10:
        raise ValueError('TEN_SUBJECTS must have exactly 10 unique subjects')
    if not set(FIVE_SUBJECTS).issubset(TEN_SUBJECTS):
        raise ValueError('The five subjects must all be in TEN_SUBJECTS')


def find_registered_image(subject, contrast, echo):
    """Return exactly one AHEAD-aligned echo; otherwise return an error."""
    anat_dir = ROOT / subject / SESSION / 'anat'
    pattern = (
        f'{subject}_{SESSION}_*acq-{contrast}_*echo-{echo:02d}_*'
        f'part-mag*reorient_align-ahead_def-img.nii*'
    )
    matches = sorted(glob(str(anat_dir / pattern)))
    if len(matches) != 1:
        kind = 'MISSING' if not matches else 'AMBIGUOUS'
        return None, (kind, f'{len(matches)} matches for {anat_dir / pattern}')
    return Path(matches[0]), None


def spatial_shape(img):
    """Allow 3D images and 4D images with a singleton fourth dimension."""
    shape = img.shape
    if len(shape) < 3 or any(dim != 1 for dim in shape[3:]):
        raise ValueError(f'Expected a 3D image, got shape {shape}')
    return tuple(shape[:3])


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def preflight():
    """Check every requested file before producing any means."""
    inputs = {}
    problems = []
    reference_shape = None
    reference_affine = None
    reference_path = None

    for contrast, echoes in ECHOES_BY_CONTRAST.items():
        for echo in echoes:
            for subject in TEN_SUBJECTS:
                path, error = find_registered_image(subject, contrast, echo)
                if error:
                    problems.append([subject, contrast, f'{echo:02d}', *error])
                    continue

                try:
                    img = nib.load(str(path))  # reads NIfTI header, not full voxel data
                    shape = spatial_shape(img)
                    affine = img.affine
                    if not np.all(np.isfinite(affine)):
                        raise ValueError('Nonfinite affine matrix')
                    if reference_shape is None:
                        reference_shape = shape
                        reference_affine = affine.copy()
                        reference_path = path
                    elif shape != reference_shape or not np.allclose(
                        affine, reference_affine, rtol=0, atol=1e-5
                    ):
                        raise ValueError(
                            f'Grid differs from {reference_path}: '
                            f'shape {shape} vs {reference_shape}'
                        )
                    inputs[(contrast, echo, subject)] = path
                except Exception as exc:
                    problems.append([
                        subject, contrast, f'{echo:02d}',
                        'INVALID / INCOMPATIBLE', f'{path}: {exc}'
                    ])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT_ROOT / 'fixed_subjects_preflight_issues.csv'
    write_csv(report,
              ['subject', 'contrast', 'echo', 'problem', 'details'],
              problems)

    if problems:
        print(f'\nPreflight FAILED: {len(problems)} issue(s).')
        for subject, contrast, echo, problem, details in problems:
            print(f'  {subject} {contrast} echo-{echo}: {problem}: {details}')
        print(f'Full report: {report}')
        raise SystemExit('No averages written. Fix inputs, then rerun.')

    print(f'Preflight passed: {len(inputs)} registered images on one grid.')
    print(f'Common grid: {reference_shape} | reference: {reference_path}')
    print(f'Issue report (header only): {report}')
    return inputs, reference_shape


def save_nifti(data, reference_path, output_path, dtype):
    """Save data on the reference image's spatial grid."""
    ref = nib.load(str(reference_path))
    header = ref.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)
    result = nib.Nifti1Image(data, ref.affine, header=header)
    nib.save(result, str(output_path))


def save_group(contrast, echo, subjects, inputs, sums, counts):
    n = len(subjects)
    output_dir = OUTPUT_ROOT / contrast
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{contrast}_echo-{echo:02d}_N{n:02d}'

    mean_path = output_dir / f'{stem}_mean.nii.gz'
    coverage_path = output_dir / f'{stem}_coverage.nii.gz'
    subjects_path = output_dir / f'{stem}_subjects.csv'

    if not OVERWRITE_OUTPUTS and (
        mean_path.exists() or (SAVE_COVERAGE_MAPS and coverage_path.exists())
        or subjects_path.exists()
    ):
        raise FileExistsError(f'Output exists for {stem}; not overwriting')

    mean = np.zeros(counts.shape, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=(counts > 0))

    ref_path = inputs[(contrast, echo, subjects[0])]
    save_nifti(mean, ref_path, mean_path, np.float32)
    del mean

    if SAVE_COVERAGE_MAPS:
        save_nifti(counts.copy(), ref_path, coverage_path, np.uint8)

    write_csv(
        subjects_path,
        ['subject', 'image_path'],
        [[subject, str(inputs[(contrast, echo, subject)])] for subject in subjects]
    )
    print(f'  Saved {n}-subject mean: {mean_path}')


def make_means(contrast, echo, inputs, shape):
    """Process the fixed five first, save N05, then add five more for N10."""
    remaining_five = [s for s in TEN_SUBJECTS if s not in FIVE_SUBJECTS]
    process_order = FIVE_SUBJECTS + remaining_five

    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint8)

    for i, subject in enumerate(process_order, start=1):
        path = inputs[(contrast, echo, subject)]
        print(f'    Loading {subject} ({i}/10): {path.name}', flush=True)

        # Read only one subject's voxel data into memory at a time.
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32).reshape(shape)
        valid = np.isfinite(data)
        if EXCLUDE_ZERO_VOXELS:
            valid &= (data != 0)

        # Update sum and coverage only for valid voxels.
        np.add(sums, data, out=sums, where=valid)
        np.add(counts, 1, out=counts, where=valid)
        del data, valid, img

        if i == 5:
            save_group(contrast, echo, FIVE_SUBJECTS, inputs, sums, counts)

    save_group(contrast, echo, TEN_SUBJECTS, inputs, sums, counts)


# ============================================================
# MAIN
# ============================================================

def main():
    validate_cohorts()
    started = time.monotonic()
    print('Fixed N05:', ', '.join(FIVE_SUBJECTS))
    print('Fixed N10:', ', '.join(TEN_SUBJECTS))
    print('Excluded from both groups: sub-007')

    inputs, shape = preflight()

    tasks = [
        (contrast, echo)
        for contrast, echoes in ECHOES_BY_CONTRAST.items()
        for echo in echoes
    ]
    print(f'\nAveraging {len(tasks)} contrast/echo combinations; '
          f'{len(tasks) * 2} means total.\n', flush=True)

    for task_number, (contrast, echo) in enumerate(tasks, start=1):
        print(f'[{task_number}/{len(tasks)}] {contrast} echo-{echo:02d}', flush=True)
        make_means(contrast, echo, inputs, shape)
        elapsed = time.monotonic() - started
        eta = elapsed / task_number * (len(tasks) - task_number)
        print(f'  Elapsed: {timedelta(seconds=int(elapsed))} | '
              f'Estimated remaining: {timedelta(seconds=int(eta))}\n',
              flush=True)

    print(f'Finished! Results saved in {OUTPUT_ROOT}')
    print(f'Total runtime: {timedelta(seconds=int(time.monotonic() - started))}')


if __name__ == '__main__':
    main()

