#!/usr/bin/env python3
"""Progressive AHEAD-space mean MRI images, N01 through N10.

Default: MTw echo-01 only, for investigating artifacts in group means.
The processing order deliberately begins with the previously chosen five subjects,
then adds the other five so N05 and N10 match the fixed-cohort experiment.

No registration is performed: input NIfTI images must already be AHEAD-aligned.
Nonfinite or zero-valued voxels do not contribute to the mean (same as the
previous fixed-subject averaging script). Coverage maps record contributors.
"""

import csv
import time
from datetime import timedelta
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(
    '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/'
    'derivatives/nighres/echoes_to_mni'
)
OUTPUT_ROOT = ROOT / 'group_averages' / 'artifact test'
SESSION = 'ses-02'

# Order matters: each successive mean adds the next subject.
# N05 = 004, 011, 012, 015, 017 (same fixed 5 as before)
# N10 = 004, 009, 011, 012, 013, 014, 015, 016, 017, 019
SUBJECT_ORDER = [
    'sub-004',
    'sub-011',
    'sub-012',
    'sub-015',
    'sub-017',
    'sub-009',
    'sub-013',
    'sub-014',
    'sub-016',
    'sub-019',
]

# For artifact inspection, start with MTw echo 01 only.
ECHOES_BY_CONTRAST = {'MTw': [1]}

# To instead process every MPM echo, replace the line above with:
# ECHOES_BY_CONTRAST = {
#     'MTw': range(1, 5),
#     'PDw': range(1, 7),
#     'T1w': range(1, 7),
# }

EXCLUDE_ZERO_VOXELS = True
SAVE_COVERAGE_MAPS = True
OVERWRITE_OUTPUTS = False  # Change to True if intentionally rerunning.


# ============================================================
# HELPERS
# ============================================================

def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def find_image(subject, contrast, echo):
    anat_dir = ROOT / subject / SESSION / 'anat'
    pattern = (
        f'{subject}_{SESSION}_*acq-{contrast}_*echo-{echo:02d}_*'
        'part-mag*reorient_align-ahead_def-img.nii*'
    )
    matches = sorted(glob(str(anat_dir / pattern)))
    if len(matches) != 1:
        issue = 'MISSING' if not matches else 'AMBIGUOUS'
        return None, (issue, f'{len(matches)} matches: {anat_dir / pattern}')
    return Path(matches[0]), None


def image_shape(image):
    shape = image.shape
    if len(shape) < 3 or any(dim != 1 for dim in shape[3:]):
        raise ValueError(f'Expected 3D or singleton-4D image, got {shape}')
    return tuple(shape[:3])


def preflight(tasks):
    """Check the exact ten images per task and their spatial geometry."""
    inputs = {}
    issues = []
    reference_by_task = {}

    for contrast, echo in tasks:
        ref_shape = None
        ref_affine = None
        ref_path = None

        for subject in SUBJECT_ORDER:
            path, error = find_image(subject, contrast, echo)
            if error is not None:
                issues.append((subject, contrast, echo, *error))
                continue

            try:
                img = nib.load(str(path))  # Header only, not full image data.
                shape = image_shape(img)
                affine = img.affine
                if not np.isfinite(affine).all():
                    raise ValueError('NIfTI affine has nonfinite entries')

                if ref_shape is None:
                    ref_shape = shape
                    ref_affine = affine.copy()
                    ref_path = path
                elif shape != ref_shape or not np.allclose(
                    affine, ref_affine, atol=1e-5, rtol=0
                ):
                    raise ValueError(
                        f'Incompatible grid: {shape}; reference {ref_shape} at {ref_path}'
                    )
                inputs[(contrast, echo, subject)] = path
            except Exception as exc:
                issues.append((subject, contrast, echo, 'INVALID / INCOMPATIBLE',
                               f'{path}: {exc}'))

        if ref_path is not None:
            reference_by_task[(contrast, echo)] = (ref_path, ref_shape)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    issues_path = OUTPUT_ROOT / 'preflight_issues.csv'
    write_csv(
        issues_path,
        ['subject', 'contrast', 'echo', 'issue', 'details'],
        [[s, c, f'{e:02d}', issue, details]
         for s, c, e, issue, details in issues]
    )

    if issues:
        print(f'\nPreflight FAILED: {len(issues)} problem(s):', flush=True)
        for s, c, e, issue, details in issues:
            print(f'  {s} {c} echo-{e:02d}: {issue} — {details}')
        raise SystemExit(f'No averages created. See {issues_path}')

    print(f'Preflight PASSED: {len(inputs)} files found and grids match.')
    print('Note: Matching grids does NOT guarantee anatomically good registration.')
    return inputs, reference_by_task


def save_nifti(array, reference_path, destination, dtype):
    ref = nib.load(str(reference_path))
    header = ref.header.copy()
    header.set_data_shape(array.shape)
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)
    nib.save(nib.Nifti1Image(array, ref.affine, header=header), str(destination))


def check_existing_outputs(tasks):
    """Prevent accidentally overwriting any earlier artifact-test results."""
    if OVERWRITE_OUTPUTS:
        return
    existing = []
    for contrast, echo in tasks:
        directory = OUTPUT_ROOT / f'{contrast}_echo-{echo:02d}'
        for n in range(1, len(SUBJECT_ORDER) + 1):
            stem = f'{contrast}_echo-{echo:02d}_N{n:02d}'
            for suffix in ('_mean.nii.gz', '_coverage.nii.gz', '_subjects.csv'):
                candidate = directory / f'{stem}{suffix}'
                if candidate.exists():
                    existing.append(candidate)
    if existing:
        print('Refusing to overwrite previous artifact-test results:')
        for path in existing[:12]:
            print(f'  {path}')
        if len(existing) > 12:
            print(f'  ... and {len(existing) - 12} more')
        raise SystemExit('Set OVERWRITE_OUTPUTS = True to replace these outputs.')


def average_one_task(contrast, echo, inputs, reference):
    """Accumulate one subject at a time and save N01, N02, ..., N10."""
    ref_path, shape = reference
    outdir = OUTPUT_ROOT / f'{contrast}_echo-{echo:02d}'
    outdir.mkdir(parents=True, exist_ok=True)

    # Full-size arrays are allocated only once per contrast/echo task.
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint8)
    start = time.monotonic()

    for n, subject in enumerate(SUBJECT_ORDER, start=1):
        path = inputs[(contrast, echo, subject)]
        print(f'  Adding {subject} ({n}/10): {path.name}', flush=True)

        volume = nib.load(str(path))
        data = np.asarray(volume.dataobj, dtype=np.float32).reshape(shape)
        valid = np.isfinite(data)
        if EXCLUDE_ZERO_VOXELS:
            valid &= data != 0

        # Ignore missing/zero-padded voxels rather than treating them as zero signal.
        np.add(sums, data, out=sums, where=valid)
        np.add(counts, 1, out=counts, where=valid)
        del data, valid, volume

        stem = f'{contrast}_echo-{echo:02d}_N{n:02d}'
        mean_path = outdir / f'{stem}_mean.nii.gz'
        coverage_path = outdir / f'{stem}_coverage.nii.gz'
        subjects_path = outdir / f'{stem}_subjects.csv'

        mean = np.zeros(shape, dtype=np.float32)
        np.divide(sums, counts, out=mean, where=(counts > 0))
        save_nifti(mean, ref_path, mean_path, np.float32)
        del mean

        if SAVE_COVERAGE_MAPS:
            save_nifti(counts, ref_path, coverage_path, np.uint8)

        write_csv(
            subjects_path,
            ['subject', 'image_path'],
            [[s, str(inputs[(contrast, echo, s)])]
             for s in SUBJECT_ORDER[:n]]
        )

        elapsed = time.monotonic() - start
        eta = elapsed / n * (len(SUBJECT_ORDER) - n)
        print(
            f'  Saved {stem} | elapsed {timedelta(seconds=int(elapsed))} | '
            f'ETA {timedelta(seconds=int(eta))}',
            flush=True
        )


def main():
    if len(SUBJECT_ORDER) != 10 or len(set(SUBJECT_ORDER)) != 10:
        raise SystemExit('SUBJECT_ORDER must contain exactly 10 unique subjects.')

    tasks = [
        (contrast, echo)
        for contrast, echo_numbers in ECHOES_BY_CONTRAST.items()
        for echo in echo_numbers
    ]
    if not tasks:
        raise SystemExit('No contrasts/echoes configured.')

    print('Progressive subject order (one additional subject at each N):')
    for n, subject in enumerate(SUBJECT_ORDER, start=1):
        print(f'  N{n:02d}: +{subject}')
    print(f'\nOutput directory: {OUTPUT_ROOT}\n', flush=True)

    inputs, references = preflight(tasks)
    check_existing_outputs(tasks)

    started = time.monotonic()
    for index, (contrast, echo) in enumerate(tasks, start=1):
        print(f'\n[{index}/{len(tasks)}] {contrast} echo-{echo:02d}', flush=True)
        average_one_task(contrast, echo, inputs, references[(contrast, echo)])

    print(f'\nAll {len(tasks) * len(SUBJECT_ORDER)} progressive means saved.')
    print(f'Results: {OUTPUT_ROOT}')
    print(f'Total runtime: {timedelta(seconds=int(time.monotonic() - started))}')


if __name__ == '__main__':
    main()

