#!/usr/bin/env python3
"""Average AHEAD-registered LC slab MT-on and MT-off images for fixed N05/N10 cohorts.

Reads:
  echoes_to_mni/sub-XXX/ses-02/anat/AHEAD/LCslab/
      sub-XXX_ses-02_mt-{on,off}_chunk-LCslab_MTR_in-PDw_header_align-ahead_def-img.nii[.gz]

Writes:
  echoes_to_mni/group_averages/LCslab/
      LCslab_mt-{on,off}_N{05,10}_{mean,coverage}.nii.gz
      LCslab_mt-{on,off}_N{05,10}_subjects.csv
      LCslab_preflight_issues.csv

No registration or intensity normalization is performed. Zero-padding and nonfinite
voxels are excluded from voxelwise means; uncovered output voxels are set to zero.
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
OUTPUT_DIR = ROOT / 'group_averages' / 'LCslab'
SESSION = 'ses-02'

FIVE_SUBJECTS = [
    'sub-004', 'sub-011', 'sub-012', 'sub-015', 'sub-017'
]
TEN_SUBJECTS = [
    'sub-004', 'sub-009', 'sub-011', 'sub-012', 'sub-013',
    'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-019'
]

MT_CONDITIONS = ('on', 'off')
EXCLUDE_ZERO_VOXELS = True
SAVE_COVERAGE_MAPS = True
OVERWRITE_OUTPUTS = True
AFFINE_TOLERANCE_MM = 1e-5


# ============================================================
# HELPERS
# ============================================================

def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def validate_cohorts():
    if len(FIVE_SUBJECTS) != 5 or len(set(FIVE_SUBJECTS)) != 5:
        raise ValueError('FIVE_SUBJECTS must have exactly 5 unique subjects')
    if len(TEN_SUBJECTS) != 10 or len(set(TEN_SUBJECTS)) != 10:
        raise ValueError('TEN_SUBJECTS must have exactly 10 unique subjects')
    if not set(FIVE_SUBJECTS).issubset(TEN_SUBJECTS):
        raise ValueError('FIVE_SUBJECTS must be a subset of TEN_SUBJECTS')


def find_lc_slab(subject, condition):
    """Locate exactly one AHEAD-aligned LC slab for subject and MT condition."""
    directory = ROOT / subject / SESSION / 'anat' / 'AHEAD' / 'LCslab'
    pattern = (
        f'{subject}_{SESSION}_mt-{condition}_chunk-LCslab_MTR_'
        'in-PDw_header_align-ahead_def-img.nii*'
    )
    matches = sorted(p for p in glob(str(directory / pattern))
                     if Path(p).is_file())
    if len(matches) != 1:
        issue = 'MISSING' if not matches else 'AMBIGUOUS'
        return None, (issue, f'{len(matches)} matches: {directory / pattern}; {matches}')
    return Path(matches[0]), None


def spatial_shape(image):
    """Accept 3D volumes and 4D volumes with only singleton trailing axes."""
    shape = image.shape
    if len(shape) < 3 or any(n != 1 for n in shape[3:]):
        raise ValueError(f'Expected 3D (or singleton 4D) image, got {shape}')
    return tuple(shape[:3])


def check_grid(path, expected_shape, expected_affine):
    img = nib.load(str(path))
    shape = spatial_shape(img)
    if not np.all(np.isfinite(img.affine)):
        raise ValueError('Affine contains nonfinite values')
    if expected_shape is not None and (
        shape != expected_shape or not np.allclose(
            img.affine, expected_affine,
            rtol=0, atol=AFFINE_TOLERANCE_MM
        )
    ):
        raise ValueError(f'Spatial grid mismatch: shape={shape}, '
                         f'reference shape={expected_shape}')
    return shape, img.affine.copy()


def preflight():
    """Require every slab from the fixed N10 group on one common AHEAD grid."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = []
    inputs = {}
    reference_shape = None
    reference_affine = None
    reference_path = None

    # When available, also confirm the slabs share the existing MPM AHEAD grid.
    mpm_reference = ROOT / 'group_averages' / 'MTw' / 'MTw_echo-01_N10_mean.nii.gz'
    if mpm_reference.exists():
        reference_shape, reference_affine = check_grid(mpm_reference, None, None)
        reference_path = mpm_reference
        print(f'Using existing MTw average as spatial reference: {mpm_reference}')
    else:
        print('Note: MTw group mean unavailable; using first valid slab as grid reference.')

    for condition in MT_CONDITIONS:
        for subject in TEN_SUBJECTS:
            path, issue = find_lc_slab(subject, condition)
            if issue:
                problems.append([subject, f'mt-{condition}', *issue])
                continue
            try:
                if reference_shape is None:
                    reference_shape, reference_affine = check_grid(path, None, None)
                    reference_path = path
                else:
                    check_grid(path, reference_shape, reference_affine)
                inputs[(condition, subject)] = path
            except Exception as exc:
                problems.append([subject, f'mt-{condition}',
                                 'INVALID / INCOMPATIBLE', f'{path}: {exc}'])

    report = OUTPUT_DIR / 'LCslab_preflight_issues.csv'
    write_csv(report, ['subject', 'condition', 'problem', 'details'], problems)

    if problems:
        print(f'\nPreflight FAILED: {len(problems)} issue(s).')
        for subject, condition, problem, details in problems:
            print(f'  {subject} {condition}: {problem}: {details}')
        raise SystemExit(f'No averages written. Full issue report: {report}')

    print(f'Preflight passed: {len(inputs)} LC slabs on one AHEAD grid.')
    print(f'Shape: {reference_shape} | Grid reference: {reference_path}')
    print(f'Preflight report: {report} (header only)\n')
    return inputs, reference_shape


def save_nifti(data, reference_path, target, dtype):
    reference = nib.load(str(reference_path))
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)
    image = nib.Nifti1Image(data, reference.affine, header=header)
    nib.save(image, str(target))


def save_group(condition, subjects, inputs, sums, counts):
    n = len(subjects)
    prefix = f'LCslab_mt-{condition}_N{n:02d}'
    mean_path = OUTPUT_DIR / f'{prefix}_mean.nii.gz'
    coverage_path = OUTPUT_DIR / f'{prefix}_coverage.nii.gz'
    subjects_path = OUTPUT_DIR / f'{prefix}_subjects.csv'

    expected_outputs = [mean_path, subjects_path]
    if SAVE_COVERAGE_MAPS:
        expected_outputs.append(coverage_path)
    if not OVERWRITE_OUTPUTS:
        existing = [str(p) for p in expected_outputs if p.exists()]
        if existing:
            raise FileExistsError(f'Output(s) already exist: {existing}')

    mean = np.zeros(counts.shape, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=(counts > 0))
    reference_path = inputs[(condition, subjects[0])]

    save_nifti(mean, reference_path, mean_path, np.float32)
    del mean
    if SAVE_COVERAGE_MAPS:
        save_nifti(counts, reference_path, coverage_path, np.uint8)

    write_csv(
        subjects_path,
        ['subject', 'image_path'],
        [[subject, str(inputs[(condition, subject)])] for subject in subjects]
    )

    covered = np.count_nonzero(counts)
    full_coverage = np.count_nonzero(counts == n)
    print(f'  Saved {prefix}: {mean_path}')
    print(f'    Nonzero-coverage voxels: {covered:,}; '
          f'full {n}/{n} coverage: {full_coverage:,}', flush=True)


def average_condition(condition, inputs, shape):
    """Add fixed N05 subjects first, then five more to create fixed N10."""
    extra_five = [s for s in TEN_SUBJECTS if s not in FIVE_SUBJECTS]
    process_order = FIVE_SUBJECTS + extra_five

    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint8)

    for i, subject in enumerate(process_order, 1):
        path = inputs[(condition, subject)]
        print(f'  Loading mt-{condition}: {subject} ({i}/10)', flush=True)
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32).reshape(shape)
        valid = np.isfinite(data)
        if EXCLUDE_ZERO_VOXELS:
            valid &= (data != 0)

        np.add(sums, data, out=sums, where=valid)
        np.add(counts, 1, out=counts, where=valid)
        del data, valid, img

        if i == 5:
            save_group(condition, FIVE_SUBJECTS, inputs, sums, counts)

    save_group(condition, TEN_SUBJECTS, inputs, sums, counts)


def main():
    validate_cohorts()
    t0 = time.monotonic()
    print('N05:', ', '.join(FIVE_SUBJECTS))
    print('N10:', ', '.join(TEN_SUBJECTS))
    print('Excluded sub-007 and sub-020 from both cohorts.')
    inputs, shape = preflight()

    for i, condition in enumerate(MT_CONDITIONS, 1):
        print(f'[{i}/{len(MT_CONDITIONS)}] Averaging LC slab MT-{condition}', flush=True)
        average_condition(condition, inputs, shape)
        elapsed = time.monotonic() - t0
        eta = elapsed / i * (len(MT_CONDITIONS) - i)
        print(f'  Elapsed: {timedelta(seconds=int(elapsed))} | '
              f'Estimated remaining: {timedelta(seconds=int(eta))}\n')

    print(f'Finished. Files saved in {OUTPUT_DIR}')
    print(f'Total runtime: {timedelta(seconds=int(time.monotonic() - t0))}')


if __name__ == '__main__':
    main()

