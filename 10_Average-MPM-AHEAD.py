
#!/usr/bin/env python3
"""Average original-qMRI-derived AHEAD quantitative maps in fixed N05/N10 cohorts.

Requires the images AND JSON provenance sidecars made by 09_MPM-to-AHEAD.py.
This prevents accidental mixing of images derived from original qMRI maps with
older AHEAD outputs created from brain-extracted Nighres maps.

Input:
  echoes_to_mni/sub-XXX/ses-02/anat/AHEAD/MPM/<MAP>/
    sub-XXX_ses-02_<MAP>_align-ahead_def-img.nii.gz
    sub-XXX_ses-02_<MAP>_align-ahead_def-img.json

Output:
  echoes_to_mni/group_averages/MPM/<MAP>/
    <MAP>_N05_mean.nii.gz       <MAP>_N10_mean.nii.gz
    <MAP>_N05_coverage.nii.gz   <MAP>_N10_coverage.nii.gz
    <MAP>_N05_subjects.csv      <MAP>_N10_subjects.csv

Methods: arithmetic mean of finite, nonzero voxels, as in your echo/slab
averaging scripts. A zero means no valid voxels contributed; see coverage map.
No registration, interpolation, smoothing, or quantitative refitting occurs.

Uses disk-backed accumulators to limit RAM use for large AHEAD images.
Releases slice views before cleanup to prevent temporary-file errors.

Usage:
  python3 -u 10_Average-MPM-AHEAD.py --dry-run
  python3 -u 10_Average-MPM-AHEAD.py
  python3 -u 10_Average-MPM-AHEAD.py --overwrite

Run AFTER all 40 maps have been registered by 09_MPM-to-AHEAD.py.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time

from datetime import datetime, timedelta
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# DIRECTORIES AND SETTINGS
# ============================================================

ROOT = Path(
    '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/'
    'derivatives/nighres/echoes_to_mni'
)

OUT = ROOT / 'group_averages' / 'MPM'

SESSION = 'ses-02'

MAPS = (
    'R1map',
    'R2starmap',
    'MTsat',
    'PDmap'
)

N05 = (
    'sub-004',
    'sub-011',
    'sub-012',
    'sub-015',
    'sub-017'
)

N10 = (
    'sub-004',
    'sub-009',
    'sub-011',
    'sub-012',
    'sub-013',
    'sub-014',
    'sub-015',
    'sub-016',
    'sub-017',
    'sub-019'
)

ORDER = N05 + tuple(
    subject for subject in N10
    if subject not in N05
)

REQUIRED_SOURCE = 'original-qmri-canonical'

GRID_TOL_MM = 1e-3


# ============================================================
# INPUT AND OUTPUT PATHS
# ============================================================

def input_paths(subject, name):
    folder = (
        ROOT
        / subject
        / SESSION
        / 'anat'
        / 'AHEAD'
        / 'MPM'
        / name
    )

    stem = (
        f'{subject}_{SESSION}_{name}'
        f'_align-ahead_def-img'
    )

    return (
        folder / f'{stem}.nii.gz',
        folder / f'{stem}.json'
    )


def output_paths(name, n):
    folder = OUT / name

    stem = f'{name}_N{n:02d}'

    return (
        folder / f'{stem}_mean.nii.gz',
        folder / f'{stem}_coverage.nii.gz',
        folder / f'{stem}_subjects.csv'
    )


# ============================================================
# IMAGE AND GRID VALIDATION
# ============================================================

def scalar_image(path):
    image = nib.load(str(path))

    if (
        len(image.shape) < 3
        or any(
            size != 1
            for size in image.shape[3:]
        )
    ):
        raise ValueError(
            f'Expected 3D scalar image: '
            f'{path}, shape={image.shape}'
        )

    if not np.isfinite(image.affine).all():
        raise ValueError(
            f'Nonfinite image affine: {path}'
        )

    return image


def same_grid(image, reference):
    return (
        image.shape[:3] == reference.shape[:3]
        and np.allclose(
            image.affine,
            reference.affine,
            atol=GRID_TOL_MM,
            rtol=0
        )
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight(overwrite):
    """Validate all 40 inputs before averaging anything."""

    if (
        len(N05) != 5
        or len(set(N05)) != 5
        or len(N10) != 10
        or len(set(N10)) != 10
        or not set(N05).issubset(N10)
    ):
        raise ValueError(
            'Fixed N05/N10 cohort definitions are invalid'
        )

    ahead_ref_path = (
        ROOT
        / 'group_averages'
        / 'PDw'
        / 'PDw_echo-01_N10_mean.nii.gz'
    )

    reference = scalar_image(ahead_ref_path)

    print(
        f'AHEAD reference: {ahead_ref_path}, '
        f'shape={reference.shape[:3]}',
        flush=True
    )

    inputs = {}
    problems = []

    for name in MAPS:

        for subject in N10:

            path, sidecar = input_paths(
                subject, name
            )

            try:
                image = scalar_image(path)

                if not same_grid(image, reference):
                    raise ValueError(
                        'Grid does not match AHEAD reference: '
                        f'{image.shape[:3]} vs '
                        f'{reference.shape[:3]}'
                    )

                with sidecar.open(
                    'r',
                    encoding='utf-8'
                ) as stream:
                    provenance = json.load(stream)

                for key, expected in (
                    ('source_type', REQUIRED_SOURCE),
                    ('subject', subject),
                    ('session', SESSION),
                    ('parameter_map', name),
                    ('canonical_reorientation', True),
                    ('ahead_resampling_interpolation', 'linear'),
                ):
                    if provenance.get(key) != expected:
                        raise ValueError(
                            f'Bad provenance {key}: '
                            f'{provenance.get(key)!r}, '
                            f'expected {expected!r}'
                        )

                # Require original qMRI provenance.
                if (
                    '/derivatives/qMRI/'
                    not in provenance.get('source_file', '')
                ):
                    raise ValueError(
                        'Provenance does not point '
                        'to original qMRI map'
                    )

                inputs[(name, subject)] = path

            except (
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError
            ) as exc:

                problems.append(
                    (
                        subject,
                        name,
                        'MISSING/INVALID INPUT',
                        str(exc)
                    )
                )

        # Check existing group averages.
        for n in (5, 10):

            for path in output_paths(name, n):

                if path.exists() and not overwrite:

                    problems.append(
                        (
                            '-',
                            name,
                            'OUTPUT EXISTS',
                            f'{path}; use --overwrite '
                            'to archive and replace'
                        )
                    )

    return inputs, reference, problems


# ============================================================
# SAVE HELPERS
# ============================================================

def write_csv(path, header, rows):
    with path.open(
        'x',
        newline='',
        encoding='utf-8'
    ) as stream:

        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_image(data, reference, dest, dtype):
    """Save an image on the common AHEAD grid."""

    header = reference.header.copy()

    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)

    image = nib.Nifti1Image(
        data,
        reference.affine,
        header=header
    )

    nib.save(image, str(dest))

    saved = scalar_image(dest)

    if not same_grid(saved, reference):
        raise RuntimeError(
            f'Written image grid mismatch: {dest}'
        )


# ============================================================
# SAFE OUTPUT COMMIT AND BACKUP
# ============================================================

def commit_three(
    staged,
    destinations,
    overwrite,
    run_id
):
    """
    Commit mean, coverage, and subject CSV.

    If overwrite is enabled, archive older outputs.
    Restore backups if the commit fails.
    """

    backed_up = []
    committed = []

    folder = destinations[0].parent

    try:

        if (
            not overwrite
            and any(
                path.exists()
                for path in destinations
            )
        ):
            raise FileExistsError(
                f'Output exists in {folder}; '
                'use --overwrite'
            )

        if (
            overwrite
            and any(
                path.exists()
                for path in destinations
            )
        ):

            backup_dir = (
                folder
                / 'backup'
                / run_id
            )

            backup_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for old in destinations:

                if old.exists():

                    backup = backup_dir / old.name

                    if backup.exists():
                        raise FileExistsError(
                            f'Backup already exists: {backup}'
                        )

                    os.replace(old, backup)

                    backed_up.append(
                        (old, backup)
                    )

        for temp, destination in zip(
            staged,
            destinations
        ):

            os.replace(temp, destination)

            committed.append(destination)

    except Exception:

        # Remove newly committed outputs.
        for new in committed:
            new.unlink(missing_ok=True)

        # Restore previous outputs.
        for old, backup in reversed(backed_up):

            if (
                backup.exists()
                and not old.exists()
            ):
                os.replace(backup, old)

        raise


# ============================================================
# SAVE N05 OR N10 AVERAGES
# ============================================================

def save_cohort(
    name,
    cohort,
    sums,
    counts,
    ref,
    inputs,
    tempdir,
    chunk_slices,
    overwrite,
    run_id
):
    """Calculate and save one cohort's group mean."""

    n = len(cohort)

    mean_path, coverage_path, csv_path = (
        output_paths(name, n)
    )

    mean_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shape = ref.shape[:3]

    mean_work = (
        Path(tempdir)
        / f'{name}_N{n:02d}_working_mean.bin'
    )

    means = np.memmap(
        mean_work,
        dtype=np.float32,
        mode='w+',
        shape=shape
    )

    stage_dir = (
        Path(tempdir)
        / f'staged_N{n:02d}'
    )

    stage_dir.mkdir(exist_ok=True)

    staged = [
        stage_dir / path.name
        for path in (
            mean_path,
            coverage_path,
            csv_path
        )
    ]

    try:

        # Calculate mean in chunks.
        for z in range(
            0,
            shape[2],
            chunk_slices
        ):

            end = min(
                shape[2],
                z + chunk_slices
            )

            sl = (
                slice(None),
                slice(None),
                slice(z, end)
            )

            current_sum = np.asarray(
                sums[sl]
            )

            current_count = np.asarray(
                counts[sl]
            )

            block = np.zeros(
                current_sum.shape,
                dtype=np.float32
            )

            np.divide(
                current_sum,
                current_count,
                out=block,
                where=(current_count > 0),
                casting='unsafe'
            )

            means[sl] = block

            # Release slice views before cleanup.
            del current_sum, current_count, block

        means.flush()

        # Save staged outputs.
        write_image(
            means,
            ref,
            staged[0],
            np.float32
        )

        write_image(
            counts,
            ref,
            staged[1],
            np.uint8
        )

        write_csv(
            staged[2],
            ['subject', 'image_path'],
            [
                [
                    subject,
                    str(inputs[(name, subject)])
                ]
                for subject in cohort
            ]
        )

        # Commit completed outputs.
        commit_three(
            staged,
            (
                mean_path,
                coverage_path,
                csv_path
            ),
            overwrite,
            run_id
        )

    finally:

        # Close memory-mapped mean.
        del means

        mean_work.unlink(missing_ok=True)

    print(
        f'  SAVED {name} N{n:02d}: {mean_path}',
        flush=True
    )


# ============================================================
# AVERAGE ONE QUANTITATIVE PARAMETER
# ============================================================

def average_map(
    name,
    inputs,
    ref,
    chunk_slices,
    overwrite,
    run_id
):
    """
    Process N05 first, save it, then add five subjects
    and save N10.

    Uses disk-backed arrays to reduce RAM usage.
    """

    folder = OUT / name

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    shape = ref.shape[:3]

    with tempfile.TemporaryDirectory(
        prefix='.working_mpm_',
        dir=folder
    ) as work:

        sums = np.memmap(
            Path(work) / 'sums.bin',
            dtype=np.float64,
            mode='w+',
            shape=shape
        )

        counts = np.memmap(
            Path(work) / 'counts.bin',
            dtype=np.uint8,
            mode='w+',
            shape=shape
        )

        sums[:] = 0
        counts[:] = 0

        for index, subject in enumerate(
            ORDER,
            start=1
        ):

            path = inputs[(name, subject)]

            print(
                f'  [{index}/10] Reading {subject}: '
                f'{path.name}',
                flush=True
            )

            img = scalar_image(path)

            # Load one subject at a time.
            voxels = np.asarray(
                img.dataobj,
                dtype=np.float32
            ).reshape(shape)

            # Process image in small chunks.
            for z in range(
                0,
                shape[2],
                chunk_slices
            ):

                stop = min(
                    z + chunk_slices,
                    shape[2]
                )

                sl = (
                    slice(None),
                    slice(None),
                    slice(z, stop)
                )

                block = voxels[sl]

                valid = (
                    np.isfinite(block)
                    & (block != 0)
                )

                total = sums[sl]
                coverage = counts[sl]

                # Update sums.
                np.add(
                    total,
                    block,
                    out=total,
                    where=valid
                )

                # Update contributing subject counts.
                np.add(
                    coverage,
                    1,
                    out=coverage,
                    where=valid
                )

                # IMPORTANT FIX:
                # Release views into the memory-mapped
                # arrays before temporary-file cleanup.
                #
                # Otherwise, NFS may leave .nfs files,
                # causing "Directory not empty" errors.

                del total, coverage, block, valid

            del voxels, img

            # Save the five-subject mean.
            if index == 5:

                save_cohort(
                    name,
                    N05,
                    sums,
                    counts,
                    ref,
                    inputs,
                    work,
                    chunk_slices,
                    overwrite,
                    run_id
                )

        # Save the ten-subject mean.
        save_cohort(
            name,
            N10,
            sums,
            counts,
            ref,
            inputs,
            work,
            chunk_slices,
            overwrite,
            run_id
        )

        sums.flush()
        counts.flush()

        # Release disk-backed arrays before
        # TemporaryDirectory removes its files.
        del sums, counts


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate only; write nothing'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Archive existing means, coverage maps, '
             'and CSV files before replacement'
    )

    parser.add_argument(
        '--chunk-slices',
        type=int,
        default=12,
        help='Number of slices per chunk; default 12'
    )

    args = parser.parse_args()

    if args.chunk_slices < 1:
        parser.error(
            '--chunk-slices must be >= 1'
        )

    print(
        'N05:', ', '.join(N05),
        flush=True
    )

    print(
        'N10:', ', '.join(N10),
        flush=True
    )

    print(
        'Require all inputs to originate '
        'from original qMRI maps.',
        flush=True
    )

    print(
        'Arithmetic: finite nonzero voxels; '
        'coverage counts saved.',
        flush=True
    )

    # Validate all inputs first.
    inputs, reference, problems = preflight(
        args.overwrite
    )

    if problems:

        print(
            f'PREFLIGHT FAILED: {len(problems)} '
            'problem(s). No means written.',
            file=sys.stderr,
            flush=True
        )

        for (
            subject,
            name,
            category,
            details
        ) in problems:

            print(
                f'  {subject} {name}: '
                f'{category}: {details}',
                file=sys.stderr,
                flush=True
            )

        return 1

    print(
        f'PREFLIGHT PASSED: {len(inputs)} '
        'individual quantitative maps.',
        flush=True
    )

    if args.dry_run:

        for name in MAPS:

            print(
                f'  {name}: N05 and N10 means '
                '+ coverage + CSV',
                flush=True
            )

        print(
            'DRY RUN: no files changed.',
            flush=True
        )

        return 0

    # Unique ID for backups.
    run_id = (
        datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )
        + f'_{os.getpid()}'
    )

    started = time.monotonic()

    for i, name in enumerate(
        MAPS,
        start=1
    ):

        print(
            f'[{i}/{len(MAPS)}] Averaging {name}',
            flush=True
        )

        average_map(
            name,
            inputs,
            reference,
            args.chunk_slices,
            args.overwrite,
            run_id
        )

        elapsed = (
            time.monotonic()
            - started
        )

        eta = (
            elapsed / i
            * (len(MAPS) - i)
        )

        print(
            f'Elapsed '
            f'{timedelta(seconds=int(elapsed))}; '
            f'ETA {timedelta(seconds=int(eta))}',
            flush=True
        )

    print(
        'COMPLETE: 8 means, 8 coverage maps, '
        f'8 subject CSV files under {OUT}',
        flush=True
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    try:
        sys.exit(main())

    except (
        OSError,
        ValueError,
        RuntimeError
    ) as error:

        print(
            f'ERROR: {error}',
            file=sys.stderr
        )

        sys.exit(1)
