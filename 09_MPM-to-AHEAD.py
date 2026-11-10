
#!/usr/bin/env python3
"""Register original quantitative MPM maps to AHEAD with EXISTING Nighres mappings.

Source (original, not brain-masked):
  derivatives/qMRI/sub-XXX/ses-02/anat/sub-XXX_ses-02_{R1map,R2starmap,MTsat,PDmap}.nii

Each original qMRI map is reoriented to closest-canonical RAS+ voxel ordering
(no interpolation), and its shape/affine MUST match the native Nighres R1 brain
reference. Then the two previously estimated coordinate mappings are applied:
  native MPM -> MNI -> AHEAD (linear interpolation; zero padding).

Output (one map per folder):
  echoes_to_mni/sub-XXX/ses-02/anat/AHEAD/MPM/<MAP>/
    sub-XXX_ses-02_<MAP>_align-ahead_def-img.nii.gz
    sub-XXX_ses-02_<MAP>_align-ahead_def-img.json   (provenance)

No source files, existing mappings, or other processing results are modified.
Existing outputs are NEVER silently reused. Use --overwrite to archive older
outputs before replacement, particularly sub-004 previously made from
--source nighres-brain. Complete preflight happens BEFORE processing anything.

Usage (inside Nighres Apptainer):
  python3 -u 09_MPM-to-AHEAD.py --subjects sub-004 --dry-run --overwrite
  python3 -u 09_MPM-to-AHEAD.py --subjects sub-004 --overwrite
  python3 -u 09_MPM-to-AHEAD.py --dry-run
  python3 -u 09_MPM-to-AHEAD.py

Run only after any older MPM registration job writing these outputs has exited.
Dependencies: numpy, nibabel, nighres (nighres not imported during dry-run).
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# DIRECTORIES AND SETTINGS
# ============================================================

BASE = Path(
    '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives'
)

QMRI = BASE / 'qMRI'
NIGHRES = BASE / 'nighres'
ROOT = NIGHRES / 'echoes_to_mni'

SESSION = 'ses-02'

MAPS = (
    'R1map',
    'R2starmap',
    'MTsat',
    'PDmap'
)

SUBJECTS_N10 = (
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

TOL_MM = 1e-3
SOURCE_TYPE = 'original-qmri-canonical'


# ============================================================
# SUBJECT AND FILE HELPERS
# ============================================================

def normalize_subject(item):
    number = str(item).removeprefix('sub-')

    if (
        not number.isdigit()
        or len(number) > 3
        or int(number) <= 0
    ):
        raise ValueError(f'Invalid subject: {item}')

    return f'sub-{int(number):03d}'


def required_nifti(stem):
    matches = [
        Path(str(stem) + extension)
        for extension in ('.nii', '.nii.gz')
        if Path(str(stem) + extension).is_file()
    ]

    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one NIfTI (.nii or .nii.gz) for '
            f'{stem}; found {len(matches)}'
        )

    return matches[0]


def read_scalar(path):
    image = nib.load(str(path))

    if (
        len(image.shape) < 3
        or any(size != 1 for size in image.shape[3:])
    ):
        raise ValueError(
            f'Expected 3D scalar image: {path} | {image.shape}'
        )

    if not np.all(np.isfinite(image.affine)):
        raise ValueError(f'Invalid affine: {path}')

    return image


# ============================================================
# VOXEL GRID VALIDATION
# ============================================================

def same_grid(shape, affine, reference):
    return (
        tuple(shape[:3]) == tuple(reference.shape[:3])
        and np.allclose(
            affine,
            reference.affine,
            atol=TOL_MM,
            rtol=0
        )
    )


def check_grid(shape, affine, reference, name):
    if not same_grid(shape, affine, reference):
        raise ValueError(
            f'{name}: incompatible voxel grid after canonical reorientation\n'
            f'  shape={tuple(shape[:3])}; affine=\n{affine}\n'
            f'  expected shape={reference.shape[:3]}; '
            f'affine=\n{reference.affine}\n'
            'Refusing to apply mappings to a nonmatching voxel grid.'
        )


# ============================================================
# CANONICAL ORIENTATION CALCULATION
# ============================================================

def canonical_geometry_only(image):
    """
    Calculate canonical header shape and affine without
    loading the voxel data.

    IMPORTANT:
    The output shape uses the INVERSE axis permutation.
    This corrects the previous orientation preflight bug.
    """

    start = nib.orientations.io_orientation(image.affine)

    target = nib.orientations.axcodes2ornt(
        ('R', 'A', 'S')
    )

    transform = nib.orientations.ornt_transform(
        start,
        target
    )

    # CORRECTED:
    # The orientation transform describes each input axis
    # in terms of its destination axis.
    #
    # To compute the output dimensions, use the inverse
    # permutation, exactly as apply_orientation does.

    axis_order = np.argsort(
        transform[:, 0].astype(int)
    )

    shape = tuple(
        image.shape[int(axis)]
        for axis in axis_order
    )

    affine = (
        image.affine
        @ nib.orientations.inv_ornt_aff(
            transform,
            image.shape[:3]
        )
    )

    return shape, affine


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def output_paths(subject, kind):
    folder = (
        ROOT
        / subject
        / SESSION
        / 'anat'
        / 'AHEAD'
        / 'MPM'
        / kind
    )

    stem = (
        f'{subject}_{SESSION}_{kind}'
        f'_align-ahead_def-img'
    )

    return (
        folder / f'{stem}.nii.gz',
        folder / f'{stem}.json'
    )


def ahead_reference():
    path = (
        ROOT
        / 'group_averages'
        / 'PDw'
        / 'PDw_echo-01_N10_mean.nii.gz'
    )

    return path, read_scalar(path)


# ============================================================
# SUBJECT PREFLIGHT
# ============================================================

def preflight_one(subject, ahead_ref):
    native_dir = (
        NIGHRES / subject / SESSION / 'anat'
    )

    original_dir = (
        QMRI / subject / SESSION / 'anat'
    )

    prefix = f'{subject}_{SESSION}_'

    # Native R1 image used as the registration reference
    native_ref_path = required_nifti(
        native_dir / f'{prefix}R1map_brain'
    )

    native_ref = read_scalar(native_ref_path)

    # Existing registration mappings
    m1 = required_nifti(
        native_dir / f'{prefix}R1map_brain_ants-map'
    )

    m2 = required_nifti(
        native_dir / f'{prefix}R1map_brain_ants-def0_ants-map'
    )

    # Validate coordinate maps
    for mapping_path in (m1, m2):
        mapping = nib.load(str(mapping_path))

        if mapping.ndim != 4 or mapping.shape[3] != 3:
            raise ValueError(
                f'Expected three-component coordinate map: '
                f'{mapping_path}: {mapping.shape}'
            )

    # Final mapping must use the AHEAD destination grid
    m2_img = nib.load(str(m2))

    check_grid(
        m2_img.shape,
        m2_img.affine,
        ahead_ref,
        str(m2)
    )

    jobs = []

    for kind in MAPS:
        source = required_nifti(
            original_dir / f'{prefix}{kind}'
        )

        original = read_scalar(source)

        # Header-only canonical orientation preflight
        shape, affine = canonical_geometry_only(original)

        check_grid(
            shape,
            affine,
            native_ref,
            str(source)
        )

        output, metadata = output_paths(
            subject,
            kind
        )

        jobs.append(
            dict(
                subject=subject,
                kind=kind,
                source=source,
                native_reference=native_ref_path,
                mapping1=m1,
                mapping2=m2,
                output=output,
                metadata=metadata
            )
        )

    return jobs


# ============================================================
# TEMPORARY FILE AND METADATA HELPERS
# ============================================================

def temp_path(destination):
    """Retain recognizable extensions for temporary files."""

    tag = uuid.uuid4().hex

    if destination.name.endswith('.nii.gz'):
        return destination.with_name(
            f'.{destination.name[:-7]}.{tag}.tmp.nii.gz'
        )

    return destination.with_name(
        f'.{destination.stem}.{tag}.tmp.json'
    )


def write_json(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True
        )

        stream.write('\n')


# ============================================================
# SAFE OUTPUT SAVING AND BACKUP
# ============================================================

def save_pair(
    image,
    provenance,
    dest,
    sidecar,
    overwrite,
    run_id
):
    """
    Stage and validate both outputs before replacing anything.

    Existing outputs are archived under backup/<run_id>
    when --overwrite is explicitly requested.

    The averaging script requires provenance metadata,
    preventing accidental mixing of different sources.
    """

    dest.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp_nii = temp_path(dest)
    tmp_json = temp_path(sidecar)

    moved = []
    committed = []

    try:
        # Stage the NIfTI
        nib.save(image, str(tmp_nii))

        staged = read_scalar(tmp_nii)

        if (
            staged.shape[:3] != image.shape[:3]
            or not np.allclose(
                staged.affine,
                image.affine,
                atol=1e-5,
                rtol=0
            )
        ):
            raise RuntimeError(
                f'Unable to validate staged image: {tmp_nii}'
            )

        # Stage the provenance JSON
        write_json(tmp_json, provenance)

        with tmp_json.open(encoding='utf-8') as stream:
            json.load(stream)

        if (
            dest.exists() or sidecar.exists()
        ) and not overwrite:
            raise FileExistsError(
                f'Output already exists, use --overwrite: {dest}'
            )

        archive_dir = (
            dest.parent / 'backup' / run_id
        )

        if overwrite and (
            dest.exists() or sidecar.exists()
        ):
            archive_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for old in (dest, sidecar):
                if old.exists():
                    backup = archive_dir / old.name

                    if backup.exists():
                        raise FileExistsError(
                            f'Backup path already exists: {backup}'
                        )

                    os.replace(old, backup)

                    moved.append((old, backup))

        # Commit new output files
        for stage, target in (
            (tmp_nii, dest),
            (tmp_json, sidecar)
        ):
            os.replace(stage, target)
            committed.append(target)

    except Exception:
        # Restore previous outputs if committing fails
        for path in committed:
            path.unlink(missing_ok=True)

        for old, backup in reversed(moved):
            if backup.exists() and not old.exists():
                os.replace(backup, old)

        raise

    finally:
        tmp_nii.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)

    return [
        str(backup)
        for _, backup in moved
    ]


# ============================================================
# APPLY NIGHRES COORDINATE MAPPINGS
# ============================================================

def register_one(
    job,
    ref_ahead,
    overwrite,
    run_id,
    nighres
):
    # Load original, unmasked quantitative map
    original = read_scalar(job['source'])

    # Reorient WITHOUT interpolation
    canonical = nib.as_closest_canonical(original)

    # Confirm that actual reoriented data matches
    # the native grid used for registration
    ref_native = read_scalar(
        job['native_reference']
    )

    check_grid(
        canonical.shape,
        canonical.affine,
        ref_native,
        f'{job["source"]} (actual canonical image)'
    )

    # Reuse previously calculated transformations
    # native MPM -> MNI -> AHEAD
    result = (
        nighres.registration.apply_coordinate_mappings(
            canonical,
            mapping1=str(job['mapping1']),
            mapping2=str(job['mapping2']),
            interpolation='linear',
            padding='zero',
            save_data=False
        )['result']
    )

    # Verify resulting AHEAD grid
    check_grid(
        result.shape,
        result.affine,
        ref_ahead,
        str(job['output'])
    )

    # Save the source and processing parameters
    provenance = {
        'pipeline': '09_MPM-to-AHEAD.py',
        'source_type': SOURCE_TYPE,
        'subject': job['subject'],
        'session': SESSION,
        'parameter_map': job['kind'],
        'source_file': str(job['source']),
        'native_reference': str(job['native_reference']),
        'mapping_native_to_mni': str(job['mapping1']),
        'mapping_mni_to_ahead': str(job['mapping2']),
        'canonical_reorientation': True,
        'reorientation_interpolates': False,
        'ahead_resampling_interpolation': 'linear',
        'padding': 'zero',
        'source_size_bytes': job['source'].stat().st_size,
        'source_mtime_ns': job['source'].stat().st_mtime_ns,
        'generated_utc': datetime.now(
            timezone.utc
        ).isoformat()
    }

    archives = save_pair(
        result,
        provenance,
        job['output'],
        job['metadata'],
        overwrite,
        run_id
    )

    return archives


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='e.g. sub-004 sub-011; default fixed N10'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Read headers only; write nothing'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Archive existing AHEAD maps before replacing them'
    )

    args = parser.parse_args()

    # Use selected subjects, or the fixed N10 group
    subjects = (
        list(
            dict.fromkeys(
                normalize_subject(s)
                for s in args.subjects
            )
        )
        if args.subjects
        else list(SUBJECTS_N10)
    )

    print(
        'Source: ORIGINAL derivatives/qMRI maps (unmasked)',
        flush=True
    )

    print(
        'Subjects:', ', '.join(subjects),
        flush=True
    )

    print(
        'Pipeline: canonical reorientation -> '
        'native-to-MNI -> MNI-to-AHEAD',
        flush=True
    )

    print(
        'Interpolation: none during reorientation; '
        'linear during mapping',
        flush=True
    )

    # Read AHEAD reference
    ahead_path, ahead_img = ahead_reference()

    print(
        f'AHEAD reference: {ahead_path}',
        flush=True
    )

    # Preflight all requested subjects before registration
    jobs = []
    issues = []

    for subject in subjects:
        try:
            found = preflight_one(
                subject,
                ahead_img
            )

            jobs.extend(found)

            print(
                f'  {subject}: all 4 original maps match '
                'native mapping grid after reorientation',
                flush=True
            )

        except Exception as error:
            issues.append(
                (subject, str(error))
            )

            print(
                f'  PREFLIGHT ERROR: {subject}: {error}',
                file=sys.stderr,
                flush=True
            )

    # Refuse to overwrite without explicit authorization
    for job in jobs:
        existing = [
            p
            for p in (
                job['output'],
                job['metadata']
            )
            if p.exists()
        ]

        if existing and not args.overwrite:
            issues.append(
                (
                    job['subject'],
                    f'Output exists: {existing}. '
                    'Use --overwrite to archive and replace.'
                )
            )

    if issues:
        print(
            f'PREFLIGHT FAILED: {len(issues)} problem(s). '
            'No changes made.',
            file=sys.stderr,
            flush=True
        )

        return 1

    print(
        f'PREFLIGHT PASSED: '
        f'{len(jobs)} quantitative maps.',
        flush=True
    )

    # Stop here if only checking inputs
    if args.dry_run:
        for job in jobs:
            print(
                f'  {job["source"]}  ->  {job["output"]}',
                flush=True
            )

        print(
            'DRY RUN: no files changed.',
            flush=True
        )

        return 0

    # Import Nighres only for actual registration
    try:
        import nighres

    except ImportError as exc:
        print(
            f'ERROR: Nighres unavailable: {exc}',
            file=sys.stderr
        )

        return 2

    # Unique run identifier for archived files
    run_id = (
        datetime.now().strftime('%Y%m%d_%H%M%S')
        + f'_{os.getpid()}'
    )

    started = time.monotonic()

    reports = []
    failed = 0

    # Process each subject and quantitative map
    for i, job in enumerate(jobs, start=1):
        print(
            f'[{i}/{len(jobs)}] '
            f'{job["subject"]} {job["kind"]}',
            flush=True
        )

        try:
            archives = register_one(
                job,
                ahead_img,
                args.overwrite,
                run_id,
                nighres
            )

            status = 'CREATED'
            detail = '; '.join(archives)

            print(
                f'  CREATED: {job["output"]}',
                flush=True
            )

            for old in archives:
                print(
                    f'  BACKUP: {old}',
                    flush=True
                )

        except Exception:
            failed += 1

            status = 'FAILED'
            detail = traceback.format_exc()

            print(
                detail,
                file=sys.stderr,
                flush=True
            )

        reports.append(
            [
                job['subject'],
                job['kind'],
                str(job['source']),
                str(job['output']),
                status,
                detail
            ]
        )

        # Progress and ETA
        elapsed = time.monotonic() - started

        remaining = (
            elapsed / i
            * (len(jobs) - i)
        )

        print(
            f'  Progress {i}/{len(jobs)} '
            f'| elapsed {timedelta(seconds=int(elapsed))}'
            f' | ETA {timedelta(seconds=int(remaining))}',
            flush=True
        )

    # ========================================================
    # SAVE REGISTRATION LOG
    # ========================================================

    logs = ROOT / 'logs'

    logs.mkdir(
        parents=True,
        exist_ok=True
    )

    report = logs / (
        f'mpm_original_registration_{run_id}.csv'
    )

    with report.open(
        'x',
        newline='',
        encoding='utf-8'
    ) as stream:

        writer = csv.writer(stream)

        writer.writerow(
            [
                'subject',
                'map',
                'original_qmri_source',
                'output',
                'status',
                'backup_or_error'
            ]
        )

        writer.writerows(reports)

    print(
        f'Finished: {len(jobs) - failed}/{len(jobs)} '
        f'registered. Log: {report}',
        flush=True
    )

    return 1 if failed else 0


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
    ) as exc:

        print(
            f'ERROR: {exc}',
            file=sys.stderr,
            flush=True
        )

        sys.exit(1)
