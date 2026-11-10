#!/usr/bin/env python3
"""Register LC MT-on/off slabs to PDw, then reuse PDw->MNI->AHEAD maps.

This script can be run in two environments:

  # FSL-enabled host shell (no NiBabel or Nighres needed):
  python3 -u 04_LCslabs_to_AHEAD.py --stage pdw

  # Nighres/Apptainer Python environment (no FSL needed):
  python3 -u 04_LCslabs_to_AHEAD.py --stage ahead

Or, if FSL and Nighres are available together:
  python3 -u 04_LCslabs_to_AHEAD.py --stage all

Default method: FSL header-based alignment (-applyxfm -usesqform),
which assumes trustworthy LC-slab and PDw NIfTI geometry. It does NOT
estimate an LC-to-PDw rigid registration from image intensities.

Optional: --mode translation estimates a translation-only refinement using
LC MT-off after header alignment, and applies the same refinement to MT-on.
This requires QC before interpreting the outputs.

Existing results are reused; no older data are overwritten.
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

BIDS = Path('/data/pt_03187/data/in_vivo/bids')
NIGHRES_ROOT = BIDS / 'derivatives/LORAKS/derivatives/nighres'
ROOT = NIGHRES_ROOT / 'echoes_to_mni'
SESSION = 'ses-02'
CONTRASTS = ('PDw', 'T1w', 'MTw')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('all', 'pdw', 'ahead'), default='all',
                        help='pdw = FSL slab-to-PDw; ahead = Nighres PDw-to-AHEAD')
    parser.add_argument('--mode', choices=('header', 'translation'), default='header',
                        help='header is the original -applyxfm -usesqform approach (default)')
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='Optional subset, e.g. --subjects sub-004 sub-011')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs and list planned work, without modifying images')
    return parser.parse_args()


def unique(path_or_pattern, subject, label, issues):
    paths = sorted(Path(p) for p in glob(str(path_or_pattern)) if Path(p).is_file())
    if len(paths) != 1:
        issue = ('MISSING' if len(paths) == 0 else 'AMBIGUOUS')
        issues.append((subject, issue, label,
                       f'{path_or_pattern} (found {len(paths)})'))
        return None
    return paths[0]


def get_jobs(subjects, issues):
    jobs = []
    for subject in subjects:
        out_dir = ROOT / subject / SESSION / 'anat'
        raw_dir = BIDS / subject / SESSION / 'anat'
        map_dir = NIGHRES_ROOT / subject / SESSION / 'anat'

        required = {}
        required['pdw_native'] = unique(
            out_dir / f'{subject}*acq-PDw*echo-01*part-mag*_reorient.nii*',
            subject, 'PDw echo-01 reoriented reference', issues)

        # Require all first-echo qMRI source images already AHEAD-aligned.
        for contrast in CONTRASTS:
            required[f'{contrast}_ahead'] = unique(
                out_dir / f'{subject}*acq-{contrast}*echo-01*reorient_align-ahead_def-img.nii*',
                subject, f'{contrast} echo-01 AHEAD output', issues)

        required['mpm2mni'] = unique(
            map_dir / f'{subject}_{SESSION}_R1map_brain_ants-map.nii*',
            subject, 'MPM-to-MNI mapping', issues)
        required['mni2ahead'] = unique(
            map_dir / f'{subject}_{SESSION}_R1map_brain_ants-def0_ants-map.nii*',
            subject, 'MNI-to-AHEAD mapping', issues)

        for state in ('off', 'on'):
            required[f'lc_{state}'] = unique(
                raw_dir / f'{subject}_{SESSION}_mt-{state}_chunk-LCslab_MTR.nii*',
                subject, f'LC slab MT-{state}', issues)

        if all(p is not None for p in required.values()):
            jobs.append({'subject': subject, 'out_dir': out_dir, **required})
        else:
            print(f'SKIP {subject}: missing or ambiguous prerequisite(s)', flush=True)
    return jobs


def flair_flirt():
    exe = shutil.which('flirt')
    if exe:
        return exe
    fsldir = os.environ.get('FSLDIR', '')
    if fsldir:
        path = Path(fsldir) / 'bin/flirt'
        if path.is_file():
            return str(path)
    raise RuntimeError(
        'Cannot find FSL flirt. Run --stage pdw from a shell with FSL loaded, '
        'or set FSLDIR to the FSL installation directory.')


def run_command(argv):
    print('  $ ' + ' '.join(str(v) for v in argv), flush=True)
    subprocess.run([str(v) for v in argv], check=True)


def validate_output(path, subject, label):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise RuntimeError(f'{subject}: {label} missing/empty after processing: {path}')


def generate_pdw_images(job, mode, flirt_exe):
    """Resample both raw LC slabs onto the saved reoriented PDw grid."""
    subject, out_dir = job['subject'], job['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = job['pdw_native']
    header_paths = {}
    for state in ('off', 'on'):
        output = out_dir / f'{subject}_{SESSION}_mt-{state}_chunk-LCslab_MTR_in-PDw_header.nii.gz'
        if not output.is_file():
            run_command([
                flirt_exe, '-in', job[f'lc_{state}'], '-ref', ref,
                '-applyxfm', '-usesqform', '-interp', 'trilinear',
                '-out', output,
            ])
        else:
            print(f'  Reusing header-aligned MT-{state}: {output.name}', flush=True)
        validate_output(output, subject, f'MT-{state} header alignment')
        header_paths[state] = output

    if mode == 'header':
        return header_paths

    # Optional translation-only refinement. Estimate from MT-off in PDw grid,
    # then reuse that exact refinement for MT-on (also already in PDw grid).
    fsldir = os.environ.get('FSLDIR')
    if not fsldir:
        # Derive FSLDIR from executable when FSLDIR isn't exported.
        fsldir = str(Path(flirt_exe).resolve().parent.parent)
    schedule = Path(fsldir) / 'etc/flirtsch/sch3Dtrans_3dof'
    if not schedule.is_file():
        raise RuntimeError(f'Translation-only FLIRT schedule not found: {schedule}')

    mat = out_dir / f'{subject}_{SESSION}_LCslab_header_to_PDw_translation.mat'
    translated_off = out_dir / f'{subject}_{SESSION}_mt-off_chunk-LCslab_MTR_in-PDw_translation.nii.gz'
    translated_on = out_dir / f'{subject}_{SESSION}_mt-on_chunk-LCslab_MTR_in-PDw_translation.nii.gz'

    if not (mat.is_file() and translated_off.is_file()):
        run_command([
            flirt_exe, '-in', header_paths['off'], '-ref', ref,
            '-nosearch', '-cost', 'normmi', '-schedule', schedule,
            '-dof', '3', '-interp', 'trilinear',
            '-omat', mat, '-out', translated_off,
        ])
    else:
        print(f'  Reusing translation-only MT-off: {translated_off.name}', flush=True)

    validate_output(mat, subject, 'translation matrix')
    validate_output(translated_off, subject, 'translated MT-off')
    if not translated_on.is_file():
        run_command([
            flirt_exe, '-in', header_paths['on'], '-ref', ref,
            '-applyxfm', '-init', mat, '-interp', 'trilinear',
            '-out', translated_on,
        ])
    validate_output(translated_on, subject, 'translated MT-on')
    return {'off': translated_off, 'on': translated_on}


def expected_pdw_paths(job, mode):
    subject, out_dir = job['subject'], job['out_dir']
    return {
        state: out_dir / f'{subject}_{SESSION}_mt-{state}_chunk-LCslab_MTR_in-PDw_{mode}.nii.gz'
        for state in ('off', 'on')
    }


def check_pdw_grid(job, image):
    import nibabel as nib
    import numpy as np
    reference = nib.load(str(job['pdw_native']))
    candidate = nib.load(str(image))
    if reference.shape[:3] != candidate.shape[:3] or not np.allclose(
            reference.affine, candidate.affine, atol=1e-4, rtol=0):
        raise ValueError(
            f'LC slab resampled image does not match native PDw grid: {image}; '
            f'reference={job["pdw_native"]}')


def apply_ahead(job, pdw_images, mode):
    import nibabel as nib
    import numpy as np
    import nighres

    result_paths = {}
    for state in ('off', 'on'):
        image = pdw_images[state]
        validate_output(image, job['subject'], f'MT-{state} PDw image')
        check_pdw_grid(job, image)

        # Now image is on the same PDw grid as the first echo that was mapped
        # previously; reuse *two* mappings exactly as for PDw echo 1.
        result = nighres.registration.apply_coordinate_mappings(
            str(image),
            mapping1=str(job['mpm2mni']),
            mapping2=str(job['mni2ahead']),
            padding='zero',
            interpolation='nearest',  # exactly as original echo-to-AHEAD script
            save_data=True,
            overwrite=False,
            output_dir=str(job['out_dir']),
            file_name=f'{job["subject"]}_{SESSION}_mt-{state}_chunk-LCslab_MTR_in-PDw_{mode}_align-ahead.nii.gz',
        )['result']
        validate_output(result, job['subject'], f'MT-{state} AHEAD result')

        # Confirm actual output geometry equals previously registered PDw echo 1.
        output = nib.load(str(result))
        ahead_ref = nib.load(str(job['PDw_ahead']))
        if output.shape[:3] != ahead_ref.shape[:3] or not np.allclose(
                output.affine, ahead_ref.affine, atol=1e-4, rtol=0):
            raise ValueError(f'AHEAD grid mismatch for {job["subject"]} MT-{state}')
        result_paths[state] = str(result)
        print(f'  Saved/reused AHEAD MT-{state}: {result}', flush=True)

    return result_paths


def write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    args = parse_args()
    started = time.monotonic()
    log_dir = ROOT / 'logs'
    # Dry run is read-only; do not create directories in dry-run mode.
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'lcslab_to_ahead_{args.stage}_{args.mode}_{stamp}.log'
    issues_csv = log_dir / f'lcslab_to_ahead_issues_{args.stage}_{args.mode}_{stamp}.csv'
    manifest_csv = log_dir / f'lcslab_to_ahead_manifest_{args.stage}_{args.mode}_{stamp}.csv'

    # Discover only subjects with existing echo-registration output directories.
    subjects = sorted(p.name for p in ROOT.glob('sub-*') if p.is_dir())
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    issues = []
    jobs = get_jobs(subjects, issues)
    print(f'Candidate subjects: {len(subjects)}; eligible: {len(jobs)}; '
          f'stage={args.stage}; registration={args.mode}', flush=True)

    for job in jobs:
        print(f'  ELIGIBLE: {job["subject"]}', flush=True)

    if args.dry_run:
        for issue in issues:
            print('  ISSUE: ' + ' | '.join(issue), flush=True)
        print('Dry run complete: no files modified.', flush=True)
        return 0

    # Persist preflight results before any expensive processing.
    write_csv(issues_csv, issues, ('subject', 'type', 'item', 'details'))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    logging.info('Preflight: candidate=%s eligible=%s', len(subjects), len(jobs))
    flirt_exe = None
    if args.stage in ('pdw', 'all'):
        flirt_exe = flair_flirt()
    if args.stage in ('ahead', 'all'):
        try:
            import nighres  # noqa: F401
            import nibabel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'Nighres and NiBabel are required for --stage ahead. '
                'Run this stage in the Nighres Apptainer environment.') from exc

    manifest = []
    successes, failures = 0, 0
    for idx, job in enumerate(jobs, 1):
        subject = job['subject']
        print(f'\n[{idx}/{len(jobs)}] Processing {subject}', flush=True)
        try:
            if args.stage in ('pdw', 'all'):
                pdw_images = generate_pdw_images(job, args.mode, flirt_exe)
            else:
                pdw_images = expected_pdw_paths(job, args.mode)
            ahead_paths = {}
            if args.stage in ('ahead', 'all'):
                ahead_paths = apply_ahead(job, pdw_images, args.mode)
            manifest.append((subject, args.stage, args.mode,
                             str(pdw_images['off']), str(pdw_images['on']),
                             str(ahead_paths.get('off', '')),
                             str(ahead_paths.get('on', '')), 'OK'))
            successes += 1
        except Exception as exc:
            failures += 1
            issues.append((subject, 'PROCESSING ERROR', args.stage, str(exc)))
            logging.error('%s: %s\n%s', subject, exc, traceback.format_exc())
            print(f'  ERROR: {subject}: {exc}', flush=True)
            manifest.append((subject, args.stage, args.mode,
                             '', '', '', '', f'ERROR: {exc}'))
        elapsed = time.monotonic() - started
        remaining = elapsed / idx * (len(jobs) - idx)
        print(f'Progress {idx}/{len(jobs)} | OK {successes} | Errors {failures} | '
              f'Elapsed {timedelta(seconds=int(elapsed))} | '
              f'ETA {timedelta(seconds=int(remaining))}', flush=True)
        # Update CSVs continuously so a terminated run still has a record.
        write_csv(issues_csv, issues, ('subject', 'type', 'item', 'details'))
        write_csv(manifest_csv, manifest,
                  ('subject', 'stage', 'registration_mode', 'pdw_mt_off',
                   'pdw_mt_on', 'ahead_mt_off', 'ahead_mt_on', 'status'))

    print('\n=== LC SLAB PROCESSING SUMMARY ===')
    print(f'Eligible subjects: {len(jobs)}, success: {successes}, failed: {failures}')
    print(f'Issue report: {issues_csv}')
    print(f'Manifest: {manifest_csv}')
    print(f'Traceback log: {log_file}')
    print(f'Total runtime: {timedelta(seconds=int(time.monotonic()-started))}')
    print('Inspect MT-on and MT-off slab overlap with PDw in BOTH native and AHEAD spaces.')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

