import nighres
import nibabel
import os
import inspect
import time
import re
import logging
import traceback
from datetime import datetime, timedelta
from glob import glob
from collections import defaultdict


# ============================================================
# DIRECTORIES AND SETTINGS
# ============================================================

echoes_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/LCPCA_distCorr/'
mapping_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/'
coreg_dir = os.path.join(mapping_dir, 'echoes_to_mni')

session = 'ses-02'
contrasts = ['PDw', 'T1w', 'MTw']

# Based on the observed acquisition protocol. Change these if necessary.
# Missing later echoes are logged, but available echoes are still processed.
expected_echo_counts = {'PDw': 6, 'T1w': 6, 'MTw': 4}

os.makedirs(coreg_dir, exist_ok=True)

# Discover every subject directory, including those lacking the requested session.
subjects = sorted(
    os.path.basename(path)
    for path in glob(os.path.join(echoes_dir, 'sub-*'))
    if os.path.isdir(path)
)


# ============================================================
# PERSISTENT ERROR LOG
# ============================================================

log_dir = os.path.join(coreg_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(
    log_dir,
    f'echo_processing_errors_{datetime.now():%Y%m%d_%H%M%S}.log'
)

error_logger = logging.getLogger('echo_processing_errors')
error_logger.setLevel(logging.INFO)
error_logger.propagate = False
error_logger.handlers.clear()
file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
error_logger.addHandler(file_handler)

issues_by_subject = defaultdict(list)


def log_issue(subject, category, details, exception=False):
    """Write a subject-specific problem to disk and the terminal."""
    message = f'{subject} | {category} | {details}'
    issues_by_subject[subject].append((category, details))
    if exception:
        error_logger.error('%s\n%s', message, traceback.format_exc())
    else:
        error_logger.warning(message)
    print(f'  WARNING: {message}', flush=True)


# ============================================================
# STEP 1: REORIENT IMAGE TO CANONICAL ORIENTATION
# ============================================================

def reorient_image(image_path, output_dir):
    """Reorient a NIfTI image to canonical (RAS+) orientation."""
    filename = os.path.basename(image_path)

    if filename.endswith('.nii.gz'):
        output_name = filename.replace('.nii.gz', '_reorient.nii.gz')
    else:
        output_name = filename.replace('.nii', '_reorient.nii')

    output_path = os.path.join(output_dir, output_name)

    if not os.path.exists(output_path):
        img = nighres.io.load_volume(image_path)
        img = nibabel.as_closest_canonical(img)
        nighres.io.save_volume(output_path, img)

    return output_path


# ============================================================
# STEP 2: RIGIDLY REGISTER IMAGE TO PDw
# ============================================================

def register_to_pdw(source_image, pdw_image, output_dir, file_name):
    """Compute the original rigid registration from source to PDw."""
    options = dict(
        source_image=source_image,
        target_image=pdw_image,
        run_rigid=True,
        rigid_iterations=1000,
        run_affine=False,
        affine_iterations=1000,
        run_syn=False,
        coarse_iterations=40,
        medium_iterations=50,
        fine_iterations=40,
        scaling_factor=8,
        cost_function='MutualInformation',
        interpolation='NearestNeighbor',
        regularization='High',
        convergence=1e-6,
        mask_zero=False,
        smooth_mask=0.0,
        ignore_affine=False,
        ignore_header=False,
        save_data=True,
        overwrite=False,
        output_dir=output_dir,
        file_name=file_name
    )

    # Retains the version-dependent behavior from the previous script.
    if 'run_similarity' in inspect.signature(
        nighres.registration.embedded_antspy
    ).parameters:
        options['run_similarity'] = False
        options['similarity_iterations'] = 1000

    result = nighres.registration.embedded_antspy(**options)
    return result['mapping']


# ============================================================
# STEP 3: TRANSFORM IMAGE INTO AHEAD SPACE
# ============================================================

def align_to_ahead(image_path, mpm2mni, mni2ahead,
                   output_dir, rigid_mapping=None):
    """Apply the same coordinate mappings and interpolation as before."""
    if rigid_mapping is None:
        mappings = dict(mapping1=mpm2mni, mapping2=mni2ahead)
    else:
        mappings = dict(
            mapping1=rigid_mapping,
            mapping2=mpm2mni,
            mapping3=mni2ahead
        )

    filename = os.path.basename(image_path)

    if filename.endswith('.nii.gz'):
        output_name = filename.replace('.nii.gz', '_align-ahead.nii.gz')
    else:
        output_name = filename.replace('.nii', '_align-ahead.nii')

    result = nighres.registration.apply_coordinate_mappings(
        image_path,
        **mappings,
        padding='zero',
        interpolation='nearest',
        save_data=True,
        overwrite=False,
        output_dir=output_dir,
        file_name=output_name
    )

    return result['result']


# ============================================================
# PREFLIGHT: FIND INPUTS AND LOG MISSING / AMBIGUOUS FILES
# ============================================================

def find_unique_mapping(subject, map_dir, pattern, label):
    """Return exactly one mapping, otherwise log the problem."""
    matches = sorted(p for p in glob(os.path.join(map_dir, pattern))
                     if os.path.isfile(p))

    if not matches:
        log_issue(subject, 'MISSING MAPPING',
                  f'{label}: expected {os.path.join(map_dir, pattern)}')
        return None

    if len(matches) > 1:
        log_issue(subject, 'AMBIGUOUS MAPPING',
                  f'{label}: found {len(matches)} matches: {matches}')
        return None

    return matches[0]


def check_subject(subject):
    """Check inputs before processing, returning a job or None if unsafe."""
    input_dir = os.path.join(echoes_dir, subject, session, 'anat')
    proc_dir = os.path.join(coreg_dir, subject, session, 'anat')
    map_dir = os.path.join(mapping_dir, subject, session, 'anat')

    print(f'Checking {subject}', flush=True)

    if not os.path.isdir(input_dir):
        log_issue(subject, 'MISSING DIRECTORY', input_dir)
        return None

    echoes = {}
    valid = True

    for contrast in contrasts:
        pattern = os.path.join(
            input_dir, f'{subject}*acq-{contrast}*echo-*part-mag*.nii*'
        )
        paths = sorted(p for p in glob(pattern) if os.path.isfile(p))
        by_number = {}

        for path in paths:
            match = re.search(r'_echo-(\d+)_', os.path.basename(path))
            if match is None:
                log_issue(subject, 'UNRECOGNIZED ECHO FILENAME', path)
                continue

            echo_num = int(match.group(1))
            if echo_num in by_number:
                log_issue(
                    subject, 'DUPLICATE ECHO',
                    f'{contrast} echo-{echo_num:02d}: '
                    f'{by_number[echo_num]} AND {path}'
                )
                valid = False
            else:
                by_number[echo_num] = path

        echoes[contrast] = dict(sorted(by_number.items()))
        print(f'  {contrast}: found {len(echoes[contrast])} echoes', flush=True)

        if 1 not in by_number:
            log_issue(subject, 'MISSING REQUIRED FIRST ECHO',
                      f'{contrast} echo-01: searched {pattern}')
            valid = False

        # Report expected missing echo numbers without discarding other echoes.
        for echo_num in range(1, expected_echo_counts[contrast] + 1):
            if echo_num not in by_number:
                log_issue(
                    subject, 'MISSING ECHO',
                    f'{contrast} echo-{echo_num:02d}: '
                    f'no match in {input_dir}'
                )

    mpm2mni = find_unique_mapping(
        subject, map_dir,
        f'{subject}_{session}_R1map_brain_ants-map.nii*',
        'MPM to MNI'
    )
    mni2ahead = find_unique_mapping(
        subject, map_dir,
        f'{subject}_{session}_R1map_brain_ants-def0_ants-map.nii*',
        'MNI to AHEAD'
    )

    if mpm2mni is None or mni2ahead is None:
        valid = False

    if not valid:
        print(f'  SKIPPING {subject}: missing or ambiguous essential input\n',
              flush=True)
        return None

    return dict(
        subject=subject,
        proc_dir=proc_dir,
        echoes=echoes,
        mpm2mni=mpm2mni,
        mni2ahead=mni2ahead
    )


# ============================================================
# BUILD THE VALID WORK QUEUE FIRST (ACCURATE PROGRESS TOTAL)
# ============================================================

start_time = time.monotonic()
print(f'Found {len(subjects)} subject directories.', flush=True)
print(f'Error log: {log_path}\n', flush=True)

jobs = []
for subject in subjects:
    job = check_subject(subject)
    if job is not None:
        jobs.append(job)

total_echoes = sum(
    len(job['echoes'][contrast])
    for job in jobs
    for contrast in contrasts
)

succeeded_echoes = 0
failed_echoes = 0
skipped_echoes = 0


def show_progress():
    handled = succeeded_echoes + failed_echoes + skipped_echoes
    elapsed = time.monotonic() - start_time
    remaining = (elapsed / handled) * (total_echoes - handled) if handled else 0
    percent = 100 * handled / total_echoes if total_echoes else 100
    eta = str(timedelta(seconds=int(remaining))) if handled else 'unknown'

    print(
        f'  Progress: {handled}/{total_echoes} ({percent:.1f}%) | '
        f'Success: {succeeded_echoes} | Failed: {failed_echoes} | '
        f'Skipped: {skipped_echoes} | '
        f'Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta}',
        flush=True
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

for job in jobs:
    subject = job['subject']
    proc_dir = job['proc_dir']
    echoes = job['echoes']
    mpm2mni = job['mpm2mni']
    mni2ahead = job['mni2ahead']
    subject_echo_count = sum(len(echoes[c]) for c in contrasts)

    print(f'\nProcessing {subject}', flush=True)
    os.makedirs(proc_dir, exist_ok=True)

    # Steps 1 and 2: reorient first echoes and calculate rigid maps only once.
    try:
        first_echoes = {
            contrast: reorient_image(echoes[contrast][1], proc_dir)
            for contrast in contrasts
        }

        t1w2pdw = register_to_pdw(
            first_echoes['T1w'], first_echoes['PDw'], proc_dir,
            f'{subject}_{session}_coreg-t1w2pdw'
        )
        mtw2pdw = register_to_pdw(
            first_echoes['MTw'], first_echoes['PDw'], proc_dir,
            f'{subject}_{session}_coreg-mtw2pdw'
        )
    except Exception as exc:
        log_issue(subject, 'REGISTRATION FAILED', str(exc), exception=True)
        skipped_echoes += subject_echo_count
        show_progress()
        continue

    rigid_mappings = {'PDw': None, 'T1w': t1w2pdw, 'MTw': mtw2pdw}

    # Steps 1 and 3: reuse mappings for all echoes, including echo 1.
    for contrast in contrasts:
        print(f'\nAligning all {contrast} echoes to AHEAD...', flush=True)

        for echo_num, echo_path in echoes[contrast].items():
            try:
                reoriented = reorient_image(echo_path, proc_dir)
                align_to_ahead(
                    reoriented, mpm2mni, mni2ahead, proc_dir,
                    rigid_mapping=rigid_mappings[contrast]
                )
                succeeded_echoes += 1
                print(f'  Completed: {os.path.basename(echo_path)}', flush=True)
            except Exception as exc:
                failed_echoes += 1
                log_issue(
                    subject, 'ECHO PROCESSING FAILED',
                    f'{contrast} echo-{echo_num:02d} ({echo_path}): {exc}',
                    exception=True
                )
            show_progress()

    print(f'\nFinished processing {subject}', flush=True)


# ============================================================
# FINAL SUMMARY / ERROR LOG
# ============================================================

runtime = timedelta(seconds=int(time.monotonic() - start_time))
skipped_subject_count = len(subjects) - len(jobs)

print('\n' + '=' * 60)
print('PROCESSING SUMMARY')
print(f'Subject directories found: {len(subjects)}')
print(f'Subjects eligible for processing: {len(jobs)}')
print(f'Subjects skipped at preflight: {skipped_subject_count}')
print(f'Echoes successfully processed/reused: {succeeded_echoes}')
print(f'Echoes failed during processing: {failed_echoes}')
print(f'Echoes skipped due to registration failure: {skipped_echoes}')
print(f'Total runtime: {runtime}')
print(f'Error log: {log_path}')

if issues_by_subject:
    print('\nSubjects with logged issues:')
    for subject in sorted(issues_by_subject):
        print(f'  {subject}:')
        for category, details in issues_by_subject[subject]:
            print(f'    - {category}: {details}')
else:
    print('No missing files or processing errors detected.')
    error_logger.info('No missing files or processing errors detected.')

file_handler.close()
error_logger.removeHandler(file_handler)

# ============================================================
# STEP 5: GROUP AVERAGES (TO BE IMPLEMENTED)
# ============================================================
