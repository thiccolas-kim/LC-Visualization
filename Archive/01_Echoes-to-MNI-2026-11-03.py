
import nighres
import nibabel
import os
import inspect
import time
from datetime import timedelta
from glob import glob


# ============================================================
# DIRECTORIES AND SETTINGS
# ============================================================

echoes_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/LCPCA_distCorr/'
mapping_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/'
coreg_dir = os.path.join(mapping_dir, 'echoes_to_mni')

session = 'ses-02'
subjects = sorted(
    os.path.basename(subject_dir)
    for subject_dir in glob(os.path.join(echoes_dir, 'sub-*'))
    if os.path.isdir(os.path.join(subject_dir, session, 'anat'))
)
contrasts = ['PDw', 'T1w', 'MTw']

os.makedirs(coreg_dir, exist_ok=True)


# ============================================================
# PROGRESS TRACKING
# ============================================================

total_echoes = sum(
    len(glob(os.path.join(
        echoes_dir, subject, session, 'anat',
        f'{subject}*acq-{contrast}*echo-*part-mag*.nii*'
    )))
    for subject in subjects
    for contrast in contrasts
)

completed_echoes = 0
start_time = time.monotonic()

print(f'Found {len(subjects)} subjects and {total_echoes} echoes.')


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
    """Compute rigid registration from source image to PDw."""

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

    # Some Nighres versions support an additional similarity stage.
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
    """Apply coordinate mappings to transform an image to AHEAD."""

    # PDw needs two mappings; T1w and MTw need three.
    if rigid_mapping is None:
        mappings = dict(
            mapping1=mpm2mni,
            mapping2=mni2ahead
        )
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
# MAIN PIPELINE
# ============================================================

for subject in subjects:

    print(f'\nProcessing {subject}')

    input_dir = os.path.join(echoes_dir, subject, session, 'anat')
    proc_dir = os.path.join(coreg_dir, subject, session, 'anat')
    map_dir = os.path.join(mapping_dir, subject, session, 'anat')

    os.makedirs(proc_dir, exist_ok=True)

    # --------------------------------------------------------
    # FIND ALL ECHOES USING GLOB
    # --------------------------------------------------------

    echoes = {}

    for contrast in contrasts:

        pattern = os.path.join(
            input_dir,
            f'{subject}*acq-{contrast}*echo-*part-mag*.nii*'
        )

        echoes[contrast] = sorted(glob(pattern))

        if not echoes[contrast]:
            raise FileNotFoundError(f'No {contrast} echoes found: {pattern}')

        print(f'{contrast}: found {len(echoes[contrast])} echoes')

    # --------------------------------------------------------
    # FIND EXISTING MAPPINGS
    # --------------------------------------------------------

    mpm2mni = glob(os.path.join(
        map_dir, f'{subject}_{session}_R1map_brain_ants-map.nii*'
    ))[0]

    mni2ahead = glob(os.path.join(
        map_dir, f'{subject}_{session}_R1map_brain_ants-def0_ants-map.nii*'
    ))[0]

    # --------------------------------------------------------
    # FIND AND REORIENT FIRST ECHOES (STEP 1)
    # --------------------------------------------------------

    first_echoes = {}

    for contrast in contrasts:

        first_echo = next(
            (f for f in echoes[contrast]
             if 'echo-01' in os.path.basename(f)),
            None
        )

        if first_echo is None:
            raise FileNotFoundError(f'No echo-01 found for {contrast}')

        first_echoes[contrast] = reorient_image(first_echo, proc_dir)

    # --------------------------------------------------------
    # CALCULATE RIGID REGISTRATIONS ONCE (STEP 2)
    # --------------------------------------------------------

    t1w2pdw = register_to_pdw(
        first_echoes['T1w'],
        first_echoes['PDw'],
        proc_dir,
        f'{subject}_{session}_coreg-t1w2pdw'
    )

    mtw2pdw = register_to_pdw(
        first_echoes['MTw'],
        first_echoes['PDw'],
        proc_dir,
        f'{subject}_{session}_coreg-mtw2pdw'
    )

    # Save the mappings for reuse with every echo
    rigid_mappings = {
        'PDw': None,
        'T1w': t1w2pdw,
        'MTw': mtw2pdw
    }

    # --------------------------------------------------------
    # REORIENT AND TRANSFORM ALL ECHOES (STEPS 1 AND 3)
    # --------------------------------------------------------

    for contrast in contrasts:

        print(f'\nAligning all {contrast} echoes to AHEAD...')

        for echo_path in echoes[contrast]:

            # Step 1: Reorient (skips if already done)
            reoriented = reorient_image(echo_path, proc_dir)

            # Step 3: Transform into AHEAD using existing mappings
            aligned = align_to_ahead(
                reoriented,
                mpm2mni,
                mni2ahead,
                proc_dir,
                rigid_mapping=rigid_mappings[contrast]
            )

            print(f'  Completed: {os.path.basename(echo_path)}')

            # Progress and estimated time remaining
            completed_echoes += 1
            elapsed = time.monotonic() - start_time
            remaining = (elapsed / completed_echoes) * (
                total_echoes - completed_echoes
            )

            print(
                f'  Progress: {completed_echoes}/{total_echoes} '
                f'({100 * completed_echoes / total_echoes:.1f}%) | '
                f'Elapsed: {timedelta(seconds=int(elapsed))} | '
                f'ETA: {timedelta(seconds=int(remaining))}'
            )

    print(f'\nFinished processing {subject}')

print(
    f'\nTotal runtime: '
    f'{timedelta(seconds=int(time.monotonic() - start_time))}'
)

# ============================================================
# STEP 5: GROUP AVERAGES (TO BE IMPLEMENTED)
# ============================================================
