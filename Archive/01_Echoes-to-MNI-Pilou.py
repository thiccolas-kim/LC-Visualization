import nighres
import os
import nibabel
from glob import glob
import numpy
import scipy.ndimage
import random

echoes_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/LCPCA_distCorr/'
mapping_dir =  '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/'

coreg_dir = '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/derivatives/nighres/echoes_to_mni/'
os.makedirs(coreg_dir, exist_ok=True)
    
subjects = ['sub-004']

for subject in sorted(subjects, key=lambda x: random.random()):
    
    print("subject: "+subject)
    proc_dir = coreg_dir+subject+'/ses-02/anat/'
    os.makedirs(proc_dir, exist_ok=True)

    # find the needed input files
    print('search: '+echoes_dir+subject+'/ses-02/anat/')
    pdw_e1 = glob(echoes_dir+subject+'/ses-02/anat/'+subject+'*acq-PDw*echo-01*part-mag*.nii*')[0]
    t1w_e1 = glob(echoes_dir+subject+'/ses-02/anat/'+subject+'*acq-T1w*echo-01*part-mag*.nii*')[0]
    mtw_e1 = glob(echoes_dir+subject+'/ses-02/anat/'+subject+'*acq-MTw*echo-01*part-mag*.nii*')[0]
    
    # find the needed mapping files
    print('search: '+mapping_dir+subject+'/ses-02/anat/')
    mpm2mni = glob(mapping_dir+subject+'/ses-02/anat/'+subject+'_ses-02_R1map_brain_ants-map.nii*')[0]
    mni2ahead = glob(mapping_dir+subject+'/ses-02/anat/'+subject+'_ses-02_R1map_brain_ants-def0_ants-map.nii*')[0]
    
    # 1. reorient the echoes data (skull stripping could be done there too if needed)
    pdw_e1_reorient = pdw_e1.replace(echoes_dir,coreg_dir).replace('.nii','_reorient.nii')
    if not os.path.exists(pdw_e1_reorient):
        pdw_img = nighres.io.load_volume(pdw_e1)
        pdw_img = nibabel.as_closest_canonical(pdw_img)
        nighres.io.save_volume(pdw_e1_reorient, pdw_img)
    
    t1w_e1_reorient = t1w_e1.replace(echoes_dir,coreg_dir).replace('.nii','_reorient.nii')
    if not os.path.exists(t1w_e1_reorient):
        t1w_img = nighres.io.load_volume(t1w_e1)
        t1w_img = nibabel.as_closest_canonical(t1w_img)
        nighres.io.save_volume(t1w_e1_reorient, t1w_img)
    
    mtw_e1_reorient = mtw_e1.replace(echoes_dir,coreg_dir).replace('.nii','_reorient.nii')
    if not os.path.exists(mtw_e1_reorient):
        mtw_img = nighres.io.load_volume(mtw_e1)
        mtw_img = nibabel.as_closest_canonical(mtw_img)
        nighres.io.save_volume(mtw_e1_reorient, mtw_img)
    
 
    # 2. compute rigid registration between PDw and T1w, PDw and MTw
    t1w2pdw = nighres.registration.embedded_antspy(source_image=t1w_e1_reorient, target_image=pdw_e1_reorient,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=False,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=True, overwrite=False, output_dir=proc_dir,
                    file_name=subject+'_ses-02_coreg-t1w2pdw')['mapping']
    
    mtw2pdw = nighres.registration.embedded_antspy(source_image=mtw_e1_reorient, target_image=pdw_e1_reorient,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=False,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=True, overwrite=False, output_dir=proc_dir,
                    file_name=subject+'_ses-02_coreg-mtw2pdw')['mapping']
    
    
    # 3. transform echoes from original space into AHEAD space
    pdw_e1_ahead = nighres.registration.apply_coordinate_mappings(pdw_e1_reorient,
                            mapping1=mpm2mni,
                            mapping2=mni2ahead,
                            padding="zero",interpolation='nearest',
                            save_data=True, overwrite=False, output_dir=proc_dir, 
                            file_name=pdw_e1_reorient.replace('.nii','_align-ahead.nii'))['result']

    t1w_e1_ahead = nighres.registration.apply_coordinate_mappings(t1w_e1_reorient,
                            mapping1=t1w2pdw,
                            mapping2=mpm2mni,
                            mapping3=mni2ahead,
                            padding="zero",interpolation='nearest',
                            save_data=True, overwrite=False, output_dir=proc_dir, 
                            file_name=t1w_e1_reorient.replace('.nii','_align-ahead.nii'))['result']

    mtw_e1_ahead = nighres.registration.apply_coordinate_mappings(mtw_e1_reorient,
                            mapping1=mtw2pdw,
                            mapping2=mpm2mni,
                            mapping3=mni2ahead,
                            padding="zero",interpolation='nearest',
                            save_data=True, overwrite=False, output_dir=proc_dir, 
                            file_name=mtw_e1_reorient.replace('.nii','_align-ahead.nii'))['result']


    # 4. transform other echoes: repeat steps 1 and 3 (but not 2) for the later echoes
    
    
    
# 5. make group averages ...


