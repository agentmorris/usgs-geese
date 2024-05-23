########
#
# usgs-geese-scrap.py
#
# Random bits of scrap code that were useful at some point that I don't quite want
# to delete, even in the git sense of deletion.
#
########

#%% Constants and imports

import os
import json
import humanfriendly

import pandas as pd

from tqdm import tqdm

from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as vis_utils

import importlib
usgs_geese_inference = importlib.import_module('usgs-geese-inference')
usgs_geese_postprocessing = importlib.import_module('usgs-geese-postprocessing')

from usgs_geese_inference import default_yolo_category_id_to_name
from usgs_geese_inference import default_counting_confidence_thresholds
from usgs_geese_postprocessing import patch_level_preview


#%% Scrap from usgs-geese-inference.py

if False:

    pass
    
    #%% Time estimates
    
    # Time to process all patches for an image on a single GPU
    seconds_per_image = 25
    n_workers = 2
    seconds_per_image /= n_workers
    
    drive_base = '/media/user/My Passport'
    
    estimate_time_for_old_data = False
    
    if estimate_time_for_old_data:
        base_folder = os.path.join(drive_base,'2017-2019')
        image_folder = os.path.join(base_folder,'01_JPGs')
        image_folder_name = image_folder
        images_absolute = path_utils.find_images(image_folder,recursive=True)
    else:
        images_absolute = []
        image_folder_name = '2022 images'
        root_filenames = os.listdir(drive_base)
        for fn in root_filenames:
            if fn.startswith('2022'):
                dirname = os.path.join(drive_base,fn)
                if os.path.isdir(dirname):
                    images_absolute.extend(path_utils.find_images(dirname,recursive=True))        
    
    total_time_seconds = seconds_per_image * len(images_absolute)
    
    print('Expected time for {} ({} images): {}'.format(
        image_folder_name,len(images_absolute),humanfriendly.format_timespan(total_time_seconds)))
    
    
    #%% Unused variable suppression
    
    patch_results_after_nms_file = None
    patch_folder_for_folder = None
    
    
    #%% Preview results for patches at a variety of confidence thresholds
    
    project_dir = None
    patch_results_file = patch_results_after_nms_file
            
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    postprocessing_output_folder = os.path.join(project_dir,'preview')

    base_task_name = os.path.basename(patch_results_file)
        
    for confidence_threshold in [0.4,0.5,0.6,0.7,0.8]:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder_for_folder
        options.include_almost_detections = True
        options.num_images_to_sample = 7500
        options.confidence_threshold = confidence_threshold
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        options.ground_truth_json_file = None
        options.separate_detections_by_category = True
        # options.sample_seed = 0
        
        options.parallelize_rendering = True
        options.parallelize_rendering_n_cores = 16
        options.parallelize_rendering_with_threads = False
        
        output_base = os.path.join(postprocessing_output_folder,
            base_task_name + '_{:.3f}'.format(options.confidence_threshold))
        
        os.makedirs(output_base, exist_ok=True)
        print('Processing to {}'.format(output_base))
        
        options.api_output_file = patch_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        path_utils.open_file(html_output_file)
    

    #%% Render boxes on one of the original images
    
    image_fn = ''
    
    input_folder_base = '/media/user/My Passport/2022-10-09/cam3'
    md_results_image_level_nms_fn = os.path.expanduser(
        '~/tmp/usgs-inference/image_level_results/'+ \
        'media_user_My_Passport_2022-10-09_cam3_md_results_image_level_nms.json')
    
    with open(md_results_image_level_nms_fn,'r') as f:
        md_results_image_level = json.load(f)

    i_image = 0
    output_image_file = os.path.join(project_dir,'test.jpg')
    detections = md_results_image_level['images'][i_image]['detections']    
    image_fn_relative = md_results_image_level['images'][i_image]['file']
    image_fn = os.path.join(input_folder_base,image_fn_relative)
    assert os.path.isfile(image_fn)
    
    yolo_category_id_to_name = default_yolo_category_id_to_name
    
    detector_label_map = {}
    for category_id in yolo_category_id_to_name:
        detector_label_map[str(category_id)] = yolo_category_id_to_name[category_id]
        
    vis_utils.draw_bounding_boxes_on_file(input_file=image_fn,
                          output_file=output_image_file,
                          detections=detections,
                          confidence_threshold=0.4,
                          detector_label_map=detector_label_map, 
                          thickness=1, 
                          expansion=0)
    
    path_utils.open_file(output_image_file)


#%% Scrap from usgs-geese-postprocessing.py

def image_level_counting_hd_compat(image_level_results_file,
                         image_name_prefix,
                         drive_root_path,
                         output_file=None,
                         overwrite=False,
                         counting_confidence_thresholds=None):
    """
    THIS IS A DEPRECATED VERSION OF IMAGE_LEVEL_COUNTING, temporarily maintained for 
    compability with old results generated for a particular drive.  It's complicated 
    and confusing, and has been superseded by image_level_counting()
    """
    
    """
    Given an image-level results file:
        
    * Count the number of occurrences in each class above a few different thresholds
    * Write the resulting counts to .csv
    
    'image_name_pefix' is everything between the drive root path, e.g.:
    
    /media/user/My Passport1'
     
    ...and a particular image in a results file.  E.g. for the results file:
    
    media_user_My_Passport_2022-10-16_md_results_image_level_nms.json
     
    ...in which filenames look like:
    
    CAM3/CAM30033.JPG
     
    The prefix will be:
    
    2022-10-12/
    
    **eval** is a special-case prefix for handling the eval set.
    """
    
    if counting_confidence_thresholds is None:
        counting_confidence_thresholds = default_counting_confidence_thresholds
    
    if output_file is None:
        output_file = image_level_results_file + '_counts.csv'

    if os.path.isfile(output_file) and not overwrite:
        print('Output file {} exists and overwrite=False, skiping'.format(output_file))
        return
        
    with open(image_level_results_file,'r') as f:
        image_level_results = json.load(f)
        
    print('Loaded image-level results for {} images'.format(
        len(image_level_results['images'])))
    
    # Make sure images are unique
    image_filenames = [im['file'] for im in image_level_results['images']]
    assert len(image_filenames) == len(set(image_filenames)), \
        'Image uniqueness error'
    
    category_names = image_level_results['detection_categories'].values()
    category_names = [s.lower() for s in category_names]
    category_id_to_name = {}
    for category_id in image_level_results['detection_categories']:
        category_id_to_name[category_id] = \
            image_level_results['detection_categories'][category_id].lower()
    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    
    # This will be a list of dicts with fields
    #
    # image_path_local (str)
    # image_path_original (str)
    # confidence_threshold
    # E.g.: 'count_brant', 'count_other', 'count_gull', 'count_canada', 'count_emperor'
    results = []
    
    # im = image_level_results['images'][0]
    for im in tqdm(image_level_results['images']):
        
        image_path_prefix_relative = im['file']
        
        if image_name_prefix == '**eval**':
            # For the eval set, filenames look like
            # 'val-images/2019_Replicate_2019-10-11_Cam3_CAM39080.JPG'            
            prefix = '2017-2019/01_JPGs/'
            fn = os.path.basename(image_path_prefix_relative)
            fn = fn.replace('Replicate_','Replicate*').replace('Out_lagoon','Out*lagoon')
            fn = fn.replace('_','/')
            fn = fn.replace('*','_')
            image_path_drive_relative = prefix + fn
            
        else:
            image_path_drive_relative = os.path.join(image_name_prefix,image_path_prefix_relative)
        
        image_path_absolute = os.path.join(drive_root_path,image_path_drive_relative)
        assert os.path.isfile(image_path_absolute), \
            'Image file {} does not exist'.format(image_path_absolute)
        
        for confidence_threshold_set in counting_confidence_thresholds:
            
            category_id_to_count = {}
            for cat_id in image_level_results['detection_categories'].keys():
                category_id_to_count[cat_id] = 0
            
            category_id_to_threshold = {}
            
            # If we're using the same confidence threshold for all classes
            if isinstance(confidence_threshold_set,float):
                for cat_id in category_id_to_count:
                    category_id_to_threshold[cat_id] = confidence_threshold_set
            # Otherwise this should map category *names* (not IDs) to thresholds
            else:
                assert isinstance(confidence_threshold_set,dict), \
                    'Counting threshold input error'
                assert len(category_name_to_id) == len(confidence_threshold_set), \
                    'Counting threshold category error'
                for category_name in category_name_to_id:
                    assert category_name in confidence_threshold_set, \
                        'No threshold mapping for category {}'.format(category_name)
                for category_name in confidence_threshold_set:
                    category_id_to_threshold[category_name_to_id[category_name]] = \
                        confidence_threshold_set[category_name]                        
            
            # det = im['detections'][0]
            for det in im['detections']:
                
                confidence_threshold = category_id_to_threshold[det['category']]
                if det['conf'] >= confidence_threshold:                
                    category_id_to_count[det['category']] = \
                        category_id_to_count[det['category']] + 1
            
            # ...for each detection
            
            im_results = {}
            im_results['image_path_relative'] = im['file']
            im_results['image_path_drive_relative'] = image_path_drive_relative
            im_results['confidence_threshold_string'] = str(confidence_threshold_set)
            
            for category_id in category_id_to_count:
                im_results['count_' + category_id_to_name[category_id]] = \
                    category_id_to_count[category_id]
                im_results['threshold_' + category_id_to_name[category_id]] = \
                    category_id_to_threshold[category_id]
        
            results.append(im_results)
            
        # ...for each confidence threhsold        
        
    # ...for each image
    
    # Convert to a dataframe
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_file,header=True,index=False)
    
# ...def image_level_counting_hd_compat(...)


#%% Interactive driver

if False:
    
    #%% Preview
    
    n_patches = 2500
    preview_confidence_thresholds = [0.45, 0.5, 0.55, 0.6]
    image_level_results_base = os.path.expanduser('~/tmp/usgs-inference/image_level_results')
    
    # Use this to specify non-default image folder bases for individual files
    result_file_to_folder_base = {}
    
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.json')]        
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     'nms' in fn]
    
    image_level_results_filenames.sort()

    result_file_to_folder_base[
        'media_user_My_Passport1_2017-2019_01_JPGs_2017_Replicate_2017-10-03_md_results_image_level_nms.json'
        ] = \
        '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-03'
    
    preview_folder_base = os.path.expanduser('~/tmp/usgs-inference/preview')    
    
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     ('2022-10' in fn)]
    html_files = []
    
    # i_file = 0; image_level_results_file = image_level_results_filenames[i_file]
    for i_file,image_level_results_file in enumerate(image_level_results_filenames):
        
        print('\Generating previews for file {} of {}'.format(
            i_file,len(image_level_results_filenames)))
        
        if image_level_results_file in result_file_to_folder_base:
            image_folder_base = result_file_to_folder_base[image_level_results_file]
        elif 'eval' in image_level_results_file:
            image_folder_base = os.path.expanduser('~/data/usgs-geese/eval_images')
        else:
            # 'media_user_My_Passport_2022-10-09_md_results_image_level.json'
            image_folder_base = '/' + '/'.join(image_level_results_file.replace(
                'My_Passport','My Passport').\
                                         split('_')[0:4])
        
        image_folder_base = image_folder_base.replace('My Passport/','My Passport1/')
        
        assert os.path.isdir(image_folder_base), \
            'Image folder {} does not exist'.format(image_folder_base)
    
        preview_folder = os.path.join(preview_folder_base,
                                      os.path.splitext(os.path.basename(image_level_results_file))[0])
                
        image_level_results_file_absolute = os.path.join(image_level_results_base,
                                                         image_level_results_file)
        image_html_files = patch_level_preview(image_level_results_file_absolute,
                                               image_folder_base,preview_folder,
                                               n_patches=n_patches,
                                               preview_confidence_thresholds=preview_confidence_thresholds)
        html_files.extend(image_html_files)
    
    # ...for each results file
    
    for fn in html_files:
        path_utils.open_file(fn)

    
    #%% Counting
    
    output_file = None
    overwrite = True
    counting_confidence_thresholds = None
    drive_root_path = '/media/user/My Passport'
    assert os.path.isdir(drive_root_path)
    
    image_level_results_base = os.path.expanduser('~/tmp/usgs-inference/image_level_results')
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.json')]
    image_level_results_filenames = [fn for fn in image_level_results_filenames if 'nms' in fn]
    
    image_level_results_filenames.sort()
        
    image_results_file_relative_to_prefix = {}
    image_results_file_relative_to_prefix[
        'media_user_My_Passport1_2017-2019_01_JPGs_2017_Replicate_2017-10-03_md_results_image_level.json'
        ] = '2017-2019/01_JPGs/2017/Replicate_2017-10-03'
    image_results_file_relative_to_prefix[
        'media_user_My_Passport1_2017-2019_01_JPGs_2017_Replicate_2017-10-03_md_results_image_level_nms.json'
        ] = '2017-2019/01_JPGs/2017/Replicate_2017-10-03'
    
    confidence_threshold_sets = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675]
    
    # E.g.: 'count_brant', 'count_other', 'count_gull', 'count_canada', 'count_emperor'
    confidence_threshold_sets.append(
        {'brant':0.65, 'other':0.65, 'gull':0.65, 'canada':0.625, 'emperor':0.6})
    
    # image_level_results_file_relative = image_level_results_filenames[0]
    for image_level_results_file_relative in image_level_results_filenames:
        
        if image_level_results_file_relative in image_results_file_relative_to_prefix:
            image_name_prefix = image_results_file_relative_to_prefix[image_level_results_file_relative]
        elif 'eval' in image_level_results_file_relative:
            image_name_prefix = '**eval**'
        else:
            # The prefix is everything between the universal root path, e.g.:
            #
            # /media/user/My Passport1'
            # 
            # ...and a particular image in a results file.  E.g. for the results file:
            #
            # media_user_My_Passport_2022-10-16_md_results_image_level_nms.json
            # 
            # ...in which filenames look like:
            #
            # CAM3/CAM30033.JPG
            # 
            # The prefix will be:
            #
            # 2022-10-12/
            image_name_prefix = os.path.basename(image_level_results_file_relative).\
                replace('My_Passport','My Passport').\
                split('_')[3] + '/'
            # assert image_name_prefix.startswith('2022') and len(image_name_prefix) == 11
    
        image_level_results_file = os.path.join(image_level_results_base,
                                                image_level_results_file_relative)
        
        image_level_counting_hd_compat(image_level_results_file,
                             image_name_prefix,
                             drive_root_path,
                             output_file=None,
                             overwrite=True,
                             counting_confidence_thresholds=confidence_threshold_sets)    

    # ...for each results file

    
    #%% Zip results files                            
    
    image_level_results_base = os.path.expanduser('~/tmp/usgs-inference/image_level_results')
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.csv')]
    image_level_results_filenames = [os.path.join(image_level_results_base,fn) for \
                                     fn in image_level_results_filenames]

    import zipfile
    from zipfile import ZipFile

    output_path = image_level_results_base

    def zip_file(fn, overwrite=True):
        
        basename = os.path.basename(fn)
        zip_file_name = os.path.join(output_path,basename + '.zip')
        
        if (not overwrite) and (os.path.isfile(zip_file_name)):
            print('Skipping existing file {}'.format(zip_file_name))
            return
        
        print('Zipping {} to {}'.format(fn,zip_file_name))
        
        with ZipFile(zip_file_name,'w',zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(fn,arcname=basename,compresslevel=9,compress_type=zipfile.ZIP_DEFLATED)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(len(image_level_results_filenames))
    with tqdm(total=len(image_level_results_filenames)) as pbar:
        for i,_ in enumerate(pool.imap_unordered(zip_file,image_level_results_filenames)):
            pbar.update()
