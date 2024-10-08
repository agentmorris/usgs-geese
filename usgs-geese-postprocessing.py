########
#
# usgs-geese-postprocessing.py
#
# Stuff we do with model results after inference:
#
# * Generate patch-level preview pages given a set of image-level results.
#
# * Generate estimated image-level counts from image-level results (and write to .csv).
#
########

#%% Constants and imports

import os
import json
import random
import pandas as pd

from tqdm import tqdm

import importlib
usgs_geese_inference = importlib.import_module('usgs-geese-inference')

from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils import path_utils
from megadetector.detection import run_tiled_inference

default_preview_confidence_thresholds = [0.4,0.5,0.6,0.7,0.8]
default_counting_confidence_thresholds = [0.4,0.5,0.6,0.7,0.8]


#%% Support functions

def intersects(box1,box2):
    """
    Determine whether two rectangles in x,y,w,h format intersect
    """
    
    # Convert to x1,y1,x2,y2 format
    r1 = [box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]]
    r2 = [box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]]
    
    if (r1[0]>=r2[2]) or (r1[2]<=r2[0]) or (r1[3]<=r2[1]) or (r1[1]>=r2[3]):
        return False
    else:
        return True


#%% Postprocessing functions

def patch_level_preview(image_level_results_file,image_folder_base,preview_folder,
                        n_patches=10000,preview_confidence_thresholds=None,random_seed=0):
    """
    Given an image-level results file:
    
    * Map each image name to a list of detections
    * Randomly sample patches from that results file
    * For each sampled patch
      * Find all results within that patch, convert to patch-relative coordinates
      * Extract the patch and write to disk
    * Generate the preview page, including counts for each patch
    """
    
    if preview_confidence_thresholds is None:
        preview_confidence_thresholds = default_preview_confidence_thresholds
        
    assert os.path.isfile(image_level_results_file), \
        'Results file {} does not exist'.format(image_level_results_file)
    assert os.path.isdir(image_folder_base), \
        'Image folder {} does not exist'.format(image_folder_base)
        
    os.makedirs(preview_folder,exist_ok=True)
    
    with open(image_level_results_file,'r') as f:
        image_level_results = json.load(f)
        
    print('Loaded results for {} images from {}'.format(
        len(image_level_results['images']),image_level_results_file))
            
    relative_image_fn_to_results = {}
    
    # Map image filenames to results for that image
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
        relative_image_fn_to_results[im['file']] = im
      
    # Convert relative coordinates to absolute coordinates
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
        
        # Coordinates are x/y/w/h, in normalized coordinates
        image_size = [im['w'],im['h']]
        
        detections_absolute = []
        # det = im['detections'][0]
        for det in im['detections']:
            assert 'bbox' in det and len(det['bbox']) == 4, 'Results file validation error'
            bbox_absolute = [
                det['bbox'][0] * image_size[0],
                det['bbox'][1] * image_size[1],
                det['bbox'][2] * image_size[0],
                det['bbox'][3] * image_size[1]
                ]
            det_absolute = {}
            det_absolute['bbox'] = bbox_absolute
            det_absolute['conf'] = det['conf']
            det_absolute['category'] = det['category']
            detections_absolute.append(det_absolute)
        
        im['detections_absolute'] = detections_absolute
        assert len(im['detections_absolute']) == len(im['detections']), \
            'Results file conversion error'
        
    # ...for each image        
    
    del image_size
    
    patch_size = usgs_geese_inference.patch_size
    
    print('Sampling {} patches from {} images'.format(
        n_patches,len(relative_image_fn_to_results)))

    # Generate a list of all image/patch tuples to sample from
    all_image_patch_tuples = []
    for i_image,fn in enumerate(relative_image_fn_to_results.keys()):
        im = relative_image_fn_to_results[fn]
        image_size = (im['w'],im['h'])
        patch_boundaries = run_tiled_inference.get_patch_boundaries(image_size,
          patch_size,patch_stride=None)
        
        for i_patch,patch_xy in enumerate(patch_boundaries):
            all_image_patch_tuples.append((fn,patch_xy))
            
    # Sample randomly from the list of all patches
    if len(all_image_patch_tuples) <= n_patches:
        sampled_patch_tuples = all_image_patch_tuples
    else:
        random.seed(random_seed)
        sampled_patch_tuples = random.sample(all_image_patch_tuples,n_patches)
    
    patch_folder = os.path.join(preview_folder,'patches')
    os.makedirs(patch_folder,exist_ok=True)
    
    patch_level_results = {}
    patch_level_results['info'] = image_level_results['info']
    patch_level_results['detection_categories'] = image_level_results['detection_categories']
    patch_level_results['images'] = []
    
    # i_patch = 0; patch = sampled_patch_tuples[i_patch]
    for i_patch,patch in tqdm(enumerate(sampled_patch_tuples),total=len(sampled_patch_tuples)):
    
        image_fn_relative = patch[0]
        image_fn_absolute = os.path.join(image_folder_base,image_fn_relative)
        assert os.path.isfile(image_fn_absolute), \
            'Image {} does not exist'.format(image_fn_absolute)
        patch_xy = patch[1]
        
        # Generate a usable filename for this patch
        image_name = path_utils.clean_filename(image_fn_relative,force_lower=True)
        patch_fn_relative = usgs_geese_inference.patch_info_to_patch_name(
            os.path.splitext(image_name)[0],patch_xy[0],patch_xy[1]) + '.jpg'
        patch_fn_absolute = os.path.join(patch_folder,patch_fn_relative)
        
        im_results = relative_image_fn_to_results[image_fn_relative]
        
        pil_im = vis_utils.open_image(image_fn_absolute)
        assert im_results['w'] == pil_im.size[0], 'Image size error'
        assert im_results['h'] == pil_im.size[1], 'Image size error'
        
        # Extract the patch
        _ = usgs_geese_inference.extract_patch_from_image(pil_im,patch_xy,patch_size,
                                     patch_image_fn=patch_fn_absolute,
                                     patch_folder=None,image_name=None,overwrite=True)
        
        # Find detections that overlap with this patch, and convert to patch-relative,
        # normalized coordinates
        detections_this_image = im_results['detections_absolute']
        
        detections_this_patch = []
        
        # x/y/w/h
        r1 = [patch_xy[0],patch_xy[1],patch_size[0],patch_size[1]]
        
        # image_level_detection = detections_this_image[0]
        for image_level_detection in detections_this_image:
            
            r2 = image_level_detection['bbox']
            
            if intersects(r1,r2):
                
                patch_level_detection = {}
                patch_level_detection['conf'] = image_level_detection['conf']
                patch_level_detection['category'] = image_level_detection['category']
                
                # Convert to patch-relative absolute coordinates
                patch_bbox = [
                    image_level_detection['bbox'][0] - patch_xy[0],
                    image_level_detection['bbox'][1] - patch_xy[1],
                    image_level_detection['bbox'][2],
                    image_level_detection['bbox'][3]
                    ]
                    
                # Now normalize to patch-relative normalized coordinates
                patch_bbox[0] = patch_bbox[0]/patch_size[0]
                patch_bbox[1] = patch_bbox[1]/patch_size[1]
                patch_bbox[2] = patch_bbox[2]/patch_size[0]
                patch_bbox[3] = patch_bbox[3]/patch_size[1]
                
                patch_level_detection['bbox'] = patch_bbox
                                      
                detections_this_patch.append(patch_level_detection)
            
            # ...if this detection is within this patch
            
        # ...for every detection on this image
        
        # Add patch-specific results to patch_level_results['images']
        patch_im = {}
        patch_im['detections'] = detections_this_patch
        patch_im['file'] = patch_fn_relative
        patch_level_results['images'].append(patch_im)
        
    # ...for every sampled patch
    
    # Write out a .json file with the patch-level results
    patch_level_results_file = os.path.join(preview_folder,'patch_level_results.json')
    with open(patch_level_results_file,'w') as f:
        json.dump(patch_level_results,f,indent=1)
    
    # Generate the preview page
    preview_page_folder = os.path.join(preview_folder,'preview_page')
    os.makedirs(preview_page_folder,exist_ok=True)
    
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    base_task_name = os.path.basename(image_level_results_file)
        
    html_files = []
    
    for confidence_threshold in preview_confidence_thresholds:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder
        options.include_almost_detections = True
        options.num_images_to_sample = None # n_patches
        options.confidence_threshold = confidence_threshold
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        options.ground_truth_json_file = None
        options.separate_detections_by_category = True
        
        options.parallelize_rendering = False
        options.parallelize_rendering_n_cores = 16
        options.parallelize_rendering_with_threads = False
        
        output_base = os.path.join(preview_page_folder,
            base_task_name + '_{:.3f}'.format(options.confidence_threshold))
        
        os.makedirs(output_base, exist_ok=True)
        print('Processing to {}'.format(output_base))
        
        options.api_output_file = patch_level_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        # path_utils.open_file(html_output_file)
        html_files.append(html_output_file)
    
    # ...for each confidence threshold

    return html_files
    
# ...def patch_level_preview()


def image_level_counting(results_file,
                         output_file=None,
                         overwrite=False,
                         counting_confidence_thresholds=None):
    """
    Given an image-level results file:
        
    * Count the number of occurrences in each class above a few different thresholds
    * Write the resulting counts to .csv
    
    If output_file is None, writes the results to [results_file].csv
    """
    
    if counting_confidence_thresholds is None:
        counting_confidence_thresholds = default_counting_confidence_thresholds
    
    if output_file is None:
        output_file = results_file + '_counts.csv'

    if os.path.isfile(output_file) and not overwrite:
        print('Output file {} exists and overwrite=False, skiping'.format(output_file))
        return
        
    with open(results_file,'r') as f:
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
    # image_path_local
    # confidence_threshold
    # E.g.: 'count_brant', 'count_other', 'count_gull', 'count_canada', 'count_emperor'
    results = []
    
    # im = image_level_results['images'][0]
    for im in tqdm(image_level_results['images']):
        
        fn_relative = im['file']
        
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
            im_results['filename'] = fn_relative
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
    
    return output_file

# ...def image_level_counting(...)
