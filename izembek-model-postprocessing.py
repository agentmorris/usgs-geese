########
#
# izembek-model-postprocessing.py
#
# Do stuff that one does after running the Izembek goose detector, in particular:
#
# * Generating preview pages
# * Generating per-image bird counts
#
########

#%% Imports and constants

import os
import sys
import importlib
import argparse

usgs_geese_postprocessing = importlib.import_module('usgs-geese-postprocessing')

default_preview_confidence_thresholds = [0.6]
default_n_patches = 2500


#%% Main function

def postprocess_model_results(results_file,image_folder,count_file=None,preview_folder=None,
                              confidence_threshold=default_preview_confidence_thresholds):
    
    assert count_file is not None or preview_folder is not None, \
        'Must specifiy a preview folder output and/or a count file output'
    
    

#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Script to run postprocessing steps that typically follow inference with the Izembek goose model (counting, preview page generation).  At least one output (count file, preview folder) must be specified.')
    parser.add_argument(
        'results_file',
        help='Path to .json file containing inference results')
    parser.add_argument(
        'image_folder',
        help='Path to the folder of images on which inference was run')
    parser.add_argument(
        '--count_file',
        help='Path to a .csv file where we should write per-image counts')
    parser.add_argument(
        '--preview_folder',
        help='Path to a folder in which to put patch-level previews')
    parser.add_argument(
        '--n_patches',
        type=int,
        default=default_n_patches,
        help='Number of patches to use for preview pages (default {})'.format(
            default_n_patches))    
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=default_preview_confidence_thresholds[0],
        help='Confidence threshold for preview and count generation (default {})'.format(
            default_preview_confidence_thresholds[0]))
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    

if __name__ == '__main__':
    main()
    
        
#%% Interactive driver

if False:
    
    pass

    #%% 
    
    import sys
    p = r'c:\git\usgs-geese'
    if p not in sys.path:
        sys.path.append(p)
        
    #%% 

    n_patches = 100
    confidence_thresholds = default_preview_confidence_thresholds
    results_file = r"g:\temp\usgs-geese-inference-test\image_level_results\_temp_wdfw-test_md_results_image_level_nms.json"
    image_folder = r"g:\temp\wdfw-test"
    preview_folder = r"g:\temp\wdfw-test-preview"
    
    image_html_files = usgs_geese_postprocessing.patch_level_preview(results_file,
                                           image_folder,
                                           preview_folder,
                                           n_patches=n_patches,
                                           preview_confidence_thresholds=confidence_thresholds)    
