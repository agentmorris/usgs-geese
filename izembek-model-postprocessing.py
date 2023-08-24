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

from md_utils.path_utils import open_file

usgs_geese_postprocessing = importlib.import_module('usgs-geese-postprocessing')

default_preview_confidence_thresholds = [0.6]
default_n_patches = 2500


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Script to run postprocessing steps that typically follow inference with the Izembek goose model (counting, preview page generation).  At least one output (count file, preview folder) must be specified.')
    parser.add_argument(
        'results_file',
        type=str,        
        help='Path to .json file containing inference results')
    parser.add_argument(
        '--image_folder',
        type=str,
        help='Path to the folder of images on which inference was run (not required for counting)')
    parser.add_argument(
        '--count_file',
        type=str,
        default=None,
        help='Path to a .csv file where we should write per-image counts')
    parser.add_argument(
        '--preview_folder',
        type=str,
        default=None,
        help='Path to a folder in which to put patch-level previews')
    parser.add_argument(
        '--n_patches',
        type=int,
        default=default_n_patches,
        help='Number of patches to use for preview pages (default {})'.format(
            default_n_patches))    
    parser.add_argument(
        '--confidence_thresholds',
        nargs="*",
        type=float,
        default=str(default_preview_confidence_thresholds[0]),
        help='Space-separated list of confidence thresholds for preview and count generation (default {})'.format(
            str(default_preview_confidence_thresholds[0])))
    parser.add_argument(
        '--open_preview_pages',
        action='store_true',
        help='Open HTML preview pages after completion')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    
    assert os.path.isfile(args.results_file), \
        'Could not find results file {}'.format(args.results_file)
        
    if args.open_preview_pages and (args.preview_folder is None):
        print('Warning: open_preview_pages specified, but previews not requested')
        
    if args.preview_folder is not None:

        assert args.image_folder is not None, \
            'Previews requested, but image folder not specified'
            
        assert os.path.isdir(args.image_folder), \
            'Could not find image folder {}'.format(args.image_folder)

        preview_html_files = usgs_geese_postprocessing.patch_level_preview(args.results_file,
                                               args.image_folder,
                                               args.preview_folder,
                                               n_patches=args.n_patches,
                                               preview_confidence_thresholds=args.confidence_thresholds)    
        
        if args.open_preview_pages:           
            for fn in preview_html_files:
                open_file(fn)
    
    # ...if we're generating previews
    
    if args.count_file is not None:
        
        output_file = usgs_geese_postprocessing.image_level_counting(args.results_file,                             
                             output_file=args.count_file,
                             overwrite=True,
                             counting_confidence_thresholds=args.confidence_thresholds)    
        print('\nWrote counting results to {}'.format(output_file))
    
# ...main()


if __name__ == '__main__':
    main()
    
        
#%% Interactive driver

if False:
    
    pass

    #%%
    
    n_patches = 2000
    confidence_thresholds = [0.2,0.3,0.4,0.5,0.6]
    results_file = r"g:\temp\usgs-geese-inference-test\image_level_results\_temp_wdfw-brant-images_md_results_image_level_nms.json"
    image_folder = r"g:\temp\wdfw-brant-images"
    preview_folder = r"g:\temp\wdfw-test-preview-with-aug"
    script_name = 'izembek-model-postprocessing.py'
    
    threshold_string = ' '.join([str(t) for t in confidence_thresholds])
    
    s = f'python {script_name} "{results_file}" --image_folder "{image_folder}" --preview_folder "{preview_folder}"' + \
                   f' --confidence_thresholds {threshold_string} --open_preview_pages --n_patches {n_patches}'
    print(s)
    import clipboard; clipboard.copy(s)
    
    #%%
    
    n_patches = 2000
    confidence_thresholds = [0.2,0.3,0.4,0.5,0.6]
    results_file = r"g:\temp\usgs-geese-inference-test-noaug\image_level_results\_temp_wdfw-brant-images_md_results_image_level_nms.json"
    image_folder = r"g:\temp\wdfw-brant-images"
    preview_folder = r"g:\temp\wdfw-test-preview-no-aug"
    script_name = 'izembek-model-postprocessing.py'
    
    threshold_string = ' '.join([str(t) for t in confidence_thresholds])
    
    s = f'python {script_name} "{results_file}" --image_folder "{image_folder}" --preview_folder "{preview_folder}"' + \
                   f' --confidence_thresholds {threshold_string} --open_preview_pages --n_patches {n_patches}'
    print(s)
    import clipboard; clipboard.copy(s)
    
    #%%
    
    confidence_thresholds = [0.2,0.3,0.4,0.5,0.6]
    results_file = r"g:\temp\usgs-geese-inference-test\image_level_results\_temp_wdfw-brant-images_md_results_image_level_nms.json"
    output_file = results_file + '.csv'
    script_name = 'izembek-model-postprocessing.py'
    
    threshold_string = ' '.join([str(t) for t in confidence_thresholds])
    
    s = f'python {script_name} "{results_file}" --count_file "{output_file}"' + \
                   f' --confidence_thresholds {threshold_string}'
    print(s)
    import clipboard; clipboard.copy(s)
    
    