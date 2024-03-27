########
#
# run-izembek-model.py
#
# Run the Izembek brant model on an image or folder of images.  This script
# is a command-line driver for usgs-geese-inference.py
#
########

#%% Imports and constants

import os
import sys
import shutil
import importlib
import argparse

usgs_geese_inference = importlib.import_module('usgs-geese-inference')


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Script to run the Izembek brant model on an image or folder of images')
    parser.add_argument(
        'model_file',
        type=str,
        help='Path to detector model file (.pt)')
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to a folder of images')
    parser.add_argument(
        'yolo_working_dir',
        type=str,
        help='Path to the folder where the YOLO repo lives')
    parser.add_argument(
        'scratch_dir',
        type=str,
        help='Path to a folder where lots of temporary output will be stored')
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to output JSON results file, should end with a .json extension.  Always written to the scratch folder; this option just copies the final results file to a specified location.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--no_use_symlinks',
        action='store_true',
        help='Copy patches to the temporary inference folder, rather than using symlinks')
    parser.add_argument(
        '--no_cleanup',
        action='store_true',
        help='Bypass cleanup of temporary files')
    parser.add_argument(
        '--no_augment',
        action='store_true',
        help='Disable YOLOv5 test-time augmentation')
    parser.add_argument(
        '--category_mapping_file',
        type=str,
        default=None,
        help='.yaml or .json file mapping model category IDs to names (defaults to Izembek model mapping)')
    parser.add_argument(
        '--device',
        type=int,
        default=usgs_geese_inference.default_devices[0],
        help='CUDA device to run on, or -1 to force CPU inference (default {})'.format(
            usgs_geese_inference.default_devices[0]))
        
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    inference_options = usgs_geese_inference.USGSGeeseInferenceOptions(
        project_dir=args.scratch_dir,
        yolo_working_dir=args.yolo_working_dir,
        model_file=args.model_file,
        use_symlinks=(not args.no_use_symlinks),
        no_augment=(args.no_augment),
        no_cleanup=args.no_cleanup,
        devices=[args.device],
        category_mapping_file=args.category_mapping_file)

    results = usgs_geese_inference.run_model_on_folder(args.input_path,inference_options)
    
    if args.output_file is not None:
        nms_results_file = results['md_results_image_level_nms_fn']
        assert os.path.isfile(nms_results_file), \
            'Could not find output file {}'.format(nms_results_file)
        shutil.copyfile(nms_results_file,args.output_file)
        print('Copied results file to {}'.format(args.output_file))
        
if __name__ == '__main__':
    main()
    
        
#%% Interactive driver

if False:
    
    pass

    #%% Prepare inference pass
    
    image_dir = r'g:\temp\usgs-test-images'
    yolo_working_dir = r'c:\git\yolov5-current'
    scratch_dir = r'g:\temp\usgs-scratch'
    model_file = r"C:\Users\dmorr\models\usgs-geese\usgs-geese-yolov5x-230820-b8-img1280-e200-best.pt"
    use_symlinks = False
    cleanup = True
    
    inference_options = usgs_geese_inference.USGSGeeseInferenceOptions(
        project_dir=scratch_dir,
        yolo_working_dir=yolo_working_dir,
        model_file=model_file,
        use_symlinks=use_symlinks,
        no_cleanup=(not cleanup))
    # inference_options.n_cores_patch_generation = 1
    
    
    #%% Run in Python
    
    results = usgs_geese_inference.run_model_on_folder(image_dir,inference_options)
    
    
    #%% Generate command
    
    script_name = 'run-izembek-model.py'
    output_file = r'g:\temp\wdfw-results-nms.json'
        
    import clipboard  
    cmd = f'python "{script_name}" "{model_file}"  "{image_dir}" "{yolo_working_dir}" ' + \
          r'"g:\temp\usgs-geese-inference-test" --recursive'
    if not cleanup:
        cmd += ' --no_cleanup'
    if not use_symlinks:
        cmd += ' --no_use_symlinks'
    print(cmd)
    clipboard.copy(cmd)
