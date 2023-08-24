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
        help='Path to output JSON results file, should end with a .json extension')
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
        devices=[args.device])

    usgs_geese_inference.run_model_on_folder(args.input_path,inference_options)

if __name__ == '__main__':
    main()
    
        
#%% Interactive driver

if False:
    
    pass

    #%%
    
    project_dir = r'g:\temp\usgs-geese-inference-test'
    yolo_working_dir = r'c:\git\yolov5-current'
    model_file = os.path.expanduser('~/models/usgs-geese/usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt')
    
    inference_options = usgs_geese_inference.USGSGeeseInferenceOptions(
        project_dir=project_dir,
        yolo_working_dir=yolo_working_dir,
        model_file=model_file)

    image_dir = r'g:\temp\wdfw-brant-images'
    
    usgs_geese_inference.run_model_on_folder(image_dir,inference_options)
    
    #%%
    
    model_file = r"c:\users\dmorr\models\usgs-geese\usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt"
    script_name = 'run-izembek-model.py'
    yolo_folder = "c:\git\yolov5-current"
    
    import clipboard
    
    clipboard.copy(f'python "{model_file}"  "g:\temp\wdfw-brant-images" "{yolo_folder}" ' + \
          r'"g:\temp\usgs-geese-inference-test" --recursive --no_cleanup')
