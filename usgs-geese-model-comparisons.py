########
#
# usgs-geese-model-comparisons.py
#
# After re-training in 2023.09, we used this script to compare three version of
# the model (one prior to the training data bug fix, and two checkpoints from after
# that fix).
#
########

#%% Imports and constants

import os

default_yolo_working_dir = os.path.expanduser('~/git/yolov5-current')


#%% Models and folders we're going to operate over

input_folders = [
    '/home/user/data/usgs-geese/eval_images',
    '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-03'
    ]

model_base = os.path.expanduser('~/models/usgs-geese')
model_files = [
    'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt',
    'usgs-geese-yolov5x-230820-b8-img1280-2023.09.02-best.pt',
    'usgs-geese-yolov5x-230820-b8-img1280-e200-best.pt'
    ]
model_files = [os.path.join(model_base,fn) for fn in model_files]


#%% Inference

commands = []

for model_file in model_files:
    
    for input_folder in input_folders:
        
        folder_name = input_folder.split('/')[-1]
        model_name = model_file.split('/')[-1].replace('.pt','')            
        device_index = len(commands) % 2
        
        inference_run_name = '{}_{}'.format(model_name,folder_name)
        scratch_dir = os.path.expanduser('~/tmp/usgs-inference/{}'.format(inference_run_name))
        output_file = os.path.expanduser('~/tmp/usgs-inference/{}.json'.format(inference_run_name))
        
        cmd = 'python run-izembek-model.py'
        cmd += ' "{}"'.format(model_file)
        cmd += ' "{}"'.format(input_folder)
        cmd += ' "{}"'.format(default_yolo_working_dir)
        cmd += ' "{}"'.format(scratch_dir)
        cmd += ' --output_file "{}"'.format(output_file)
        cmd += ' --recursive'
        cmd += ' --device {}'.format(str(device_index))
        commands.append(cmd)
        
for s in commands:
    print(s + '\n')
        

#%% Previews

commands = []
thresholds = [0.4, 0.5, 0.6]
confidence_threshold_string = ' '.join([str(t) for t in thresholds])
n_patches_per_preview = 1000
preview_folder_base = os.path.expanduser('~/tmp/usgs-inference/previews')

for model_file in model_files:
    
    for input_folder in input_folders:
        
        folder_name = input_folder.split('/')[-1]
        model_name = model_file.split('/')[-1].replace('.pt','')            
        device_index = len(commands) % 2
        
        inference_run_name = '{}_{}'.format(model_name,folder_name)
        results_file = os.path.expanduser('~/tmp/usgs-inference/{}.json'.format(inference_run_name))
        preview_folder = os.path.join(preview_folder_base,inference_run_name)
        
        cmd = 'python izembek-model-postprocessing.py'
        cmd += ' "{}"'.format(results_file)
        cmd += ' --image_folder "{}"'.format(input_folder)
        cmd += ' --preview_folder "{}"'.format(preview_folder)
        cmd += ' --n_patches "{}"'.format(str(n_patches_per_preview))
        cmd += ' --confidence_thresholds {}'.format(confidence_threshold_string)
        cmd += ' --open_preview_pages'
        
        commands.append(cmd)
        
for s in commands:
    print(s + '\n')

