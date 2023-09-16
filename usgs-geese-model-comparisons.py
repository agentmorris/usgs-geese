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
import json

from tqdm import tqdm

default_yolo_working_dir = os.path.expanduser('~/git/yolov5-current')


#%% Models and folders we're going to operate over

input_folders = [
    # '/home/user/data/usgs-geese/eval_images',
    # '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-03',
    '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-03/Cam1',
    '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-03/Cam2'
    ]

model_base = os.path.expanduser('~/models/usgs-geese')
model_files = [
    # 'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt',
    # 'usgs-geese-yolov5x-230820-b8-img1280-2023.09.02-best.pt',
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

# import clipboard; clipboard.copy(commands[1])


#%% Merge two results files that I want to treat as one

results_files = [
    '/home/user/tmp/usgs-inference/usgs-geese-yolov5x-230820-b8-img1280-e200-best_Cam1.json',
    '/home/user/tmp/usgs-inference/usgs-geese-yolov5x-230820-b8-img1280-e200-best_Cam2.json'
]

merged_file = \
    '/home/user/tmp/usgs-inference/usgs-geese-yolov5x-230820-b8-img1280-e200-best_Replicate_2017-10-03.json'
    
images = []
info = None
detection_categories = None

# results_file = results_files[0]
for results_file in results_files:

    print('Adjusting paths in {}'.format(results_file))
    
    with open(results_file,'r') as f:
        d = json.load(f)
        info = d['info']
        detection_categories = d['detection_categories']
        
    # im = d['images'][0]
    for im in tqdm(d['images']):
        if 'Cam1' in results_file:
            assert 'CAM1' in im['file']
            im['file'] = 'Cam1/' + im['file']
        else:
            assert 'Cam2' in results_file
            assert 'CAM2' in im['file']
            im['file'] = 'Cam2/' + im['file']
        images.append(im)

print('Writing merged output')
        
with open(merged_file,'w') as f:
    
    output_data = {'images':images,'info':info,'detection_categories':detection_categories}
    json.dump(output_data,f,indent=1)
        

#%% Convert everything to drive-relative paths

drive_root = '/media/user/My Passport'

def convert_results_file_to_drive_relative(input_file,drive_root,inference_folder_drive_relative,
                                           output_file=None,check_file_existence=True):

    if output_file is None:
        output_file = input_file.replace('.json','_drive_relative.json')
    assert output_file != input_file
    assert os.path.isdir(drive_root)
    if inference_folder_drive_relative != 'eval':
        assert os.path.isdir(os.path.join(drive_root,inference_folder_drive_relative))
        
    with open(input_file,'r') as f:
        d = json.load(f)
    
    print('Converting results in {} to drive-relative paths'.format(input_file))
    
    # im = d['images'][0]        
    for im in tqdm(d['images']):
        
        input_fn = im['file']
        
        if inference_folder_drive_relative == 'eval':
            # For the eval set, filenames look like
            # 'val-images/2019_Replicate_2019-10-11_Cam3_CAM39080.JPG'            
            prefix = '2017-2019/01_JPGs/'
            fn = os.path.basename(input_fn)
            fn = fn.replace('Replicate_','Replicate*').replace('Out_lagoon','Out*lagoon')
            fn = fn.replace('_','/')
            fn = fn.replace('*','_')
            image_path_drive_relative = os.path.join(prefix,fn)
        else:
            image_path_drive_relative = os.path.join(inference_folder_drive_relative,input_fn)
        
        image_path_abs = os.path.join(drive_root,image_path_drive_relative)
        assert os.path.isfile(image_path_abs)
        
        im['file'] = image_path_drive_relative
        
    # ...for each image
    
    with open(output_file,'w') as f:
        json.dump(d,f,indent=1)

# ...convert_results_file_to_drive_relative()

results_dir = os.path.expanduser('~/tmp/usgs-inference')    
results_files = os.listdir(results_dir)
results_files = [os.path.join(results_dir,fn) for fn in results_files if fn.endswith('.json')]

for results_file in results_files:
    if 'eval' in results_file:
        inference_folder_drive_relative = 'eval'
    else:
        inference_folder_drive_relative = '2017-2019/01_JPGs/2017/Replicate_2017-10-03'
    convert_results_file_to_drive_relative(results_file,drive_root,inference_folder_drive_relative,
                                           output_file=None,check_file_existence=True)


#%% Previews

results_dir = os.path.expanduser('~/tmp/usgs-inference')    
results_files = os.listdir(results_dir)
results_files = [os.path.join(results_dir,fn) for fn in results_files if \
                 (fn.endswith('.json') and 'drive_relative' in fn)]

commands = []
thresholds = [0.4, 0.5, 0.6]
confidence_threshold_string = ' '.join([str(t) for t in thresholds])
n_patches_per_preview = 2500
preview_folder_base = os.path.expanduser('~/tmp/usgs-inference/previews')

drive_root = '/media/user/My Passport'

for results_file in results_files:
        
    if 'eval' in results_file:
        folder_name = 'eval'        
    else:
        folder_name = 'replicate_2017-10-03'
        
    model_name = results_file.split('/')[-1].split('_')[0]
    assert '/' not in model_name
    assert '_' not in model_name
    
    inference_run_name = '{}_{}'.format(model_name,folder_name)
    preview_folder = os.path.join(preview_folder_base,inference_run_name)
    
    cmd = 'python izembek-model-postprocessing.py'
    cmd += ' "{}"'.format(results_file)
    cmd += ' --image_folder "{}"'.format(drive_root)
    cmd += ' --preview_folder "{}"'.format(preview_folder)
    cmd += ' --n_patches {}'.format(str(n_patches_per_preview))
    cmd += ' --confidence_thresholds {}'.format(confidence_threshold_string)
        
    commands.append(cmd)
        
for s in commands:
    print(s + '\n')

# import clipboard; clipboard.copy(commands[0])
# import clipboard; clipboard.copy('\n\n'.join(commands))


#%% Counts

results_dir = os.path.expanduser('~/tmp/usgs-inference')    
results_files = os.listdir(results_dir)
results_files = [os.path.join(results_dir,fn) for fn in results_files if \
                 (fn.endswith('.json') and 'drive_relative' in fn)]

commands = []
thresholds = [0.4, 0.5, 0.6]
confidence_threshold_string = ' '.join([str(t) for t in thresholds])
count_folder_base = os.path.expanduser('~/tmp/usgs-inference/counts')
os.makedirs(count_folder_base,exist_ok=True)

for results_file in results_files:
        
    if 'eval' in results_file:
        folder_name = 'eval'        
    else:
        folder_name = 'replicate_2017-10-03'
        
    model_name = results_file.split('/')[-1].split('_')[0]
    assert '/' not in model_name
    assert '_' not in model_name
    
    inference_run_name = '{}_{}'.format(model_name,folder_name)
    count_file = os.path.join(count_folder_base,inference_run_name) + '.csv'
    
    cmd = 'python izembek-model-postprocessing.py'
    cmd += ' "{}"'.format(results_file)
    cmd += ' --count_file {}'.format(count_file)
    cmd += ' --confidence_thresholds {}'.format(confidence_threshold_string)    
        
    commands.append(cmd)
        
for s in commands:
    print(s + '\n')

# import clipboard; clipboard.copy(commands[0])
# import clipboard; clipboard.copy('\n\n'.join(commands))
