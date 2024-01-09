########
#
# usgs-geese-training-yolov8.py
#
# This file documents the model training process, starting from where usgs-geese-training-data-prep.py
# leaves off.  Training happens at the yolov8 CLI, and the exact command line arguments are documented
# in the "Train" cell.
#
# Later cells in this file also:
#
# * Run the YOLO validation scripts
# * Convert YOLO val results to MD .json format
# * Use the MD visualization pipeline to visualize results
# * Use the MD inference pipeline to run the trained model
#
########

#%% Environment prep (yolov8)

"""
mamba create --name yolov8 pip python==3.11 -y
mamba activate yolov8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
"""

#%% Project constants

import os

# When training on multiple GPUs, batch=-1 is ignored, and the default batch size (16) is used
batch_size = -1
image_size = 640
epochs = 300
# yolo_dataset_file='/home/user/data/usgs-geese-640/dataset.yaml'
yolo_dataset_file = '/home/user/data/usgs-geese-640px-320stride/dataset.yaml'
base_model = 'yolov8x.pt'
tag = '-stride-320'
project_dir = os.path.expanduser('~/tmp/usgs-geese-yolov8-training')
assert not project_dir.endswith('/')

training_run_name = f'usgs-geese-yolov8x-2023.12.31-b{batch_size}-img{image_size}-e{epochs}{tag}'

model_base_folder = os.path.expanduser('~/models/usgs-geese')
assert os.path.isdir(model_base_folder)


#%% Train

"""
mkdir -p ~/tmp/usgs-geese/yolov8-training
cd ~/tmp/usgs-geese/yolov8-training
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov8
"""

training_command = f'yolo detect train data="{yolo_dataset_file}" batch={batch_size} model="{base_model}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device="0,1"'
print('\n{}'.format(training_command))
# import clipboard; clipboard.copy(training_command)


#%% Resume training

import os
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')
assert os.path.isfile(resume_checkpoint)

"""
mamba activate yolov8
"""
resume_command = f'yolo detect train resume model="{resume_checkpoint}" data="{yolo_dataset_file}"'
print('\n{}'.format(resume_command))
import clipboard; clipboard.copy(resume_command)


#%% Make plots during training

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.figure
from md_utils.path_utils import open_file

assert 'yolov5' in training_run_name or 'yolov8' in training_run_name
if 'yolov5' in training_run_name:
    model_type = 'yolov5'
else:
    model_type = 'yolov8'

results_file = '{}/{}/results.csv'.format(project_dir,training_run_name)
assert os.path.isfile(results_file)

results_page_folder = '{}/{}/training-progress-report'.format(project_dir,training_run_name)
os.makedirs(results_page_folder,exist_ok=True)

fig_00_fn_abs = os.path.join(results_page_folder,'figure_00.png')
fig_01_fn_abs = os.path.join(results_page_folder,'figure_01.png')
fig_02_fn_abs = os.path.join(results_page_folder,'figure_02.png')

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())

plt.ioff()

fig_w = 12
fig_h = 8

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 
else:
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/dfl_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/dfl_loss', ax = ax, secondary_y = True) 

fig.savefig(fig_00_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

fig.savefig(fig_01_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'metrics/precision', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5:0.95', ax = ax)
else:
    df.plot(x = 'epoch', y = 'metrics/precision(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50-95(B)', ax = ax)

fig.savefig(fig_02_fn_abs,dpi=100)
plt.close(fig)

results_page_html_file = os.path.join(results_page_folder,'index.html')
with open(results_page_html_file,'w') as f:
    f.write('<html><body>\n')
    f.write('<img src="figure_00.png"><br/>\n')
    f.write('<img src="figure_01.png"><br/>\n')
    f.write('<img src="figure_02.png"><br/>\n')    
    f.write('</body></html>\n')

open_file(results_page_html_file)
# import clipboard; clipboard.copy(results_page_html_file)


#%% Back up models after (or during) training, removing optimizer state if appropriate

import shutil

checkpoint_tag = '20240108-resume'

# Import the function we need for removing optimizer state

utils_imported = False
if not utils_imported:
    try:
        from yolov5.utils.general import strip_optimizer
        utils_imported = True
    except Exception:
        pass
if not utils_imported:
    try:
        from ultralytics.utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass        
if not utils_imported:
    try:
        from ultralytics.utils.torch_utils import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass        
if not utils_imported:
    try:
        from utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass
assert utils_imported

# Input folder(s)
training_output_dir = os.path.join(project_dir,training_run_name)
training_weights_dir = os.path.join(training_output_dir,'weights')
assert os.path.isdir(training_weights_dir)

# Output folder
model_folder = '{}/{}'.format(model_base_folder,training_run_name)
model_folder = os.path.join(model_folder,checkpoint_tag)
os.makedirs(model_folder,exist_ok=True)
weights_folder = os.path.join(model_folder,'weights')
os.makedirs(weights_folder,exist_ok=True)

for weight_name in ('last','best'):
    
    source_file = os.path.join(training_weights_dir,weight_name + '.pt')
    assert os.path.isfile(source_file)
    target_file = os.path.join(weights_folder,'{}-{}.pt'.format(
        training_run_name,weight_name))
    
    shutil.copyfile(source_file,target_file)
    target_file_optimizer_stripped = target_file.replace('.pt','-stripped.pt')
    strip_optimizer(target_file,target_file_optimizer_stripped)

other_files = os.listdir(training_output_dir)
other_files = [os.path.join(training_output_dir,fn) for fn in other_files]
other_files = [fn for fn in other_files if os.path.isfile(fn)]

# source_file_abs = other_files[0]
for source_file_abs in other_files:
    assert not source_file_abs.endswith('.pt')
    fn_relative = os.path.basename(source_file_abs)
    target_file_abs = os.path.join(model_folder,fn_relative)
    shutil.copyfile(source_file_abs,target_file_abs)

print('Backed up training state to {}'.format(model_folder))
# import clipboard; clipboard.copy(model_folder)


#%% Validation (with YOLO CLI)

import os

model_base = os.path.expanduser('~/models/usgs-geese')
training_run_names = [
    'usgs-geese-yolov8x-2023.12.31-b-1-img640-e3004'
]

data_folder = os.path.expanduser('~/data/usgs-geese-640')
image_size = 640

# Doesn't impact results, just inference time
batch_size_val = 8

project_name = os.path.expanduser('~/tmp/usgs-geese-640-val')
data_file = os.path.join(data_folder,'dataset.yaml')
augment_flags = [True,False]

assert os.path.isfile(data_file)

commands = []

n_devices = 2

# training_run_name = training_run_names[0]
for training_run_name in training_run_names:
    
    for augment in augment_flags:
        
        model_file_base = os.path.join(model_base,training_run_name)
        model_files = [model_file_base + s for s in ('-last.pt','-best.pt')]
        
        # model_file = model_files[0]
        for model_file in model_files:
            
            assert os.path.isfile(model_file)
            
            model_short_name = os.path.basename(model_file).replace('.pt','')
            
            # yolo detect train data=${DATA_YAML_FILE} batch=${BATCH_SIZE} model=${BASE_MODEL} epochs=${EPOCHS} imgsz=${IMAGE_SIZE} project=${PROJECT} name=${NAME} device=0,1
            cuda_index = len(commands) % n_devices
            cuda_string = 'CUDA_VISIBLE_DEVICES={}'.format(cuda_index)
            cmd = cuda_string + \
                ' yolo detect val imgsz={} batch={} model="{}" project="{}" name="{}" data="{}" save_json'.format(
                image_size,batch_size_val,model_file,project_name,model_short_name,data_file)        
            if augment:
                cmd += ' augment'
            commands.append(cmd)

        # ...for each model
    
    # ...augment on/off        
    
# ...for each training run    

for cmd in commands:
    print('')
    print(cmd + '\n')
    
pass


#%% Results notes: no tile overlap during training

# Training stopped early at 106 epochs; best result observed @ epoch 56

# Results printed at the end of training (should be same as "best no aug" below)

"""
                 Class     Images  Instances      P          R          mAP50      mAP50-95
                   all      20644      92979      0.784      0.745      0.778      0.517
                 Brant      20644      75016      0.953        0.9      0.925      0.625
                 Other      20644       6815      0.803       0.71      0.766      0.494
                  Gull      20644       1088        0.9      0.856      0.904      0.583
                Canada      20644       9726      0.922      0.856        0.9      0.628
               Emperor      20644        334      0.342      0.401      0.398      0.257
"""

"""
Last w/aug
"""

""" 
                 Class     Images  Instances      P          R          mAP50      mAP50-95
                  all      20644      92979       0.75       0.761      0.772      0.486
                 Brant      20644      75016      0.918      0.886      0.912      0.569
                 Other      20644       6815      0.755      0.753      0.767      0.477
                  Gull      20644       1088      0.856      0.877      0.875      0.566
                Canada      20644       9726      0.909      0.864      0.898       0.58
               Emperor      20644        334      0.314      0.425      0.411      0.238
"""

"""
Best w/aug
"""

"""
                 Class     Images  Instances      P          R          mAP50      mAP50-95
                   all      20644      92979       0.77      0.752      0.776      0.529
                 Brant      20644      75016      0.944      0.895      0.922      0.636
                 Other      20644       6815      0.784      0.754      0.779       0.51
                  Gull      20644       1088      0.867       0.86      0.879      0.604
                Canada      20644       9726      0.914       0.86        0.9      0.631
               Emperor      20644        334      0.342      0.392      0.401      0.263
"""

"""
Last no aug
"""

"""
                 Class     Images  Instances      P          R          mAP50      mAP50-95
                   all      20644      92979      0.785       0.75      0.782      0.442
                 Brant      20644      75016      0.954      0.896      0.924      0.525
                 Other      20644       6815      0.801      0.699      0.754      0.421
                  Gull      20644       1088      0.924      0.875      0.912      0.501
                Canada      20644       9726      0.922      0.864      0.903      0.538
               Emperor      20644        334      0.322      0.416      0.416      0.224
"""

"""
Best no aug
"""

"""
                 Class     Images  Instances      P          R          mAP50      mAP50-95
                   all      20644      92979      0.784      0.745      0.778      0.517
                 Brant      20644      75016      0.953        0.9      0.925      0.625
                 Other      20644       6815      0.802       0.71      0.766      0.494
                  Gull      20644       1088        0.9      0.856      0.904      0.584
                Canada      20644       9726      0.922      0.856        0.9      0.628
               Emperor      20644        334      0.342      0.401      0.398      0.255
"""