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
pip install --upgrade ultralytics
mamba install -c conda-forge spyder
pip install clipboard
"""

#%% Project constants

import os

# When training on multiple GPUs, batch=-1 is ignored, and the default batch size (16) is used
batch_size = -1
image_size = 640
epochs = 300

# yolo_dataset_file = os.path.expanduser('~/data/usgs-geese-640/dataset.yaml')

# yolo_dataset_file = os.path.expanduser('~/data/usgs-geese-640px-320stride/dataset.yaml')
# project_dir = os.path.expanduser('~/tmp/usgs-geese-yolov8-training')

yolo_dataset_file = '/home/dmorris/data-wsl/usgs-geese-640px-320stride/dataset.yaml'
project_dir = '/home/dmorris/tmp/usgs-geese-yolov8-training'

# base_model = 'yolov8x.pt'
base_model = 'yolov8l.pt'

tag = '-stride-320'
assert not project_dir.endswith('/')

# Enable YOLOv8's RAM cache.  Data seems to expand ~4x into RAM, so with this dataset, you would need
# a couple hundred GB of RAM to support this.
enable_ram_cache = False

# I found that with a batch size of 32 (instead of 16), AMP caused instability.
enable_amp = True

# training_run_name = f'usgs-geese-yolov8x-2024.01.10-b{batch_size}-img{image_size}-e{epochs}{tag}'
# training_run_name = f'usgs-geese-yolov8x-2024.01.10-b{batch_size}-img{image_size}-e{epochs}{tag}'
training_run_name = f'usgs-geese-yolov8l-2024.01.17-b{batch_size}-img{image_size}-e{epochs}{tag}'

model_base_folder = os.path.expanduser('~/models/usgs-geese')
assert os.path.isdir(model_base_folder)

amp_string = "" if enable_amp else "amp=False"
ram_cache_string = "cache" if enable_ram_cache else ""

def wsl_project_path_to_windows(s):
    if os.name == 'nt' and project_dir.startswith('/'):
        s = s.replace('/home/dmorris',os.path.expanduser('~'))
        s = s.replace('data-wsl','data')
    return s


#%% Train

"""
mkdir -p ~/tmp/usgs-geese/yolov8-training
cd ~/tmp/usgs-geese/yolov8-training
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov8
"""

training_command = f'yolo detect train data="{yolo_dataset_file}" batch={batch_size} model="{base_model}" epochs={epochs} imgsz={image_size} ' + \
                   f'project="{project_dir}" name="{training_run_name}" device="0,1" {ram_cache_string} {amp_string}'
                   
print('\n{}'.format(training_command))
import clipboard; clipboard.copy(training_command)


#%% Resume training

import os
# resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')
resume_checkpoint = project_dir + '/' + training_run_name + '/weights/last.pt'
# assert os.path.isfile(resume_checkpoint)

"""
mamba activate yolov8
"""

cmd = 'if [ -f  {} ]; then\necho "Checkpoint found"\nelse\necho "Checkpoint not found"\nfi'.format(yolo_dataset_file)
import clipboard; clipboard.copy(cmd)

resume_command = f'yolo detect train resume model="{resume_checkpoint}" data="{yolo_dataset_file}" ' + \
                 f'{ram_cache_string} {amp_string}' 
#  project="{project_dir}" name="{training_run_name}"                 
                 
print('\n{}'.format(resume_command))
import clipboard; clipboard.copy(resume_command)


#%% Make plots during training

import numpy as np
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
results_file = wsl_project_path_to_windows(results_file)
assert os.path.isfile(results_file)

results_page_folder = '{}/{}/training-progress-report'.format(project_dir,training_run_name)
os.makedirs(results_page_folder,exist_ok=True)

fig_00_fn_abs = os.path.join(results_page_folder,'figure_00.png')
fig_01_fn_abs = os.path.join(results_page_folder,'figure_01.png')
fig_02_fn_abs = os.path.join(results_page_folder,'figure_02.png')

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())
# df = df.replace([np.inf, -np.inf], 1.0)

# I have seen inf values creep in for the cls_loss; pandas is loading "inf" as a string, so I
# plaster over this by converting to float here.  I'm not sure why it's only val/cls_loss.
df['val/cls_loss'] = df['val/cls_loss'].map(np.float32)

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

checkpoint_tag = '20240124-final'

# Import the function we need for removing optimizer state

strip_optimizer_state = False

if strip_optimizer_state:
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
training_output_dir = wsl_project_path_to_windows(training_output_dir)
training_weights_dir = os.path.join(training_output_dir,'weights')
assert os.path.isdir(training_weights_dir)

# Output folder
model_folder = '{}/{}'.format(model_base_folder,training_run_name)
model_folder = os.path.join(model_folder,checkpoint_tag)
os.makedirs(model_folder,exist_ok=True)
weights_folder = os.path.join(model_folder,'weights')
os.makedirs(weights_folder,exist_ok=True)

# weight_name = 'best'

for weight_name in ('last','best'):
    
    source_file = os.path.join(training_weights_dir,weight_name + '.pt')
    assert os.path.isfile(source_file)
    target_file = os.path.join(weights_folder,'{}-{}.pt'.format(
        training_run_name,weight_name))
    shutil.copyfile(source_file,target_file)
    
    if strip_optimizer_state:
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

# model_base = os.path.expanduser('~/models/usgs-geese')
model_base = '/home/dmorris/models/usgs-geese'
# project_name = os.path.expanduser('~/tmp/usgs-geese-640-val')
project_name = '/home/dmorris/tmp/usgs-geese-640-val'


training_run_names = [
    # 'usgs-geese-yolov8x-2023.12.31-b-1-img640-e3004'
    # 'usgs-geese-yolov8x-2024.01.10-b-1-img640-e300-stride-320'
    'usgs-geese-yolov8l-2024.01.17-b-1-img640-e300-stride-320'
]

# data_folder = os.path.expanduser('~/data/usgs-geese-640')
data_folder = os.path.dirname(yolo_dataset_file)
image_size = 640

# Doesn't impact results, just inference time
batch_size_val = 8

data_file = data_folder + '/dataset.yaml'
augment_flags = [True,False]

commands = []

n_devices = 2

# training_run_name = training_run_names[0]
for training_run_name in training_run_names:
    
    # augment = augment_flags[0]
    for augment in augment_flags:
        
        model_file_base = model_base + '/' + training_run_name
        model_files = os.listdir(wsl_project_path_to_windows(model_file_base))
        model_files = [model_file_base + '/' + fn for fn in model_files if fn.endswith('.pt')]
        assert len(model_files) == 2
        assert len([s for s in model_files if s.endswith('-best.pt')]) == 1
        assert len([s for s in model_files if s.endswith('-last.pt')]) == 1
        
        # model_file = model_files[0]
        for model_file in model_files:
            
            model_short_name = os.path.basename(model_file).replace('.pt','')
            if augment:
                model_short_name += '-aug'
            
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


#%% Result notes: 50% overlap during training

# Training stopped early at 90 epochs; best result observed @ epoch 61

# Results printed at the end of training (should be same as "best no aug" below)

"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      72237     327004      0.792      0.756      0.788      0.557
                 Brant      72237     264066       0.96      0.901      0.927       0.67
                 Other      72237      24536      0.815      0.727      0.785      0.529
                  Gull      72237       3853      0.926      0.872      0.918      0.651
                Canada      72237      33340      0.937      0.858      0.905      0.659
               Emperor      72237       1209      0.322      0.423      0.405      0.278
"""


"""
Last w/aug
"""

"""
                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      72237     327004      0.771      0.761      0.783      0.553
                 Brant      72237     264066      0.954      0.892      0.926      0.668
                 Other      72237      24536      0.785      0.775        0.8      0.537
                  Gull      72237       3853      0.892      0.883        0.9      0.641
                Canada      72237      33340      0.926      0.862      0.905      0.656
               Emperor      72237       1209        0.3      0.394      0.384      0.262
"""

"""
Best w/aug
"""

"""
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all      72237     327004      0.776      0.763      0.782       0.55
                 Brant      72237     264066      0.954      0.893      0.924      0.666
                 Other      72237      24536      0.786      0.757       0.79      0.529
                  Gull      72237       3853      0.899      0.877      0.906      0.643
                Canada      72237      33340      0.926      0.863      0.903      0.653
               Emperor      72237       1209      0.314      0.423      0.387      0.261
"""

"""
Last no aug
"""

"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      72237     327004      0.787      0.752      0.782      0.554
                 Brant      72237     264066       0.96      0.901      0.929      0.671
                 Other      72237      24536      0.806      0.731      0.789      0.532
                  Gull      72237       3853      0.923      0.875      0.909      0.644
                Canada      72237      33340      0.942      0.855      0.907      0.662
               Emperor      72237       1209      0.306      0.397      0.377       0.26
"""

"""
Best no aug
"""

"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      72237     327004      0.791      0.757      0.788      0.557
                 Brant      72237     264066       0.96      0.902      0.927       0.67
                 Other      72237      24536      0.814      0.727      0.785       0.53
                  Gull      72237       3853      0.926      0.872      0.918       0.65
                Canada      72237      33340      0.936      0.858      0.905      0.659
               Emperor      72237       1209       0.32      0.423      0.404      0.278
"""


#%% Result notes: YOLOv8L (instead of YOLOv8x), 50% overlap during training

# Training stopped early at 136 epochs; best result observed @ epoch 86

# Results printed at the end of training (should be same as "best no aug" below)

"""
                 Class     Images  Instances      Box(P          R      mAP50   mAP50-95
                   all      72237     327004      0.784      0.749       0.78      0.528
                 Brant      72237     264066      0.958      0.898      0.926      0.638
                 Other      72237      24536      0.799      0.701      0.756      0.491
                  Gull      72237       3853      0.932      0.871      0.913      0.604
                Canada      72237      33340      0.929      0.864      0.904      0.641
               Emperor      72237       1209      0.303      0.414      0.402      0.266
"""

"""
Last w/aug
"""

"""
                 Class     Images  Instances      Box(P          R      mAP50   mAP50-95
                   all      72237     327004      0.757      0.764      0.772      0.532
                 Brant      72237     264066      0.935      0.891      0.918      0.648
                 Other      72237      24536      0.759      0.763      0.779      0.509
                  Gull      72237       3853       0.89       0.88      0.885      0.616
                Canada      72237      33340      0.916      0.865      0.901      0.639
               Emperor      72237       1209      0.285      0.422      0.379       0.25
"""
               
"""
Best w/aug
"""

"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      72237     327004      0.758      0.759      0.772      0.534
                 Brant      72237     264066      0.929      0.892      0.916      0.649
                 Other      72237      24536      0.775      0.742       0.77      0.505
                  Gull      72237       3853      0.881      0.876      0.885      0.616
                Canada      72237      33340      0.916      0.867      0.899      0.641
               Emperor      72237       1209       0.29      0.418      0.392      0.262
"""

"""
Last no aug
"""

"""
                  Class     Images  Instances      Box(P         R      mAP50   mAP50-95
                   all      72237     327004       0.78      0.755      0.778      0.524
                 Brant      72237     264066      0.959      0.898      0.927       0.64
                 Other      72237      24536      0.777      0.723       0.76      0.487
                  Gull      72237       3853      0.929      0.875      0.912      0.606
                Canada      72237      33340      0.935      0.861      0.906      0.638
               Emperor      72237       1209        0.3      0.417      0.384       0.25
"""

"""
Best no aug
"""

"""
                  Class     Images  Instances      Box(P         R      mAP50   mAP50-95
                   all      72237     327004      0.784      0.749       0.78      0.528
                 Brant      72237     264066      0.958      0.898      0.926      0.638
                 Other      72237      24536      0.799      0.701      0.756      0.491
                  Gull      72237       3853      0.932      0.871      0.913      0.605
                Canada      72237      33340      0.929      0.864      0.904      0.641
               Emperor      72237       1209      0.303      0.414      0.402      0.266
"""
