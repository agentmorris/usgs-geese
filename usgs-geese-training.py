########
#
# usgs-geese-training.py
#
# This file documents the model training process, starting from where usgs-geese-training-data-prep.py
# leaves off.  Training happens at the yolov5 CLI, and the exact command line arguments are documented
# in the "Train" cell.
#
# Later cells in this file also:
#
# * Run the YOLOv5 validation scripts
# * Convert YOLOv5 val results to MD .json format
# * Use the MD visualization pipeline to visualize results
# * Use the MD inference pipeline to run the trained model
#
########

#%% TODO

"""

* Play with hyperparameters

https://github.com/agentmorris/MegaDetector/blob/main/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml

https://github.com/agentmorris/MegaDetector/tree/main/detection#training-with-yolov5

* Try smaller YOLOv5's

* Try 1280px YOLOv8 when it's available

* Try fancier patch sampling to minimize the number of patches that still capture 
  all of the training examples.  (to get more examples per unit of training time).

"""


#%% Train

# Tips:
#
# https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results


## Environment prep

"""
mamba create --name yolov5
mamba activate yolov5
mamba install pip
git clone https://github.com/ultralytics/yolov5 yolov5-current
cd yolov5-current
pip install -r requirements.txt
"""

#
# I got this error:
#    
# OSError: /home/user/anaconda3/envs/yolov5/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
#
# There are two ways I've found to fix this:
#
# CUDA was on my LD_LIBRARY_PATH, so this fixes it:
#
# LD_LIBRARY_PATH=
#
# Or if I do this:
# 
# pip uninstall nvidia_cublas_cu11
#
# ...when I run train.py again, it reinstalls the missing CUDA components,
# and everything is fine, but then the error comes back the *next* time I run it.
#
# So I pip uninstall again, and the circle of life continues.
#


## Training

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
BATCH_SIZE=8
IMAGE_SIZE=1280
EPOCHS=200
DATA_YAML_FILE=/home/user/data/usgs-geese/dataset.yaml

TRAINING_RUN_NAME=usgs-geese-yolov5x-230820-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}

python train.py --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} --weights yolov5x6.pt --device 0,1 --project usgs-geese --name ${TRAINING_RUN_NAME} --data ${DATA_YAML_FILE}
"""


## Monitoring training

"""
cd ~/git/yolov5-current
mamba activate yolov5
tensorboard --logdir usgs-geese
"""


## Resuming training

"""
cd ~/git/yolov5-current
mamba activate yolov5
LD_LIBRARY_PATH=
export PYTHONPATH=
python train.py --resume
"""

pass


#%% Make plots during training

import os
import pandas as pd
import matplotlib.pyplot as plt

results_file = os.path.expanduser('~/git/yolov5-current/usgs-geese/usgs-geese-yolov5x-230820-b8-img1280-e200/results.csv')

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())
    
fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 

df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'metrics/precision', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/recall', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/mAP_0.5', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/mAP_0.5:0.95', ax = ax) 

plt.show()

# plt.close('all')


#%% Back up trained weights

"""
TRAINING_RUN_NAME="usgs-geese-yolov5x-230820-b8-img1280-e200"
TRAINING_OUTPUT_FOLDER="/home/user/git/yolov5-current/usgs-geese/${TRAINING_RUN_NAME}/weights"

cp ${TRAINING_OUTPUT_FOLDER}/best.pt ~/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt
cp ${TRAINING_OUTPUT_FOLDER}/last.pt ~/models/usgs-geese/${TRAINING_RUN_NAME}-last.pt
"""

pass


#%% Validation with YOLOv5

import os

model_base = os.path.expanduser('~/models/usgs-geese')
training_run_names = [
    # This is the "round 1" model
    'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss',
    
    # This is the "round 2" model, after cleaning up my training data bug
    'usgs-geese-yolov5x-230820-b8-img1280-e200',
    
    # Somewhere around 70% of the way through "round 2", I captured a checkpoint.
    # I'm *slightly* suspicious of how YOLOv5 chooses the "best" checkpoint, so I 
    # want to evaluate this as well.
    'usgs-geese-yolov5x-230820-b8-img1280-2023.09.02'
]

data_folder = os.path.expanduser('~/data/usgs-geese')
image_size = 1280

# Note to self: validation batch size appears to have no impact on mAP
# (it shouldn't, but I verified that explicitly)
batch_size_val = 8

project_name = os.path.expanduser('~/tmp/usgs-geese-val')
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
            
            cuda_index = len(commands) % n_devices
            cuda_string = 'CUDA_VISIBLE_DEVICES={}'.format(cuda_index)
            cmd = cuda_string + \
                ' python val.py --img {} --batch-size {} --weights {} --project {} --name {} --data {} --save-txt --save-json --save-conf --exist-ok'.format(
                image_size,batch_size_val,model_file,project_name,model_short_name,data_file)        
            if augment:
                cmd += ' --augment'
            commands.append(cmd)

        # ...for each model
    
    # ...augment on/off        
    
# ...for each training run    

for cmd in commands:
    print('')
    print(cmd + '\n')
    

pass


#%% First model (20% val)

"""
Results without augmentation
"""

"""
usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.618      0.563      0.539      0.295
  Brant      11547     101770      0.861      0.927      0.908      0.526
  Other      11547      21246      0.734      0.358      0.419      0.219
   Gull      11547       1594      0.607      0.528       0.45      0.213
 Canada      11547      10961      0.766      0.853      0.844      0.479
Emperor      11547        443       0.12      0.147      0.074     0.0372
Speed: 0.5ms pre-process, 53.8ms inference, 1.1ms NMS per image at shape (8, 3, 1280, 1280)

"""

"""
Results with augmentation
"""

"""
usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.601      0.563      0.535      0.324
  Brant      11547     101770      0.844      0.928      0.906      0.562
  Other      11547      21246      0.729       0.36      0.406      0.225
   Gull      11547       1594      0.553      0.528       0.44       0.28
 Canada      11547      10961      0.764      0.857      0.849      0.513
Emperor      11547        443      0.118       0.14     0.0731      0.041
Speed: 0.5ms pre-process, 118.5ms inference, 1.8ms NMS per image at shape (8, 3, 1280, 1280)
"""


#%% Second model (15% val, not a subset of the first model's val, so not trivially comparable)

# Reported at the end of training

"""
Validating usgs-geese/usgs-geese-yolov5x-230820-b8-img1280-e200/weights/best.pt...
Fusing layers...
Model summary: 416 layers, 140009320 parameters, 0 gradients, 208.0 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.888      0.867      0.885      0.545
                 Brant       8623      80777      0.951      0.932      0.952      0.621
                 Other       8623       6180      0.862      0.743       0.83      0.475
                  Gull       8623       1008      0.927      0.927      0.933      0.558
                Canada       8623      11243      0.911      0.928       0.94      0.611
               Emperor       8623        662       0.79      0.805       0.77      0.459
"""


# ss-last w/aug (in this case, this val set includes some of the model's training data)
"""
Model summary: 416 layers, 140009320 parameters, 0 gradients, 208.0 GFLOPs
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.667       0.77      0.678      0.437
                 Brant       8623      80777      0.938      0.882      0.926       0.59
                 Other       8623       6180      0.631      0.739      0.663      0.419
                  Gull       8623       1008      0.314      0.765      0.466      0.321
                Canada       8623      11243      0.818      0.751      0.803      0.516
               Emperor       8623        662      0.634      0.716      0.531       0.34
Speed: 0.5ms pre-process, 115.4ms inference, 1.5ms NMS per image at shape (8, 3, 1280, 1280)
"""

# ss-best w/aug (in this case, this val set includes some of the model's training data)
"""
Model summary: 416 layers, 140009320 parameters, 0 gradients, 208.0 GFLOPs
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.744      0.751      0.763      0.461
                 Brant       8623      80777      0.939      0.886      0.943      0.576
                 Other       8623       6180      0.806      0.706      0.786      0.449
                  Gull       8623       1008      0.429       0.75      0.621        0.4
                Canada       8623      11243      0.923       0.79      0.912      0.553
               Emperor       8623        662      0.624      0.624      0.551      0.328
Speed: 0.9ms pre-process, 120.7ms inference, 2.2ms NMS per image at shape (8, 3, 1280, 1280)
"""

# ss-last no aug (in this case, this val set includes some of the model's training data)
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870        0.7      0.821      0.761      0.418
                 Brant       8623      80777      0.932      0.909      0.944      0.539
                 Other       8623       6180      0.671      0.753      0.778      0.411
                  Gull       8623       1008      0.407      0.889      0.615       0.31
                Canada       8623      11243      0.866      0.874      0.915      0.519
               Emperor       8623        662      0.622      0.678      0.553      0.311
Speed: 0.9ms pre-process, 85.5ms inference, 1.4ms NMS per image at shape (8, 3, 1280, 1280)
"""

# ss-best no aug (in this case, this val set includes some of the model's training data)
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.668       0.77      0.671      0.393
                 Brant       8623      80777      0.938      0.879      0.929      0.549
                 Other       8623       6180      0.631      0.737      0.662      0.387
                  Gull       8623       1008      0.315      0.761      0.422      0.227
                Canada       8623      11243      0.821      0.754      0.811       0.48
               Emperor       8623        662      0.637      0.719      0.532      0.323
Speed: 0.5ms pre-process, 83.4ms inference, 1.0ms NMS per image at shape (8, 3, 1280, 1280)
"""

# 230820-last w/aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.872      0.866      0.876      0.549
                 Brant       8623      80777      0.937       0.92      0.943      0.606
                 Other       8623       6180      0.839      0.758      0.814       0.46
                  Gull       8623       1008      0.891      0.922      0.935      0.624
                Canada       8623      11243       0.91      0.916      0.929      0.604
               Emperor       8623        662      0.784      0.816      0.762       0.45
Speed: 0.5ms pre-process, 202.7ms inference, 0.9ms NMS per image at shape (8, 3, 1280, 1280)
"""

# 230820-best w/aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.887       0.87      0.891      0.576
                 Brant       8623      80777      0.955      0.931      0.954      0.634
                 Other       8623       6180       0.86      0.763      0.843      0.503
                  Gull       8623       1008      0.907      0.929      0.938      0.646
                Canada       8623      11243      0.916      0.919      0.942      0.629
               Emperor       8623        662      0.796      0.806      0.778      0.467
Speed: 0.9ms pre-process, 209.4ms inference, 1.3ms NMS per image at shape (8, 3, 1280, 1280)
"""

# 230820-last no aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.877      0.857       0.87      0.532
                 Brant       8623      80777      0.943      0.919       0.94        0.6
                 Other       8623       6180      0.857      0.729      0.799      0.443
                  Gull       8623       1008       0.91       0.92       0.93      0.573
                Canada       8623      11243      0.908      0.921      0.928      0.603
               Emperor       8623        662      0.768      0.799      0.751      0.442
Speed: 0.5ms pre-process, 84.2ms inference, 0.9ms NMS per image at shape (8, 3, 1280, 1280)
"""

# 230820-best no aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.888      0.867      0.885      0.547
                 Brant       8623      80777      0.952      0.932      0.952      0.622
                 Other       8623       6180      0.862      0.742       0.83      0.476
                  Gull       8623       1008      0.927      0.927      0.933      0.562
                Canada       8623      11243      0.911      0.927       0.94      0.613
               Emperor       8623        662       0.79      0.805       0.77       0.46
Speed: 0.9ms pre-process, 87.6ms inference, 1.0ms NMS per image at shape (8, 3, 1280, 1280)
"""

# 2023.09.02 snapshot last w/aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.876      0.867      0.883      0.561
                 Brant       8623      80777      0.946      0.923      0.949      0.622
                 Other       8623       6180      0.847      0.757      0.829      0.482
                  Gull       8623       1008      0.898      0.929      0.938      0.629
                Canada       8623      11243      0.914      0.915      0.936      0.619
               Emperor       8623        662      0.774       0.81      0.761      0.452
"""

# 2023.09.02 snapshot best w/aug
"""
val: Scanning /home/user/data/usgs-geese/yolo_val.cache... 8623 images, 0 backgrounds, 0 corrupt
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all       8623      99870      0.887       0.87      0.891      0.576
                 Brant       8623      80777      0.955      0.931      0.954      0.634
                 Other       8623       6180       0.86      0.763      0.843      0.503
                  Gull       8623       1008      0.907      0.929      0.938      0.646
                Canada       8623      11243      0.916      0.919      0.942      0.629
               Emperor       8623        662      0.796      0.806      0.778      0.467
"""

# 2023.09.02 snapshot last no aug
"""
I didn't bother to run this.
"""

# 2023.09.02 snapshot best no aug
"""
I didn't bother to run this.
"""

#%% Convert YOLO val .json results to MD .json format

# pip install jsonpickle humanfriendly tqdm skicit-learn

import os
from data_management import yolo_output_to_md_output

import json
import glob

class_mapping_file = os.path.expanduser('~/data/usgs-geese/usgs-geese-md-class-mapping.json')
with open(class_mapping_file,'r') as f:
    category_id_to_name = json.load(f)
                        
base_folder = os.path.expanduser('~/tmp/usgs-geese-val')
run_folders = os.listdir(base_folder)
run_folders = [os.path.join(base_folder,s) for s in run_folders]
run_folders = [s for s in run_folders if os.path.isdir(s)]

image_base = os.path.expanduser('~/data/usgs-geese/yolo_val')
image_files = glob.glob(image_base + '/*.jpg')

prediction_files = []

# run_folder = run_folders[0]
for run_folder in run_folders:
    prediction_files_this_folder = glob.glob(run_folder+'/*_predictions.json')
    assert len(prediction_files_this_folder) <= 1
    if len(prediction_files_this_folder) == 1:
        prediction_files.append(prediction_files_this_folder[0])        

md_format_prediction_files = []

# prediction_file = prediction_files[0]
for prediction_file in prediction_files:

    detector_name = os.path.splitext(os.path.basename(prediction_file))[0].replace('_predictions','')
    
    # print('Converting {} to MD format'.format(prediction_file))
    output_file = prediction_file.replace('.json','_md-format.json')
    assert output_file != prediction_file
    
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=prediction_file,
        image_folder=image_base,
        output_file=output_file,
        yolo_category_id_to_name=category_id_to_name,                              
        detector_name=detector_name,
        image_id_to_relative_path=None,
        offset_yolo_class_ids=False)    
    
    md_format_prediction_files.append(output_file)

# ...for each prediction file


#%% Visualize results with the MD visualization pipeline

postprocessing_output_folder = os.path.expanduser('~/tmp/usgs-geese-previews')

from md_utils import path_utils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

# prediction_file = md_format_prediction_files[0]
for prediction_file in md_format_prediction_files:
    
    assert '_md-format.json' in prediction_file
    base_task_name = os.path.basename(prediction_file).replace('_md-format.json','')

    options = PostProcessingOptions()
    options.image_base_dir = image_base
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.15
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
    
    options.api_output_file = prediction_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file)

# ...for each prediction file


#%%

#
# Run the MD pred pipeline 
#

"""
export PYTHONPATH=/home/user/git/MegaDetector
cd ~/git/MegaDetector/detection/
mamba activate yolov5

TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
MODEL_FILE="/home/user/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt"
DATA_FOLDER="/home/user/data/usgs-geese-mini-500"
RESULTS_FOLDER=${DATA_FOLDER}/results

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_val ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${DATA_FOLDER}/usgs-geese-md-class-mapping.json

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_train ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${DATA_FOLDER}/usgs-geese-md-class-mapping.json

"""

#
# Visualize results using the MD pipeline
#

"""
mamba deactivate

cd ~/git/MegaDetector/api/batch_processing/postprocessing/

TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
DATA_FOLDER="/home/user/data/usgs-geese-mini-500"
RESULTS_FOLDER=${DATA_FOLDER}/results
PREVIEW_FOLDER=${DATA_FOLDER}/preview

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val --image_base_dir ${DATA_FOLDER}/yolo_val --n_cores 12 --confidence_threshold 0.25 --parallelize_rendering_with_processes

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train --image_base_dir ${DATA_FOLDER}/yolo_train --n_cores 12 --confidence_threshold 0.25 --parallelize_rendering_with_processes

xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val/index.html 
xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train/index.html

"""

pass
