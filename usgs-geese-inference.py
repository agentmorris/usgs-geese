########
#
# usgs-geese-inference.py
#
# This is a module that runs inference on a folder of images, by breaking each 
# image into overlapping  1280x1280 patches, running the model on each patch, and 
# eliminating redundant boxes from the results.
#
# This module does not have a command-line driver; see run-izembek-model.py for
# a command-line interface to this module.
#
########

#%% Constants and imports

import os
import stat
import json
import glob
import shutil
import humanfriendly

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

from megadetector.utils import path_utils
from megadetector.utils import process_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection.run_tiled_inference import get_patch_boundaries, in_place_nms
from megadetector.data_management.yolo_output_to_md_output import yolo_json_output_to_md_output
from megadetector.postprocessing import combine_batch_outputs
from megadetector.postprocessing.subset_json_detector_output import \
    subset_json_detector_output, SubsetJsonDetectorOutputOptions

validate_against_dataset_file = False

# This is the size of the training images, but the only thing that matters is the
# patch size, and it's possible to run arbitrary image sizes.  So at this point, this 
# is more of a consistency check for scenarios where we *think* the images will be the
# same size as the training image.
expected_image_width = 8688
expected_image_height = 5792

patch_size = (1280,1280)

# The size at which to run inference... generally should be the same as the 
# patch size.
yolo_image_size = 1280

# Right now, for debugging, we run inference with a low confidence threshold, but after
# inference, we strip out very-low-confidence detections.
post_inference_conf_thres = 0.025

parallelize_patch_generation_with_threads = True

force_patch_generation = True
overwrite_existing_patches = True
overwrite_md_results_files = True

patch_jpeg_quality = 95

default_patch_overlap = 0.1

# This isn't NMS in the usual sense of redundant model predictions; this is being
# used to de-duplicate predictions from overlapping patches.
nms_iou_threshold = 0.45

# Performing per-patch NMS is just a debugging tool, for making patch-level previews.
# There's not really a reason to  do this at the patch level when we have to do this at the
# image level anyway.
#
# The only thing we're removing when we perform NMS at the patch level is the case where nearly-identical 
# boxes are assigned to multiple classes.
do_within_patch_nms = False

# Threshold used for including results in the json file during inference
default_inference_conf_thres = '0.001'
    
default_inference_batch_size = 8

# Useful defaults on my training machine
default_project_dir = os.path.expanduser('~/tmp/usgs-inference')    
default_yolo_working_dir = os.path.expanduser('~/git/yolov5-current')
default_model_base = os.path.expanduser('~/models/usgs-geese')
default_model_file = os.path.join(default_model_base,
                          'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt')

default_yolo_category_id_to_name = \
{
    0: 'Brant',
    1: 'Other',
    2: 'Gull',
    3: 'Canada',
    4: 'Emperor'
}

default_devices = [0]

# OS-specific script line continuation character, comment character, and script extension
if os.name == 'nt':
    slcc = '^'
    scc = 'REM'
    script_extension = '.bat'
else:
    slcc = '\\'
    scc = '#' 
    script_extension = '.sh'


#%% Options management

class USGSGeeseInferenceOptions:
    """
    All of these options map to command-line options to run_izembek_model.py, so I'm
    not documenting them (again) here.
    """
       
    def __init__(self, project_dir=None,yolo_working_dir=None,model_file=None,
                 devices=None,recursive=True,allow_variable_image_size=True,
                 use_symlinks=True,no_cleanup=False,no_augment=False,
                 category_mapping_file=None):
        if project_dir is None:
            self.project_dir = default_project_dir
        else:
            self.project_dir = project_dir
        
        if yolo_working_dir is None:
            self.yolo_working_dir = default_yolo_working_dir
        else:
            self.yolo_working_dir = yolo_working_dir
        
        if model_file is None:
            self.model_file = default_model_file
        else:
            self.model_file = model_file
        
        if devices is None:
            self.devices = default_devices
        else:
            self.devices = devices
        
        self.recursive = recursive
        self.allow_variable_image_size = allow_variable_image_size
        self.use_symlinks = use_symlinks
        self.augment = (not no_augment)
        
        # What things should we clean up at the end of the process for a folder?
        self.cleanup_targets = ['patch_cache_file','dataset_files','symlink_images',
                                'yolo_results','inference_scripts','chunk_cache_file',
                                'patches','patch_level_results']
        
        if no_cleanup:
            self.cleanup_targets = []

        # Derived paths
        self.project_symlink_dir = os.path.join(project_dir,'symlink_images').replace('\\','/')
        self.project_dataset_file_dir = os.path.join(project_dir,'dataset_files').replace('\\','/')
        self.project_patch_dir = os.path.join(project_dir,'patches').replace('\\','/')
        self.project_inference_script_dir = os.path.join(project_dir,'inference_scripts').replace('\\','/')
        self.project_yolo_results_dir = os.path.join(project_dir,'yolo_results').replace('\\','/')
        self.project_image_level_results_dir = os.path.join(project_dir,'image_level_results').replace('\\','/')
        self.project_chunk_cache_dir = os.path.join(project_dir,'chunk_cache').replace('\\','/')
        self.project_md_formatted_results_dir = os.path.join(project_dir,'md_formatted_results').replace('\\','/')

        self.n_cores_patch_generation = 16
        
        if category_mapping_file is None:
            self.yolo_category_id_to_name = default_yolo_category_id_to_name
        elif category_mapping_file.endswith('.yaml'):
            self.yolo_category_id_to_name = read_classes_from_yolo_dataset_file(
                category_mapping_file)
        elif category_mapping_file.endswith('.json'):
            with open(category_mapping_file,'r') as f:
                self.yolo_category_id_to_name = json.load(f)            
        else:
            raise ValueError('Unrecognized category mapping file {}'.format(category_mapping_file))
        self.yolo_category_id_to_name = {int(k):v for k,v in \
                                         self.yolo_category_id_to_name.items()}


#%% Validate class names if requested

# It's only really useful to do this on the training machine

def read_classes_from_yolo_dataset_file(fn):

    import re

    with open(fn,'r') as f:
        lines = f.readlines()
            
    to_return = {}
    pat = '\d+: .+'
    for s in lines:
        if re.search(pat,s) is not None:
            tokens = s.split(':')
            assert len(tokens) == 2, 'Invalid token in category file {}'.format(fn)
            to_return[int(tokens[0].strip())] = tokens[1].strip()
        
    return to_return
    
if validate_against_dataset_file:
    
    dataset_definition_file = os.path.expanduser('~/data/usgs-geese/dataset.yaml')  
    yolo_category_id_to_name = read_classes_from_yolo_dataset_file(dataset_definition_file)
    assert yolo_category_id_to_name == default_yolo_category_id_to_name, \
        'Error validating YOLO category list'

# As far as I can tell, this model does not have the class names saved, so just noting to self
# that I tried this:
#
# m = torch.load(model_file)
# print(m['model'].names)
#
#    ['0', '1', '2', '3', '4']


#%% Support functions

def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    
    patch_name = image_name + '_' + \
        str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name


def extract_patch_from_image(im,patch_xy,patch_wh,
                             patch_image_fn=None,patch_folder=None,image_name=None,overwrite=True):
    """
    Extracts a patch from the provided image, writing the patch out to patch_image_fn.  im
    can be a string or a PIL image.
    
    patch_xy is a length-2 tuple specifying the upper-left corner of the patch.
    
    image_name and patch_folder are only required if patch_image_fn is None.
    
    Returns a dictionary with fields xmin,xmax,ymin,ymax,patch_fn.
    """
    
    if isinstance(im,str):
        pil_im = vis_utils.open_image(im)
    else:
        pil_im = im
        
    patch_x_min = patch_xy[0]
    patch_y_min = patch_xy[1]
    patch_x_max = patch_x_min + patch_wh[0] - 1
    patch_y_max = patch_y_min + patch_wh[1] - 1

    # PIL represents coordinates in a way that is very hard for me to get my head
    # around, such that even though the "right" and "bottom" arguments to the crop()
    # function are inclusive... well, they're not really.
    #
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    #
    # So we add 1 to the max values.
    patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
    assert patch_im.size[0] == patch_wh[0], 'Illegal patch size'
    assert patch_im.size[1] == patch_wh[1], 'Illegal patch size'

    if patch_image_fn is None:
        assert patch_folder is not None,\
            "If you don't supply a patch filename to extract_patch_from_image, you need to supply a folder name"
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(patch_folder,patch_name + '.jpg').replace('\\','/')
    
    if os.path.isfile(patch_image_fn) and (not overwrite):
        pass
    else:        
        patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
    
    patch_info = {}
    patch_info['xmin'] = patch_x_min
    patch_info['xmax'] = patch_x_max
    patch_info['ymin'] = patch_y_min
    patch_info['ymax'] = patch_y_max
    patch_info['patch_fn'] = patch_image_fn
    
    return patch_info

                             
def extract_patches_for_image(image_fn,patch_folder,image_name_base=None,
                              overwrite=True,allow_variable_image_size=True):
    """
    Extract patches from image_fn to separate image files in patch_folder.
    
    Returns a dictionary that looks like:
        
        {
             'image_fn':'/whatever/image/you/passed/in',
             'patches':
             [
                 {
                  'xmin':x0,'ymin':y0,'xmax':x1,'ymax':y1,
                  'patch_fn':'/patch/folder/patch/image/name.jpg',
                  'image_fn':'/whatever/image/you/passed/in'
                  }
             ]
        }
            
    """
        
    os.makedirs(patch_folder,exist_ok=True)
    
    if image_name_base is None:
        image_name_base = os.path.dirname(image_fn)
        
    image_relative_path = os.path.relpath(image_fn,image_name_base)    
    image_name = path_utils.clean_filename(image_relative_path,char_limit=None,force_lower=True)
    
    pil_im = vis_utils.open_image(image_fn)
    if not allow_variable_image_size:
        assert pil_im.size[0] == expected_image_width, 'Illegal image size'
        assert pil_im.size[1] == expected_image_height, 'Illegal image size'
    
    image_width = pil_im.size[0]
    image_height = pil_im.size[1]
    image_size = (image_width,image_height)
    patch_start_positions = get_patch_boundaries(image_size,patch_size,
                                                 patch_stride=(1.0-default_patch_overlap))
    
    patches = []
    
    # i_patch = 0; patch_xy = patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(patch_start_positions):        
        patch_info = extract_patch_from_image(
            pil_im,patch_xy,patch_size,
            patch_image_fn=None,patch_folder=patch_folder,
            image_name=image_name,overwrite=overwrite)
        patch_info['image_fn'] = image_fn
        patches.append(patch_info)
    
    # ...for each patch
    
    image_patch_info = {}
    image_patch_info['patches'] = patches
    image_patch_info['image_fn'] = image_fn
        
    return image_patch_info

# ...extract_patches_for_image()
    

def generate_patches_for_image(image_fn_relative,patch_folder_base,input_folder_base,
                               overwrite=True,allow_variable_image_size=True):
    """
    Wrapper for extract_patches_for_image() that chooses a patch folder name based
    on the image name.
    
    See extract_patches_for_image for return format.
    """
    
    image_fn = os.path.join(input_folder_base,image_fn_relative).replace('\\','/')
    assert os.path.isfile(image_fn), 'Image file {} does not exist'.format(image_fn)
    patch_folder = os.path.join(patch_folder_base,image_fn_relative).replace('\\','/')
    image_patch_info = extract_patches_for_image(image_fn,patch_folder,
                                                 input_folder_base,overwrite=overwrite,
                                                 allow_variable_image_size=allow_variable_image_size)
    return image_patch_info
    

def create_symlink_folder_for_patches(patch_files,symlink_dir,inference_options=None):
    """
    Create a folder of symlinks pointing to patches, so we have a flat folder 
    structure that's friendly to the way YOLOv5's val.py works.  Returns a dict
    mapping patch IDs to files.
    
    On Windows, this requires admin permissions, so we use copying as a workaround for now.
    The fact that this function is still called "create_symlink_..." suggests that
    I see all of this as a temporary solution.
    """
    
    if inference_options is None:
        inference_options = USGSGeeseInferenceOptions()
    
    os.makedirs(symlink_dir,exist_ok=True)

    def safe_create_link(link_exists,link_new):
        
        if os.path.exists(link_new) or os.path.islink(link_new):
            assert os.path.islink(link_new), 'Oops, {} is a real file, not a link'.format(link_new)
            if not os.readlink(link_new) == link_exists:
                os.remove(link_new)
                os.symlink(link_exists,link_new)
        else:
            os.symlink(link_exists,link_new)
    
    patch_id_to_file = {}
    
    # i_patch = 0; patch_fn = patch_files[i_patch]
    for i_patch,patch_fn in tqdm(enumerate(patch_files),total=len(patch_files)):
        
        ext = os.path.splitext(patch_fn)[1]
        
        patch_id_string = str(i_patch).zfill(10)
        patch_id_to_file[patch_id_string] = patch_fn
        symlink_name = patch_id_string + ext
        symlink_full_path = os.path.join(symlink_dir,symlink_name).replace('\\','/')
        if inference_options.use_symlinks:
            safe_create_link(patch_fn,symlink_full_path)
        else:
            shutil.copyfile(patch_fn,symlink_full_path)
        
    # ...for each image
    
    return patch_id_to_file


def create_yolo_dataset_file(dataset_file,symlink_dir,yolo_category_id_to_name):
    """
    Create a dataset.yml file that YOLOv5's val.py can read, telling it which
    folder to run inference on.
    """
    
    category_ids = sorted(list(yolo_category_id_to_name.keys()))
    
    with open(dataset_file,'w') as f:
        f.write('path: {}\n'.format(symlink_dir))
        f.write('train: .\n')
        f.write('val: .\n')
        f.write('test: .\n')
        f.write('\n')
        f.write('nc: {}\n'.format(len(yolo_category_id_to_name)))
        f.write('\n')
        f.write('names:\n')
        for category_id in category_ids:
            assert isinstance(category_id,int), 'Illegal category ID {}'.format(category_id)
            f.write('  {}: {}\n'.format(category_id,yolo_category_id_to_name[category_id]))
    


def run_yolo_model(project_dir,run_name,dataset_file,model_file,yolo_working_dir,
                   execute=True,augment=True,device_string='0',
                   batch_size=default_inference_batch_size,
                   conf_thres=default_inference_conf_thres):
    """
    Invoke Python in a shell to run the model on an existing YOLOv5-formatted dataset.
    
    If 'execute' if false, just prepares the list of commands to run the model, but
    doesn't actually run it.
    """
    
    run_dir = os.path.join(project_dir,run_name).replace('\\','/')
    os.makedirs(run_dir,exist_ok=True)    
    
    image_size_string = str(round(yolo_image_size))
            
    cmd = 'python val.py --data "{}"'.format(dataset_file)
    cmd += ' --weights "{}"'.format(model_file)
    cmd += ' --batch-size {} --imgsz {} --conf-thres {} --task test'.format(
        batch_size,image_size_string,conf_thres)
    cmd += ' --device "{}" --save-json'.format(device_string)
    cmd += ' --project "{}" --name "{}" --exist-ok'.format(project_dir,run_name)
    
    # To save .txt results (useful for debugging)
    # cmd += ' --save-txt --save-conf'
    
    if augment:
        cmd += ' --augment'
    
    if (execute):
        
        initial_working_dir = os.getcwd()
        os.chdir(yolo_working_dir)
        
        from ct_utils import execute_command_and_print
        cmd_result = execute_command_and_print(cmd)
        
        assert cmd_result['status'] == 0, 'Error running YOLOv5'
        
        os.chdir(initial_working_dir)
    
    return cmd

# ...run_yolo_model()


#%% The main function: run the model recursively on  folder

def run_model_on_folder(input_folder_base,inference_options=None):
    """
    Run the goose detection model on all images in a folder
    """
    
    ##%% Input validation
    
    if inference_options is None:
        inference_options = USGSGeeseInferenceOptions()
        
    assert os.path.isdir(input_folder_base), \
        'Could not find input folder {}'.format(input_folder_base)
    
    folder_name_clean = input_folder_base.replace('\\','/').replace('/','_').replace(' ','_').replace(':','_')
    if folder_name_clean.startswith('_'):
        folder_name_clean = folder_name_clean[1:]
    
    
    ##%% Enumerate images
    
    images_absolute = path_utils.find_images(input_folder_base,
                                             recursive=inference_options.recursive,
                                             convert_slashes=True)
    images_relative = [os.path.relpath(fn,input_folder_base) for fn in images_absolute]    
    

    ##%% Generate patches
    
    os.makedirs(inference_options.project_chunk_cache_dir,exist_ok=True)

    # This is a .json file that includes metadata about our patches; this is only used during
    # debugging, when we want to re-start from this point but don't want to re-generate patches
    patch_cache_file = os.path.join(inference_options.project_chunk_cache_dir,
                                    folder_name_clean + '_patch_info.json').replace('\\','/')
    patch_folder_for_folder = os.path.join(inference_options.project_patch_dir,
                                           folder_name_clean).replace('\\','/')
                                           
    if force_patch_generation or (not os.path.isfile(patch_cache_file)):
                
        print('Generating patches for {}'.format(input_folder_base))
            
        if inference_options.n_cores_patch_generation == 1:            
            all_image_patch_info = []
            # image_fn_relative = images_relative[0]
            for image_fn_relative in tqdm(images_relative):
                image_patch_info = generate_patches_for_image(image_fn_relative,patch_folder_for_folder,
                                     input_folder_base,
                                     allow_variable_image_size=inference_options.allow_variable_image_size)
                all_image_patch_info.append(image_patch_info)
        else:                
            if parallelize_patch_generation_with_threads:
                pool = ThreadPool(inference_options.n_cores_patch_generation)
                print('Generating patches on a pool of ' + \
                      '{} threads'.format(inference_options.n_cores_patch_generation))
            else:
                pool = Pool(inference_options.n_cores_patch_generation)
                print('Generating patches on a pool of ' + \
                      '{} processes'.format(inference_options.n_cores_patch_generation))
    
            all_image_patch_info = list(tqdm(pool.imap(
                partial(generate_patches_for_image,
                        patch_folder_base=patch_folder_for_folder,
                        input_folder_base=input_folder_base,
                        overwrite=overwrite_existing_patches,
                        allow_variable_image_size=inference_options.allow_variable_image_size), 
                images_relative), total=len(images_relative)))
            
        all_patch_files = []
        for image_patch_info in all_image_patch_info:
            image_patch_files = [pi['patch_fn'] for pi in image_patch_info['patches']]
            all_patch_files.extend(image_patch_files)
            
        total_patch_size_bytes = 0
        for fn in tqdm(all_patch_files):
            total_patch_size_bytes += os.path.getsize(fn)
        total_patch_size_str = humanfriendly.format_size(total_patch_size_bytes)
        
        print('Generated {} patches for {} images in folder {}, taking up {}'.format(
            len(all_patch_files),len(all_image_patch_info),
            input_folder_base,total_patch_size_str))

        with open(patch_cache_file,'w') as f:
            json.dump(all_image_patch_info,f,indent=1)
            
        print('Wrote patch info to {}'.format(patch_cache_file))        
        
        del image_patch_info
        del all_patch_files
    
    else:
        
        print('Loading cached patch information from {}'.format(patch_cache_file))
        
        with open(patch_cache_file,'r') as f:
            all_image_patch_info = json.load(f)
    
    # See extract_patches_for_image for the format of all_image_patch_info
    all_patch_files = []
    for image_patch_info in all_image_patch_info:
        for patch_info in image_patch_info['patches']:
            all_patch_files.append(patch_info['patch_fn'])
            
    # Double-check that we have the right number of patches (n_images * n_patches_per_image)
    if not inference_options.allow_variable_image_size:
        n_patches_per_image = len(get_patch_boundaries(
            (expected_image_width,expected_image_height),
            patch_size,
            patch_stride=(1.0-default_patch_overlap)))
        assert len(all_patch_files) == n_patches_per_image * len(images_relative), \
            'Unexpected number of patches'
    
    
    ##%% Split patches into chunks (one per GPU), and generate symlink folder(s)
    
    def split_list(L, n):
        k, m = divmod(len(L), n)
        return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
            
    # Split patches into chunks
    n_chunks = len(inference_options.devices)
    patch_chunks = split_list(all_patch_files,n_chunks)
    assert sum([len(p) for p in patch_chunks]) == len(all_patch_files), \
        'Unexpected number of patch chunks'
    
    chunk_info = []
    
    folder_symlink_dir = os.path.join(inference_options.project_symlink_dir,
                                      folder_name_clean).replace('\\','/')
    symlink_folder_existed = os.path.isdir(folder_symlink_dir)
    
    for i_chunk,chunk_files in enumerate(patch_chunks):
    
        chunk_symlink_dir = os.path.join(
            folder_symlink_dir,'chunk_{}'.format(str(i_chunk).zfill(2))).replace('\\','/')
    
        print('Generating symlinks for chunk {} in folder {}'.format(
            i_chunk,chunk_symlink_dir))
    
        chunk_patch_id_to_file = create_symlink_folder_for_patches(chunk_files,
                                                                   chunk_symlink_dir,
                                                                   inference_options)
        
        # If we just created the symlink folder, double-check that it has exactly as many
        # patch files as we think it should.  This is just a debugging consistency check
        # during dev.
        if not symlink_folder_existed:
            chunk_symlinks = os.listdir(chunk_symlink_dir)
            assert len(chunk_symlinks) == len(chunk_patch_id_to_file), \
                'Unexpected number of patch files'            
        
        chunk = {'chunk_id':'chunk_{}'.format(str(i_chunk).zfill(2)),
                           'symlink_dir':chunk_symlink_dir,
                           'patch_id_to_file':chunk_patch_id_to_file}
        
        chunk_info.append(chunk)

    # ...for each chunk

    
    ##%% Generate .yaml files (to tell YOLO where the data is)
    
    folder_dataset_file_dir = os.path.join(inference_options.project_dataset_file_dir,
                                           folder_name_clean).replace('\\','/')
    
    os.makedirs(folder_dataset_file_dir,exist_ok=True)
    
    for i_chunk,chunk in enumerate(chunk_info):
                
        chunk['dataset_file'] = os.path.join(
            folder_dataset_file_dir,chunk['chunk_id'] + '_dataset.yaml').replace('\\','/')
        print('Writing dataset file for chunk {} to {}'.format(i_chunk,chunk['dataset_file']))
        create_yolo_dataset_file(chunk['dataset_file'],chunk['symlink_dir'],
                                 inference_options.yolo_category_id_to_name)
    
        
    ##%% Prepare commands to run the model on symlink folder(s)
    
    folder_inference_script_dir = os.path.join(inference_options.project_inference_script_dir,
                                               folder_name_clean).replace('\\','/')
    os.makedirs(folder_inference_script_dir,exist_ok=True)
    
    folder_yolo_results_dir = os.path.join(inference_options.project_yolo_results_dir,
                                           folder_name_clean).replace('\\','/')
    os.makedirs(folder_yolo_results_dir,exist_ok=True)
    
    for i_chunk,chunk in enumerate(chunk_info):
        
        device = inference_options.devices[i_chunk]
        if device < 0:
            device_string = 'cpu'
        else:
            device_string = str(device)
        
        chunk['run_name'] = 'inference-output-' + chunk['chunk_id']        
        chunk['run_output_dir'] = os.path.join(
            folder_yolo_results_dir,chunk['run_name']).replace('\\','/')
        chunk['cmd'] = run_yolo_model(folder_yolo_results_dir,
                                      chunk['run_name'],
                                      chunk['dataset_file'],
                                      inference_options.model_file,
                                      inference_options.yolo_working_dir,
                                      execute=False,
                                      augment=inference_options.augment,
                                      device_string=device_string)
        chunk['script_name'] = os.path.join(
            folder_inference_script_dir,'run_chunk_{}_device_{}{}'.format(
            str(i_chunk).zfill(2),device_string,script_extension)).replace('\\','/')
        with open(chunk['script_name'],'w') as f:
            f.write(chunk['cmd'])
        st = os.stat(chunk['script_name'])
        os.chmod(chunk['script_name'], st.st_mode | stat.S_IEXEC)
        print('Wrote chunk {} script to {}'.format(i_chunk,chunk['script_name']))
        
    # ...for each chunk
    
    
    ##%% Save/load chunk state for debugging (because stuff crashes)
    
    chunk_cache_file = os.path.join(inference_options.project_chunk_cache_dir,
                                    folder_name_clean + '_chunk_info.json').replace('\\','/')
    os.makedirs(inference_options.project_chunk_cache_dir,exist_ok=True)
    
    if False:
        
        # Save state
        with open(chunk_cache_file,'w') as f:
            json.dump(chunk_info,f,indent=1)
    
    if False:

        # Load state
        with open(chunk_cache_file,'r') as f:
            chunk_info = json.load(f)
    

    ##%% Run inference
    
    # Changes the current working directory, making no attempt to change it back.          
    execute_inline = True    
    
    print('Inference commands:\n')
    for chunk in chunk_info:
        print('{}'.format(chunk['script_name']))
    print('')
    
    if (execute_inline):
        
        chunk_commands = [chunk['script_name'] for chunk in chunk_info]
        n_workers = len(inference_options.devices)
       
        # Should we use threads (vs. processes) for parallelization?
        use_threads = True
       
        def run_chunk(cmd):
            os.environ['LD_LIBRARY_PATH']=''
            os.chdir(inference_options.yolo_working_dir)
            return process_utils.execute_and_print(cmd,print_output=True,encoding='utf-8',verbose=True)
           
        if n_workers == 1:  
         
          results = []
          for i_command,command in enumerate(chunk_commands):    
            results.append(run_chunk(command))
         
        else:
         
          if use_threads:
            print('Starting parallel thread pool with {} workers'.format(n_workers))
            pool = ThreadPool(n_workers)
          else:
            print('Starting parallel process pool with {} workers'.format(n_workers))
            pool = Pool(n_workers)
       
          results = list(pool.map(run_chunk,chunk_commands))
          
          assert all([r['status'] == 0 for r in results]), \
              'Error running one or more inference processes'
          
    else:
    
        print('Bypassing inline execution')
        
        
    ##%% Convert patch results for each chunk to MD format
    
    model_short_name = os.path.basename(inference_options.model_file).replace('.pt','')    
    
    # i_chunk = 0; chunk = chunk_info[i_chunk]
    for i_chunk,chunk in enumerate(chunk_info):
        
        run_dir = chunk['run_output_dir']
        assert os.path.isdir(run_dir), 'Output folder {} does not exist'.format(run_dir)
        
        json_files = glob.glob(run_dir + '/*.json')
        json_files = [fn for fn in json_files if 'md_format' not in fn]
        assert len(json_files) == 1, 'Inference failed, no output written by YOLO script'
        
        yolo_json_file = json_files[0]
        chunk['yolo_json_file'] = yolo_json_file
        
        md_formatted_results_file = yolo_json_file.replace('.json','-md_format.json')
        chunk['md_formatted_results_file'] = md_formatted_results_file
        assert md_formatted_results_file != yolo_json_file, '.json file ambiguity error'
            
        if os.path.isfile(md_formatted_results_file) and (not overwrite_md_results_files):
            
            print('Bypassing YOLO --> MD conversion for {}, output file exists'.format(
                yolo_json_file))
            
        else:
            
            print('Reading results from {}'.format(yolo_json_file))
            
            with open(yolo_json_file,'r') as f:
                yolo_results = json.load(f)
            
            print('Read {} results for {} patches'.format(
                len(yolo_results),len(chunk['patch_id_to_file'])))
    
            # Convert patch results to MD output format
                
            patch_id_to_relative_path = {}
            
            # i_patch = 0; patch_id = next(iter(chunk['patch_id_to_file'].keys()))
            for patch_id in chunk['patch_id_to_file'].keys():
                fn = chunk['patch_id_to_file'][patch_id]
                assert patch_folder_for_folder in fn, 'Patch lookup error'
                relative_fn = os.path.relpath(fn,patch_folder_for_folder)
                patch_id_to_relative_path[patch_id] = relative_fn
               
            yolo_json_output_to_md_output(yolo_json_file,
                                          image_folder=patch_folder_for_folder,
                                          output_file=md_formatted_results_file,
                                          yolo_category_id_to_name=inference_options.yolo_category_id_to_name,
                                          detector_name=model_short_name,
                                          image_id_to_relative_path=patch_id_to_relative_path,
                                          offset_yolo_class_ids=False)
    
        del md_formatted_results_file,run_dir
        
    # ...for each chunk
    
    
    ##%% Merge results files from each chunk into one (patch-level) results file for the folder
    
    os.makedirs(inference_options.project_md_formatted_results_dir,exist_ok=True)
    md_formatted_results_files_for_chunks = [chunk['md_formatted_results_file'] for chunk in chunk_info]
    md_formatted_results_file_for_folder = os.path.join(
        inference_options.project_md_formatted_results_dir,
        folder_name_clean + '.json').replace('\\','/')
    
    _ = combine_batch_outputs.combine_batch_output_files(md_formatted_results_files_for_chunks,
                                                         md_formatted_results_file_for_folder,
                                                         require_uniqueness=True)
    assert os.path.isfile(md_formatted_results_file_for_folder), \
        'Results file {} does not exist'.format(md_formatted_results_file_for_folder)
    
    
    ##%% Remove low-confidence detections
    
    md_formatted_results_file_for_folder_thresholded = md_formatted_results_file_for_folder.replace(
        '.json','_threshold_{}.json'.format(post_inference_conf_thres))
    
    with open(md_formatted_results_file_for_folder,'r') as f:
        d_before_thresholding = json.load(f)
    
    n_detections_before_thresholding = 0    
    for im in d_before_thresholding['images']:
        n_detections_before_thresholding += len(im['detections'])

    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = post_inference_conf_thres
    options.overwrite_json_files = True
    
    d_after_thresholding = subset_json_detector_output(md_formatted_results_file_for_folder, 
                                                       md_formatted_results_file_for_folder_thresholded, 
                                                       options, d_before_thresholding)
    
    n_detections_after_thresholding = 0    
    for im in d_after_thresholding['images']:
        n_detections_after_thresholding += len(im['detections'])
      
    print('Thresholding reduced the total number of detections from {} to {}'.format(
        n_detections_before_thresholding,n_detections_after_thresholding))
    
    del d_before_thresholding,d_after_thresholding

    
    ##%% Optionallly perform NMS within each patch
    
    if do_within_patch_nms:
        
        print('Loading merged results file')
        
        with open(md_formatted_results_file_for_folder_thresholded,'r') as f:        
            md_results = json.load(f)
        
        print('Eliminating redundant detections')
        
        in_place_nms(md_results,iou_thres=nms_iou_threshold)
        
        patch_results_after_nms_file = md_formatted_results_file_for_folder_thresholded.replace('.json',
                                                                      '_patch-level_nms.json')
        assert patch_results_after_nms_file != md_formatted_results_file_for_folder_thresholded, \
            'Results file naming convention error'        
        
        with open(patch_results_after_nms_file,'w') as f:
            json.dump(md_results,f,indent=1)

    else:
        
        patch_results_after_nms_file = None
        
    
    ##%% Combine all the patch results to an image-level results set
        
    patch_results_file = md_formatted_results_file_for_folder_thresholded
    
    with open(patch_results_file,'r') as f:
        all_patch_results = json.load(f)
    
    # Map absolute paths to detections; we need this because we used absolute paths
    # to map patches back to images.
    #
    # This contains patches for all images in the folder.
    patch_fn_to_results = {}
    for im in tqdm(all_patch_results['images']):
        abs_fn = os.path.join(patch_folder_for_folder,im['file']).replace('\\','/')
        patch_fn_to_results[abs_fn] = im

    md_results_image_level = {}
    md_results_image_level['info'] = all_patch_results['info']
    md_results_image_level['detection_categories'] = all_patch_results['detection_categories']
    md_results_image_level['images'] = []
    
    image_fn_to_patch_info = { x['image_fn']:x for x in all_image_patch_info }
    
    # i_image = 0; image_fn_relative = images_relative[i_image]
    for i_image,image_fn_relative in tqdm(enumerate(images_relative),total=len(images_relative)):
        
        image_fn = os.path.join(input_folder_base,image_fn_relative).replace('\\','/')
        assert os.path.isfile(image_fn), 'Image file {} doesn\'t exist'.format(image_fn)
                
        output_im = {}
        output_im['file'] = image_fn_relative
        output_im['detections'] = []
            
        pil_im = vis_utils.open_image(image_fn)
        if not inference_options.allow_variable_image_size:
            assert pil_im.size[0] == expected_image_width, 'Illegal image size'
            assert pil_im.size[1] == expected_image_height, 'Illegal image size'
        
        image_w = pil_im.size[0]
        image_h = pil_im.size[1]
        
        output_im['w'] = image_w
        output_im['h'] = image_h
        
        image_patch_info = image_fn_to_patch_info[image_fn]
        assert image_patch_info['patches'][0]['image_fn'] == image_fn, 'Image/patch mapping error'
        
        # Patches just for this image
        patch_fn_to_patch_info_this_image = {}
        
        for patch_info in image_patch_info['patches']:
            patch_fn_to_patch_info_this_image[patch_info['patch_fn']] = patch_info
                
        # For each patch
        # i_patch = 0; patch_fn = list(patch_fn_to_patch_info_this_image.keys())[i_patch]
        for i_patch,patch_fn in enumerate(patch_fn_to_patch_info_this_image.keys()):
            
            patch_results = patch_fn_to_results[patch_fn]
            patch_info = patch_fn_to_patch_info_this_image[patch_fn]
            
            # patch_results['file'] is a relative path, and a subset of patch_info['patch_fn']
            assert patch_results['file'] in patch_info['patch_fn'], 'Patch results lookup error'
            
            patch_w = (patch_info['xmax'] - patch_info['xmin']) + 1
            patch_h = (patch_info['ymax'] - patch_info['ymin']) + 1
            assert patch_w == patch_size[0], 'Illegal patch size'
            assert patch_h == patch_size[1], 'Illegal patch size'
            
            # det = patch_results['detections'][0]
            for det in patch_results['detections']:
            
                bbox_patch_relative = det['bbox']
                xmin_patch_relative = bbox_patch_relative[0]
                ymin_patch_relative = bbox_patch_relative[1]
                w_patch_relative = bbox_patch_relative[2]
                h_patch_relative = bbox_patch_relative[3]
                
                # Convert from patch-relative normalized values to image-relative absolute values
                w_pixels = w_patch_relative * patch_w
                h_pixels = h_patch_relative * patch_h
                xmin_patch_pixels = xmin_patch_relative * patch_w
                ymin_patch_pixels = ymin_patch_relative * patch_h
                xmin_image_pixels = patch_info['xmin'] + xmin_patch_pixels
                ymin_image_pixels = patch_info['ymin'] + ymin_patch_pixels
                
                # ...and now to image-relative normalized values
                w_image_normalized = w_pixels / image_w
                h_image_normalized = h_pixels / image_h
                xmin_image_normalized = xmin_image_pixels / image_w
                ymin_image_normalized = ymin_image_pixels / image_h
                
                bbox_image_normalized = [xmin_image_normalized,
                                         ymin_image_normalized,
                                         w_image_normalized,
                                         h_image_normalized]
                
                output_det = {}
                output_det['bbox'] = bbox_image_normalized
                output_det['conf'] = det['conf']
                output_det['category'] = det['category']
                
                output_im['detections'].append(output_det)
                
            # ...for each detection
            
        # ...for each patch

        md_results_image_level['images'].append(output_im)
        
    # ...for each image    
    
    os.makedirs(inference_options.project_image_level_results_dir,exist_ok=True)
    
    md_results_image_level_fn = os.path.join(inference_options.project_image_level_results_dir,
                                             folder_name_clean + '_md_results_image_level.json').replace('\\','/')
    print('Saving image-level results to {}'.format(md_results_image_level_fn))
          
    with open(md_results_image_level_fn,'w') as f:
        json.dump(md_results_image_level,f,indent=1)


    ##%% Perform image-level NMS
    
    in_place_nms(md_results_image_level,iou_thres=nms_iou_threshold)
    
    md_results_image_level_nms_fn = md_results_image_level_fn.replace('.json','_nms.json')
    
    print('Saving image-level results (after NMS) to {}'.format(md_results_image_level_nms_fn))
    
    with open(md_results_image_level_nms_fn,'w') as f:
        json.dump(md_results_image_level,f,indent=1)


    ##%% Clean up

    """
    For all the things we're supposed to be cleaning up, before we delete a bunch of stuff
    we worked hard to generate, make sure the files we're deleting look like what we expect.    
    """
    
    execute_cleanup = True
    
    def safe_delete(fn,verbose=True):
        
        if fn is None or len(fn) == 0:
            return
        
        try:
            if os.path.isfile(fn):
                if verbose:
                    print('Cleaning up file {}'.format(fn))
                if execute_cleanup:
                    os.remove(fn)
            elif os.path.isdir(fn):
                if verbose:
                    print('Cleaning up folder {}'.format(fn))
                if execute_cleanup:
                    shutil.rmtree(fn)
                    pass
            else:
                print('Skipping cleanup of {}, does not exist'.format(fn))
        except Exception as e:
            print('Error cleaning up {}: {}'.format(fn,str(e)))
                
    if 'patch_cache_file' in inference_options.cleanup_targets:
        safe_delete(patch_cache_file)
    else:
        print('Bypassing cleanup of patch cache file')
    
    if 'chunk_cache_file' in inference_options.cleanup_targets:
        safe_delete(chunk_cache_file)
    else:
        print('Bypassing cleanup of chunk cache file')
        
    if 'dataset_files' in inference_options.cleanup_targets:
        if os.path.isdir(folder_dataset_file_dir):
            dataset_files = os.listdir(folder_dataset_file_dir)
            assert all([fn.endswith('.yaml') for fn in dataset_files]), 'dataset file lookup error'
            safe_delete(folder_dataset_file_dir)        
    else:
        print('Bypassing cleanup of dataset files')
        
    if 'patch_level_results' in inference_options.cleanup_targets:
        safe_delete(md_formatted_results_file_for_folder)
        safe_delete(md_formatted_results_file_for_folder_thresholded)
        safe_delete(patch_results_after_nms_file)
    else:
        print('Bypassing cleanup of patch-level results')
        
    if 'inference_scripts' in inference_options.cleanup_targets:
        if os.path.isdir(folder_inference_script_dir):
            inference_scripts = os.listdir(folder_inference_script_dir)
            assert all([(fn.endswith('.sh') or fn.endswith('.bat')) for fn in inference_scripts]), \
                'Inference script not found'
            safe_delete(folder_inference_script_dir)
    else:
        print('Bypassing cleanup of inference scripts')
    
    if 'patches' in inference_options.cleanup_targets:
        if os.path.isdir(patch_folder_for_folder):            
            safe_delete(patch_folder_for_folder)
    else:
        print('Bypassing cleanup of patches')
        
    if 'symlink_images' in inference_options.cleanup_targets:
        if os.path.isdir(folder_symlink_dir):
            # These are either folders called "chunk_00" or yolov5 cache files called "chunk_00.cache"
            symlink_folders_and_cache_files = os.listdir(folder_symlink_dir)
            assert all([fn.startswith('chunk') for fn in symlink_folders_and_cache_files]), \
                'symlink folder validation error during cleanup'
            safe_delete(folder_symlink_dir)
    else:
        print('Bypassing cleanup of symlink folder')
        
    if 'yolo_results' in inference_options.cleanup_targets:
        if os.path.isdir(folder_yolo_results_dir):
            yolo_results_folders = os.listdir(folder_yolo_results_dir)
            assert all([os.path.isdir(os.path.join(folder_yolo_results_dir,fn)) for \
                        fn in yolo_results_folders]), \
                        'YOLO results folder validation error during cleanup'
            assert all([fn.startswith('inference-output') for fn in yolo_results_folders]), \
                'Inference output folder validation error during cleanup'
            safe_delete(folder_yolo_results_dir)
    else:
        print('Bypassing cleanup of YOLO-formatted results')              
    
    # Reserving this for future use, but right now it would be silly to delete this
    if 'image_level_results' in inference_options.cleanup_targets:
        safe_delete(md_results_image_level_fn)
    else:
        print('Bypassing cleanup of image-level results')
    
    
    ##%% Prepare return values
    
    to_return = {}
    to_return['md_formatted_results_file_for_folder_thresholded'] = \
        md_formatted_results_file_for_folder_thresholded
    to_return['md_results_image_level_fn'] = \
        md_results_image_level_fn
    to_return['md_results_image_level_nms_fn'] = \
        md_results_image_level_nms_fn
    
    ##%%
    
    return to_return
    
# ...run_model_on_folder()


#%% Interactive driver

if False:
    
    pass

    #%% Run the model programmatically on one folder
    
    input_folder_base = r'c:\temp\usgs-test-images'
    
    inference_options = USGSGeeseInferenceOptions()
    
    results = run_model_on_folder(input_folder_base,recursive=True)
    