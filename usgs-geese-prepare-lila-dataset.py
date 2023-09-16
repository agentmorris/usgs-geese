########
#
# usgs-geese-prepare-lila-dataset.py
#
# Prepare a public (re-)release of the images and annotation used in training.
#
########

#%% Imports and constants

import os
import json
import copy
import shutil

import humanfriendly

from tqdm import tqdm
from collections import defaultdict

from md_visualization import visualization_utils as vis_utils

input_file = os.path.expanduser('~/data/usgs_geese.json')
output_dir = os.path.expanduser('~/data/usgs-geese-lila')
output_file = os.path.join(output_dir,'izembek-lagoon-birds.json')
output_dir_images = os.path.join(output_dir,'images')

drive_root = '/media/user/My Passport'
image_base = os.path.join(drive_root,'2017-2019/01_JPGs')
annotation_base = os.path.join(drive_root,'2017-2019/03_Manually_corrected_annotations')

os.makedirs(output_dir_images,exist_ok=True)
assert os.path.isfile(input_file)
assert os.path.isdir(image_base)
assert os.path.isdir(annotation_base)


#%% Read input data

with open(input_file,'r') as f:
    d = json.load(f)


#%% Validate input data and compute total size

total_size_bytes_positives = 0

image_id_to_annotations = defaultdict(list)
for ann in d['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)

# First the positive images

# im = d['images'][0]
for im in tqdm(d['images']):
    image_id = im['id']    
    assert image_id in image_id_to_annotations
    assert len(image_id_to_annotations[image_id]) > 0
    
    input_fn_abs = os.path.join(image_base,im['file_name'])
    assert os.path.isfile(input_fn_abs)    
    total_size_bytes_positives += os.path.getsize(input_fn_abs)
    
# ...for each non-empty image

del im,ann

# Now the negatives
max_non_empty_id = max([cat['id'] for cat in d['categories']])
empty_id = max_non_empty_id + 1

empty_images = []
empty_annotations = []

total_size_bytes_negatives = 0

for i_fn,fn_csv in tqdm(enumerate(d['empty_images']),total=len(d['empty_images'])):
    
    # E.g.:
    # '2017/Replicate_2017-09-30/Cam1/293A0036_i0022.csv'
    
    fn_json = fn_csv.replace('.csv','.json')
    fn_csv_abs = os.path.join(annotation_base,fn_csv)
    fn_json_abs = os.path.join(annotation_base,fn_json)
    
    assert os.path.isfile(fn_csv_abs)
    assert os.path.isfile(fn_json_abs)
        
    dirname = os.path.dirname(fn_csv)
    basename = os.path.basename(fn_csv)
    fn_jpg = os.path.join(dirname,basename.split('_')[0] + '.JPG')
    
    im = {}
    im['file_name'] = fn_jpg
    im['id'] = im['file_name'].replace('/','_')
    im['annotation_file'] = fn_json
    
    fn_jpg_abs = os.path.join(image_base,im['file_name'])    
    
    # Occasionally we have an empty annotation file and a non-empty annotation file for the
    # same image file.  Don't call those empty.
    if im['id'] in image_id_to_annotations:
        continue
    
    pil_im = vis_utils.open_image(fn_jpg_abs)
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]
    
    im['width'] = image_w
    im['height'] = image_h
    
    ann = {}
    ann['image_id'] = im['id']
    ann['id'] = im['id']
    ann['category_id'] = empty_id
    
    empty_annotations.append(ann)
    empty_images.append(im)    
    image_id_to_annotations[im['id']] = [ann]
    
    assert os.path.isfile(fn_jpg_abs)    
    total_size_bytes_negatives += os.path.getsize(fn_jpg_abs)
    
# ...for each empty image    
    
total_size_bytes = total_size_bytes_positives + total_size_bytes_negatives

print('Validated {} files totaling {}'.format(
    len(d['images']),humanfriendly.format_size(total_size_bytes)))


#%% Merge the empty/non-empty structs

empty_category = {'id':empty_id,'name':'Empty'}

output_d = copy.deepcopy(d)
output_d['categories'].append(empty_category)
output_d['images'].extend(empty_images)
output_d['annotations'].extend(empty_annotations)

for ann in output_d['annotations']:
    if ann['category_id'] == empty_id:
        assert 'bbox' not in ann
    else:
        assert 'bbox' in ann


#%% Copy images

# Takes ~30 minutes

ids_copied = set()

# im = output_d['images'][0]
for im in tqdm(output_d['images']):
    
    input_fn_abs = os.path.join(image_base,im['file_name'])
    image_id = im['id']
    assert image_id.endswith('.JPG')
    assert image_id not in ids_copied
    ids_copied.add(image_id)    
    output_fn_abs = os.path.join(output_dir_images,image_id)
    shutil.copyfile(input_fn_abs,output_fn_abs)


#%% Convert the .json representation to point to the output data

output_image_base = output_dir

# im = output_d['images'][0]    
for im in output_d['images']:
    output_fn_relative = 'images/' + im['id']
    output_fn_abs = os.path.join(output_image_base,output_fn_relative)
    assert os.path.isfile(output_fn_abs)
    im['file_name'] = output_fn_relative
    

#%% Write the output file

with open(output_file,'w') as f:
    json.dump(output_d,f,indent=1)
    

#%% Check DB integrity

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = output_dir
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sorted_categories, _, _= integrity_check_json_db.integrity_check_json_db(output_file, options)

#%%

"""
424790 Brant
 47561 Canada
 41275 Other
  5631 Gull
  4281 Empty
  2013 Emperor
"""

#%% Preview some images

from md_visualization import visualize_db
from md_utils.path_utils import open_file

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 20
viz_options.trim_to_images_with_bboxes = True
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

html_output_file, _ = visualize_db.process_images(db_path=output_file,
                                                    output_dir=os.path.join(output_dir,'preview'),
                                                    image_base_dir=output_dir,
                                                    options=viz_options)
open_file(html_output_file)

