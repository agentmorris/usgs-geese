## Izembek Brant Goose Detector

### Overview

The code in this repo trains, runs, and evaluates models to detect geese in aerial images, based on the <a href="https://www.usgs.gov/data/aerial-photo-imagery-fall-waterfowl-surveys-izembek-lagoon-alaska-2017-2019">Izembek Lagoon dataset</a> (complete citation [below](#data-source)).  Images are annotated with points, labeled as brant, Canada, gull, emperor, and other.  The goal is accuracy on brant, which is by far the most common class (there are around 400k "brant" points, and less than 100k of everything else combined).

There are around 100,000 images total, about 95% of which contain no geese.  Images are 8688 x 5792.  A typical ground truth image looks like this:

<img src="sample_image.jpg" width="800px;"><br/>

The annotations you can vaguely see as different colors correspond to different species of goose.  Most of this repo operates on 1280x1280 patches that look like this:

<img src="annotated_patch.png" width="800px;"><br/>

### Sample results

Here's a random patch of predictions, but you should <i>never</i> put any stock into a "random" image of results that someone shows you on the Internet:

<img src="sample_results_patch.jpg" width="800px;"><br/>

Maybe all the results really look like that, maybe they don't.  I pinky-swear that the image from which this patch was cropped was not used in training, and that in general the results really do look like this, but... never trust random results on the Internet.

If you want to dig a little deeper, here is a set of patch-level previews for validation data (patches selected from images excluded from training, but from flights that were included in training) or test data (patches selected from flights that were excluded from training):

<https://lila.science/public/usgs-izembek-results/>

### Files

These are listed in roughly the order in which you would use them.

#### usgs-geese-data-import.py

* Match images to annotation files
* Read the original annotations (in the format exported by [CountThings](https://countthings.com/))
* Convert to COCO format
* Do a bunch of miscellaneous consistency checking

#### usgs-geese-training-data-prep.py

* For all the images with at least one annotation, slice into mostly-non-overlapping patches
* Optionally sample hard negatives (I did not end up actually using any hard negatives)
* Split into train/val
* Export to YOLO annotation format

#### usgs-geese-training.py

* Train the model (training happens at the YOLOv5 CLI, but this script documents the commands)
* Run the YOLOv5 validation scripts
* Convert YOLOv5 val results to MD .json format
* Example code to use the MD visualization pipeline to visualize results
* Example code to use the MD inference pipeline to run the trained model

#### usgs-geese-inference.py

* Run inference on a folder of images, which means, for each image:

    * Split the image into overlapping patches
    * Run inference on each patch
    * Resolve redundant detections
    * Convert YOLOv5 output to .json (in MegaDetector format)

#### usgs-geese-postprocessing.py

* Generate patch-level previews from image-level model results
* Generate estimated image-level bird counts from image-level model results (and write to .csv)


## Running the model

This section describes the environment setup and command line process for running inference.  Training is not yet set up to be fully run from the command line, though it's close, and the environment should be the same.  Much of the code and environment is borrowed from [MegaDetector](https://github.com/agentmorris/MegaDetector), a model that does similar stuff for camera trap images.

### Environment setup

#### 1. Install prerequisites: Mambaforge, Git, and NVIDIA stuff

Install prerequisites according to the [MegaDetector instructions for prerequisite setup](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#1-install-prerequisites-mambaforge-git-and-nvidia-stuff).  If you already have Mambaforge, git, and the latest NVIDIA driver installed, nothing to see here.

#### 2. Download the model file and dataset definition file

Download the [Izembek bird detector](https://github.com/agentmorris/usgs-geese/releases/download/v1.0.0/usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt) and associated [dataset definition file]() to your computer.  These instructions will assume that you have downloaded the model to a folder called "c:\izembek-brant" (on Windows) or to a folder called "izembek-brant" within your home folder (on Linux/Mac), but if you put it somewhere else, that's fine, just be sure to change it in the steps below that point to a model file.

#### 3. Clone the relevant git repos and add them to your path, and set up your Python environment

You will need the contents of three git repos to make everything work: this repo (usgs-geese), the [YOLOv5](https://github.com/ultralytics/yolov5) repo, and the [MegaDetector](https://github.com/agentmorris/MegaDetector).  You will also need to set up a Python environment with all the Python packages that our code depends on.  In this section, we provide <a href="#windows-instructions-for-gitpython-stuff">Windows</a>, <a href="#linux-instructions-for-gitpython-stuff">Linux</a>, and <a href="#mac-instructions-for-gitpython-stuff">Mac</a> instructions for doing all of this stuff.

##### Windows instructions for git/Python stuff

The first time you set all of this up, open your Mambaforge prompt, and run:

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/agentmorris/MegaDetector
git clone https://github.com/agentmorris/usgs-geese
git clone https://github.com/ultralytics/yolov5/
cd c:\git\usgs-geese
mamba env create --file environment-inference.yml
mamba activate usgs-geese-inference
set PYTHONPATH=%PYTHONPATH%;c:\git\MegaDetector;c:\git\yolov5;c:\git\usgs-geese
```

<a name="windows-new-shell"></a>
Your environment is set up now!  In the future, when you open your Mambaforge prompt, you only need to run:

```batch
cd c:\git\usgs-geese
mamba activate usgs-geese-inference
set PYTHONPATH=%PYTHONPATH%;c:\git\MegaDetector;c:\git\yolov5;g:\git\usgs-geese
```

Pro tip: if you have administrative access to your machine, rather than using the "set PYTHONPATH" steps, you can also create a permanent PYTHONPATH environment variable.  Here's a [good page](https://www.computerhope.com/issues/ch000549.htm) about editing environment variables in Windows.  But if you just want to "stick to the script" and do it exactly the way we recommend above, that's fine.

You can also install the requirements via pip (we're still using mamba here to create a Python environment, but we're using pip to install all the dependencies):

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/agentmorris/MegaDetector
git clone https://github.com/agentmorris/usgs-geese
git clone https://github.com/ultralytics/yolov5/
cd c:\git\usgs-geese
mamba create -n usgs-geese-inference python=3.11
mamba activate usgs-geese-inference
mamba install -y pip
pip install -r requirements.txt
set PYTHONPATH=%PYTHONPATH%;c:\git\MegaDetector;c:\git\yolov5;c:\git\usgs-geese
```

##### Linux instructions for git/Python stuff

If you have installed Mambaforge on Linux, you are probably always at an Mambaforge prompt; i.e., you should see "(base)" at your command prompt.  Assuming you see that, the first time you set all of this up, and run:

```batch
mkdir ~/git
cd ~/git
git clone https://github.com/agentmorris/MegaDetector
git clone https://github.com/agentmorris/usgs-geese
git clone https://github.com/ultralytics/yolov5/
cd ~/git/usgs-geese
mamba env create --file environment-inference.yml
mamba activate usgs-geese-inference
export PYTHONPATH="$PYTHONPATH:$HOME/git/MegaDetector:$HOME/git/yolov5:$HOME/git/usgs-geese"
```

<a name="linux-new-shell"></a>
Your environment is set up now!  In the future, whenever you start a new shell, you just need to do:

```batch
cd ~/git/MegaDetector
mamba activate usgs-geese-inference
export PYTHONPATH="$PYTHONPATH:$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Pro tip: rather than updating your PYTHONPATH every time you start a new shell, you can add the "export" line to your .bashrc file.

You can also install the requirements via pip (we're still using mamba here to create a Python environment, but we're using pip to install all the dependencies):

```batch
mkdir ~/git
cd ~/git
git clone https://github.com/agentmorris/MegaDetector
git clone https://github.com/agentmorris/usgs-geese
git clone https://github.com/ultralytics/yolov5/
cd ~/git/usgs-geese
mamba create -n usgs-geese-inference python=3.11
mamba activate usgs-geese-inference
mamba install -y pip
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$HOME/git/MegaDetector:$HOME/git/yolov5:$HOME/git/usgs-geese"
```

### Actually running the model

You can run the model with [run_izembek_model.py](run_izembek_model.py).  First, when you open a new Mambaforge prompt, don't forget to do this (on Windows):

```batch
cd c:\git\usgs-geese
mamba activate usgs-geese-inference
set PYTHONPATH=%PYTHONPATH%;c:\git\MegaDetector;c:\git\yolov5;g:\git\usgs-geese
```

...or this (on Linux):

```batch
cd ~/git/MegaDetector
mamba activate usgs-geese-inference
export PYTHONPATH="$PYTHONPATH:$HOME/git/MegaDetector:$HOME/git/yolov5"
```

Then you can run the script like this (using Windows syntax), substituting real paths for all the arguments:

```batch
python run-izembek-model.py [MODEL_LOCATION] [IMAGE_FOLDER] [YOLO_FOLDER] [SCRATCH_FOLDER] --recursive --no_use_symlinks
```

* MODEL_LOCATION is the location of the .pt you downloaded earlier
* IMAGE_FOLDER is the root folder of all the images you want to process (recursively, if you specify "--recursive")
* YOLO_FOLDER is the folder where you checked out the YOLOv5 repo, e.g. "c:\git\yolov5"
* SCRATCH_FOLDER is a folder you have permission to write to, on a drive that has at least twice as much free space as the size of the image folder

The "--no_use_symlinks" argument tells the script not to attempt symlink creation; currently this results in more temporary disk space usage.  Typically this is useful on Windows machines where you don't have admin privileges.

If you have a GPU, and it's being utilized correctly, near the beginning of the output, you should see:

`GPU available: True`

If you have an Nvidia GPU, and it's being utilized correctly, near the beginning of the output, you should see:

`GPU available: True`

If you have an Nvidia GPU and you see "GPU available: False", your GPU environment may not be set up correctly.  95% of the time, this is fixed by <a href="https://www.nvidia.com/en-us/geforce/drivers/">updating your Nvidia driver"</a> and rebooting.  If you have an Nvidia GPU, and you've installed the latest driver, and you've rebooted, and you're still seeing "GPU available: False", <a href="mailto:agentmorris+izembek@gmail.com">email me</a>.

### Where do the results go?

If everything worked correctly, in your scratch folder, there will be a folder called "image_level_results".  Within that, look for a file that looks like:

`something_something_something_md_results_image_level_nms.json`

The first bit (something something something) corresponds to the folder name you just processed.  The idea is that you will use the same scratch folder every time, so this part gives the results files a unique name.  This .json file contains the locations of all detections, in the [MegaDetector output format](https://github.com/agentmorris/MegaDetector/tree/main/api/batch_processing#megadetector-batch-output-format).

### Previewing the results

TODO

### Generating counts

TODO

## Data source

All images are sampled from:

Weiser, E. L., Flint, P. L., Marks, D. K., Shults, B. S., Wilson, H. M., Thompson, S. J. and Fischer, J. B., 2022, Aerial photo imagery from fall waterfowl surveys, Izembek Lagoon, Alaska, 2017-2019: U.S. Geological Survey data release, <a href="https://doi.org/10.5066/P9UHP1LE">https://doi.org/10.5066/P9UHP1LE</a>.


## Major open items

* Move this open items list to GitHub issues
* Populate the "previewing the results" and "generating counts" sections in this README
* Add checkpointing, currently you lose the whole result set if your job crashes
* Clean up the extensive scratch space use, especially on Windows, where we create patches, then copy all of those patches