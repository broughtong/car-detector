Self-learning Car Detector

# Arranging the data
Place the rosbags into data/rosbags
Place the gt into data/gt

# Set-up your system
You will need to have tf_bag package installed and sourced, from here: `https://github.com/IFL-CAMP/tf_bag`
For the rest of the dependencies, there are conda environments provided in the conda folder.
The ros environment is used for everything, except for running lanoising
You can create the environments like this: `mamba env create --file lanoise.yml`

# Extract the rosbags
Run preprocessing/rosbag_extractor.py to extract the rosbags
Run preprocessing/tf_extractor.py to extract the transforms

# Perform simple car detection
Run supervisor/detector.py to run the method

# Prepare Annotations
Run supervisor/temporal.py to run the temporal filter

You will need the weather simulator network models, explained on the simulator page.
The origal page is here https://github.com/cavayangtao/lanoising
And a mirror is provided here: https://github.com/broughtong/lanoising
Copy the models into supervisor/lanoising/models/

Switch to the lanoise conda environment.
For me, I need to make sure PYTHONPATH is blank, activate the conda environment, and then source ros, in that order.
Run the lanoising weather simulator in supervisor/lanoising/lanoising.py

Run supervisor/annotator.py to create annotations.
Run supervisor/datasetDivision.py to break the images into training/testing/evaluation subset by rosbag (all bags with gt go to eval)

Run networks/generate_data.py
Train pointnet using python train_segmentation.py and python train_segmentation.py --lanoise
Optionally add the feature transform flag --feature_transform
Train unet using python trian_unet.py
Optional flags --num_channels 3 (1/2/3) --lanoise and --numc 2 (2/3)

# Train the networks
The MaskRCNN network can be trained by running networks/maskrcnn/training.py
Alternatively, you can run the files in network/ to train the other networks.



## Evaluation 

For pointnet use networkEval.py
Make sure the feature transform flag is set correctly.
For each network, go into the folder and run the associated evaluation script.
To generate the graphs, run the scripts in the evaluation folder.

