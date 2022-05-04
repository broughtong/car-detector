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
If you wish to analyse the results of the temporal filter, run evaluation/temporal.py

You will need the weather simulator network models, explained on the simulator page.
The origal page is here https://github.com/cavayangtao/lanoising
And a mirror is provided here: https://github.com/broughtong/lanoising
Copy the models into supervisor/lanoising/models/

Run the lanoising weather simulator in supervisor/lanoising/lanoising.py
I dumped the conda environment into the env.yaml file
You can load it with mamba env create -f env.yaml

Run supervisor/annotator_mask.py to create annotations.
Run supervisor/datasetDivision.py to break the images into training/testing/evaluation subset by rosbag (all bags with gt go to eval)

# Train the networks
The MaskRCNN network can be trained by running networks/maskrcnn/training.py
The networks can be trained by running networks/
