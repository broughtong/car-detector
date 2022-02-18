Self-learning Car Detector

# Arranging the data
Place the rosbags into data/rosbags
Place the gt into data/gt

# Extract the rosbags
Run preprocessing/rosbag_extractor.py to extract the rosbags
Run preprocessing/tf_extractor.py to extract the transforms

# Perform simple car detection
Run supervisor/detector.py to run the method
To evaluate the results of this only, run evaluation/simple.py

# Prepare Annotations
Run supervisor/temporal.py to run the temporal filter
If you wish to analyse the results of the temporal filter, run evaluation/temporal.py

If you wish to run the weather simulate on the results, run supervisor/augment.py
Run supervisor/annotator.py to create annotations.
Run supervisor/datasetDivision.py to break the images into training/evaluation subset by rosbag

# Train the networks
The MaskRCNN network can be trained by running networks/maskrcnn/training.py
The networks can be trained by running networks/
