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
Run supervisor/annotator.py to create annotations