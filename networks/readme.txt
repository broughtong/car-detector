generate_data_carloc.py
- for generating h5py files from pickle files
- example code is in the bottom
- loop through bags that should be regenerated (in code array "bags")
	- create instance of DataGenerator(path to bag, target name)
	- run class function generate()

merge_bags.py
- for merging multiple h5py files
- loop through bags that should be regenerated (in code array "bags")
	- open every bag with function load_h5 which opens the h5py files
	- stack the data
- after all data are stacked, call save_h5



%-------------------------------------------------------------------------------------
PointNet
- requirements should be only pytorch and h5py

script train_segmentation.py
- on line 51 change path to the dataset
- then run the script with hyperparameters
- for me works well parameters: --bs 128 --lr 0.0001  --normalize --feature_transform 
	- I leave others (except gpu index) default
%-------------------------------------------------------------------------------------
UNet
- requirements should be only pytorch and h5py

script train_unet.py
- on line 56 change path to the dataset
- then run the script with hyperparameters
- parameters that works well: --lr 0.001  --bs 16 (or 32 if GPU can take it)
- note that from losses it's impossible to determine best model
- you can uncomment calculating confidence matrix (line 100 for training dataset or sth like that)
- I determined the best model from confidence matrix on both training and validation datasets

%------------------------------------------------------------------------------------7
if you  want to use validation data, uncomment paths to it below the path to training data
- and uncomment the validation part in training loop (Unet - line 109, PointNet - line 123)

