# Data Preprocessing 
!!! tip
	Run Neural Network Models on GPU for faster results

Each layer of of the network can comprise of one of the following: `Dense`, `Conv2D`, `MaxPool2D` or `Flatten`, etc.

```python
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import tensorflow as tf
import pandas as pd
```

Datasets can be found on [kaggle](https://www.kaggle.com/), to unzip these datasets, use the below code

```python
# Source: Stackoverflow
def unzip_file(filename):
	"""@param filename(str): Path to the .zip file"""
	zip_ref = zipfile.ZipFile(filename, 'r')
	zip_ref.extractall()
	zip_ref.close()

unzip(<path_to_your_zip_file>)
```

## Data Augmentation

It is not practically possible to collect huge amount of data whenever required, which is why we can modify the existing image data using certain techniques (changing brightness, rotation, etc) to create new data for the training.

!!! warning
	The validation data must be from the subset of original data, while the augmented data and the unused original data must be used for training

```python
# 1/255 for rescale as its better to get everything into 0 to 1 range - RGB Value
train_data_gen = idg(rescale=1/255, rotation_range=37, width_shift_range=0.10, height_shift_range=0.21, validation_split=0.2)

# Reshape every file to target size, better to keep it at 256 or close to 300
train_data = train_data_gen.flow_from_directory("<directory_of_samples>", target_size=(256, 256), subset="training") 
validation_data = train_data_gen.flow_from_directory("<directory_of_samples>", target_size=(256, 256), subset="validation") 
```
