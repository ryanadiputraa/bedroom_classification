import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, zipfile


# print(tf.__version__)
base_path = '/home/ryan/Documents/Data Science Tutorial/Machine Learning/bedroom_image_classification'
local_zip = 'messy-vs-clean-room.zip'
zip_ref = zipfile.ZipFile(os.path.join(base_path, local_zip), 'r')
zip_ref.extractall(base_path)
zip_ref.close()

base_dir = '/home/ryan/Documents/Data Science Tutorial/Machine Learning/bedroom_image_classification/images'
train_dir = os.path.join(base_path, 'train')
validation_dir = os.path.join(base_path, 'val')


# create clean room dir on data training dir
train_clean_dir = os.path.join(train_dir, 'clean')

# create messy room dir on data training dir
train_messy_dir = os.path.join(train_dir, 'messy')

# create clean room dir on data validation dir
validation_clean_dir = os.path.join(validation_dir, 'clean')

# create messy room dir on data validation dir
validation_messy_dir = os.path.join(validation_dir, 'messy')