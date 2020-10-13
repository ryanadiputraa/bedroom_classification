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
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')


# create clean room dir on data training dir
train_clean_dir = os.path.join(train_dir, 'clean')

# create messy room dir on data training dir
train_messy_dir = os.path.join(train_dir, 'messy')

# create clean room dir on data validation dir
validation_clean_dir = os.path.join(validation_dir, 'clean')

# create messy room dir on data validation dir
validation_messy_dir = os.path.join(validation_dir, 'messy')


# image augmentation
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=20,
  horizontal_flip=True,
  shear_range=.2,
  fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=20,
  horizontal_flip=True,
  shear_range=.2,
  fill_mode='nearest'
)


train_generator = train_datagen.flow_from_directory(
  train_dir,   # train data dir
  target_size=(150, 150),   # change all images resolution to 150x150 pixels
  batch_size=4,
  class_mode='binary'   # 2 class classification so its binary
)

validation_generator = test_datagen.flow_from_directory(
  validation_dir,   # train data dir
  target_size=(150, 150),   # change all images resolution to 150x150 pixels
  batch_size=4,
  class_mode='binary'   # 2 class classification so its binary
)