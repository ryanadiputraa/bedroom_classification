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


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compiling, set lost function and optimizer
model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])

# train model
model.fit(
  train_generator,
  steps_per_epoch=25,  # how much batch will execute on each epoch
  epochs=20,
  validation_data=validation_generator,  # show accuracy on data validation testing
  validation_steps=5,   # how much batch will execute on each epoch
  verbose=2
)
