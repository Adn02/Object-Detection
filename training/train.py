from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os

IMAGE_SIZE = 96
NUM_CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 2

BASE_DIR = 'C:/Users/aidan_000/Desktop/UNCC/Github/proj1-detection-adn02/Images'
TARGET_CLASS_DIR = 'C:/Users/aidan_000/Desktop/UNCC/Github/proj1-detection-adn02/Images/Positives'
NON_TARGET_CLASS_DIR = 'C:/Users/aidan_000/Desktop/UNCC/Github/proj1-detection-adn02/Images/Negatives'

target_images = [os.path.join(TARGET_CLASS_DIR, img) for img in os.listdir(TARGET_CLASS_DIR)]
non_target_images = [os.path.join(NON_TARGET_CLASS_DIR, img) for img in os.listdir(NON_TARGET_CLASS_DIR)]

target_labels = [1]*len(target_images)
non_target_labels = [0]*len(non_target_images)

images = target_images + non_target_images
labels = target_labels + non_target_labels

images, labels = shuffle(images, labels)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)

train_images = [image.img_to_array(image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))) for img_path in train_images]
val_images = [image.img_to_array(image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))) for img_path in val_images]

train_images = np.array(train_images)
train_labels = np.array(train_labels)

val_images = np.array(val_images)
val_labels = np.array(val_labels) 

# Convert labels to categorical after they are split and before they are used in the model
train_labels = to_categorical(np.array(train_labels))
val_labels = to_categorical(np.array(val_labels))

datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=40,  # Rotate by up to 40 degrees
    width_shift_range=0.2,  # Shift width by up to 20%
    height_shift_range=0.2,  # Shift height by up to 20%
    zoom_range=0.2,  # Zoom in by up to 20%
    horizontal_flip=True,  # Flip horizontally
    shear_range=0.2,  # Apply shear transformation
    brightness_range=[0.5, 1.5],  # Adjust brightness between 0.5 to 1.5 times
    channel_shift_range=50,  # Shift color channels by up to 50
    validation_split=0.2,  # Split the data into training and validation sets
)

# Load training images
train_generator = datagen.flow(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    subset='training',  # Use the training subset
    shuffle=True,  # Shuffle the training images
)

# Load validation images
validation_generator = datagen.flow(
    val_images,
    val_labels,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=True,  # Do not shuffle the validation set
)

def objectNet():
    # Mobilenet parameters
    input_shape = [IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS] # resized to 96x96 per EEMBC requirement # was [96,96,3]
    num_classes = NUM_CLASSES # plane and non-plane
    num_filters = 8 # normally 32, but running with alpha=.25 per EEMBC requirement

    inputs = Input(shape=input_shape)
    x = inputs # Keras model uses ZeroPadding2D()

    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten layer
    x = Reshape((-1,))(x)
    
    # Dense layers
    x = Dense(16, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Sigmoid activation for binary classification
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate, steps_per_epoch=None):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  if steps_per_epoch is None:
    steps_per_epoch = len(train_generator)
  history_fine = model.fit(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE)
  return model

epochs = 20
lr = .001

model = objectNet()
model = train_epochs(model, train_generator, validation_generator, epochs, lr)

model.summary()

# model.save('objectNet.h5')
