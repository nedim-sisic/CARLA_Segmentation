
import numpy as np
import os
import tensorflow as tf

# Import network layers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D


# Directories containing images and segmentations
images_path = ''
segmentations_path = ''

# Lists of image and segmentation paths 
image_list = [images_path + filename for filemane in os.listdir(images_path)]
segmentation_list = [segmentations_path + filename for filename in os.listdir(segmentations_path)]

# Create dataset tensors
image_filenames = tf.constant(image_list)
segmentation_filenames = tf.constant(segmentation_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, segmentation_filenames))

# Resize and normalize image and segmentation
def resize_and_normalize(image, mask):

    resized_image = tf.image.resize(image, [96, 128], method='nearest')
    resized_mask = tf.image.resize(mask, [96, 128], method='nearest')
    
    # Normalize the image to [0, 1] range
    normalized_image = resized_image / 255.0
    
    return normalized_image, resized_mask


# Load and process image and mask files
def load_and_prepare(image_fp, mask_fp):

    # Read and decode image file
    image = tf.image.decode_png(tf.io.read_file(image_fp), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Read, decode, and process mask file
    mask = tf.image.decode_png(tf.io.read_file(mask_fp), channels=3)
    mask = tf.reduce_max(mask, axis=-1, keepdims=True)
    
    return image, mask

# Apply transformations to dataset
image_dataset = dataset.map(load_and_prepare)
processed_dataset = image_dataset.map(resize_and_normalize)

# Convolutional block
def conv_block(inputs=None, n_filters=32, max_pooling=True):

    conv = Conv2D(n_filters, 3, kernel_initializer= 'he_normal', padding='same', activation='relu')(inputs)
    conv = Conv2D(n_filters, 3, kernel_initializer= 'he_normal', padding='same', activation='relu')(conv)
                 
    if max_pooling:
        next_layer = MaxPooling2D(2, strides=2)(conv)        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


# Upsampling block
def upsampling_block(downsampling_input, upsamling_input, n_filters=32):
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(upsamling_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, downsampling_input], axis=3)
    
    conv = Conv2D(n_filters, 3, kernel_initializer= 'he_normal', padding='same', activation='relu')(merge)
    conv = Conv2D(n_filters, 3, kernel_initializer= 'he_normal', padding='same', activation='relu')(conv)
    
    return conv


# Create model
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)

    # Donwsampling path
    down_block1 = conv_block(inputs=inputs, n_filters=n_filters*1)
    down_block2 = conv_block(inputs=down_block1[0], n_filters=n_filters*2)
    down_block3 = conv_block(inputs=down_block2[0], n_filters=n_filters*4)
    down_block4 = conv_block(inputs=down_block3[0], n_filters=n_filters*8)
    down_block5 = conv_block(inputs=down_block4[0], n_filters=n_filters*16, max_pooling=False)
    
    # Upsampling path
    up_block6 = upsampling_block(down_block5[0], down_block4[1], n_filters*8)    
    up_block7 = upsampling_block(up_block6, down_block3[1], n_filters*4)
    up_block8 = upsampling_block(up_block7, down_block2[1], n_filters*2)
    up_block9 = upsampling_block(up_block8, down_block1[1], n_filters*1)

    conv9 = Conv2D(n_filters, 3, kernel_initializer= 'he_normal', padding='same', activation='relu')(up_block9)

    # Classification layer
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# Create model
unet = unet_model((96, 128, 3))
unet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train model
processed_dataset.batch(32)
train_dataset = processed_dataset.cache().shuffle(200).batch(32)
model_history = unet.fit(train_dataset, epochs=20)