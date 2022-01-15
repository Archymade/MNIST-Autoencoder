import tensorflow as tf
from tensorflow.keras import utils

def load_dataset(input_shape, DIR, batch_size):
    ### Import dataset
    train_images = utils.image_dataset_from_directory(DIR, image_size = input_shape, color_mode = 'grayscale',
                                                      batch_size = batch_size, shuffle = True, labels = None)

    ### Scale pixel values
    train_images = train_images.map((lambda x: (x/255, x/255)), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    return train_images
