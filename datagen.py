import sys
from unicodedata import category

import tensorflow as tf
import numpy as np

from common import *

cache = {}


# https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
# https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ClockImageDataGen(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = 32, input_size=(128, 128, 3), shuffle=True, to_fit=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.to_fit = to_fit

        self.mode = Mode.SCALED
        
        self.n = len(self.df)
        self.n_hours = 24
        self.n_minutes = 60
        self.n_time = 24 * 60

    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    

    def __get_input(self, path, target_size):
        if USE_CACHE and path in cache:
            return cache[path]
        else:
            image = tf.keras.preprocessing.image.load_img(path)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

            scaled = image_arr / 255

            if USE_CACHE: 
                if sys.getsizeof(cache) > 1048000:
                    cache.clear()
                cache[path] = scaled
                
            return scaled
    

    def __get_output(self, label, num_classes):
        #print(label, num_classes, tf.keras.utils.to_categorical(label, num_classes=num_classes))
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batchs):
        # Generates data containing batch_size samples

        path_batch = batchs['filepath']
        hour_batch = batchs['hour']
        minute_batch = batchs['minute']
        time_batch = batchs['time']

        X_batch = np.array([self.__get_input(x, self.input_size) for x in path_batch])

        X = X_batch

        #y_batch = np.array(norm_batch).reshape(-1, 1)
        if self.mode == Mode.CATEGORICAL:
            y0_batch = np.asarray([self.__get_output(y, self.n_hours) for y in hour_batch])
            y1_batch = np.asarray([self.__get_output(y, self.n_minutes) for y in minute_batch])
            y = tuple([y0_batch, y1_batch])
        elif self.mode == Mode.SCALED:
            y0_batch = hour_batch.to_numpy() / self.n_hours
            y1_batch = minute_batch.to_numpy() / self.n_minutes
            y = tuple([y0_batch, y1_batch])
        elif self.mode == Mode.TIME_LABEL:
            y0_batch = np.asarray([self.__get_output(y, self.n_time) for y in time_batch])
            y = y0_batch
        
        if self.to_fit:
            return X, y
        else:
            return X
    

    def __getitem__(self, index):
        batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]

        return self.__get_data(batch)

    
    def __len__(self):
        return self.n // self.batch_size
