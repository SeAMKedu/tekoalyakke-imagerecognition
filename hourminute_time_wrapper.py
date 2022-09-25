import tensorflow as tf
from keras.models import Model
from keras.layers import *

from tqdm.keras import TqdmCallback

from common import *
from wrapper import ModelWrapper

class TimeCategoryModel(ModelWrapper):
    ds_mode = Mode.TIME_LABEL

    def __init__(self, ds, epochs):
        super().__init__("time", "models/time_model", ds, epochs)
        self.outputs = ["time_output"]
        self.metrics = ["accuracy"]
        self.ds.mode = self.ds_mode    


    def setup_model(self, img_size):
        inputs = Input(shape=(img_size[0], img_size[1], 3))
        x = Rescaling(1./255)(inputs)
        x = Conv2D(64, 4, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 4, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 4, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 4, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(24 * 60 * 4, activation='relu', kernel_regularizer='l2')(x)
        x = Dense(24 * 60 * 2, activation='relu', kernel_regularizer='l2')(x)
        x = Dense(24 * 60, activation='softmax', name='time_output')(x)

        self.model = Model(inputs=inputs, outputs = x, name="time")


    def compile_model(self):
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, epsilon = 1e-05, amsgrad = False, name = 'Adam')
        nadam_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")

        cg = tf.keras.losses.CategoricalCrossentropy()
        bc = tf.keras.losses.BinaryCrossentropy(from_logits=False),
        huber = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")

        self.model.compile(optimizer=adam_optimizer, loss=cg, metrics=self.metrics)
        
        if DEBUG: self.model.summary()


    def train_model(self, save = False):
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        self.history = self.model.fit(
            self.ds.training,
            validation_data = self.ds.validation,
            epochs          = self.epochs, 
            callbacks       = [earlystop, reduce_lr, TqdmCallback(verbose=1)],
            shuffle         = True,
            verbose         = 0
            )

        if save: self.model.save(self.model_savepath)
