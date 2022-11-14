import tensorflow as tf
from keras.models import Model
from keras.layers import *

from tqdm.keras import TqdmCallback

from common import *
from wrapper import ModelWrapper

class HourMinuteModel(ModelWrapper):
    ds_mode = Mode.SCALED

    def __init__(self, ds, epochs):
        super().__init__("hhmm", "models/hhmm_model", ds, epochs)
        self.outputs = ["hour_output", "minute_output"]
        self.metrics = ["mean_absolute_percentage_error"]
        self.ds.mode = self.ds_mode
        

    def setup_model(self, img_size):
        inputs = Input(shape=(img_size[0], img_size[1], 3))
        x = Rescaling(1./255)(inputs)
        x = Conv2D(32, 8, activation='relu')(x)
        x = Conv2D(16, 8, activation='relu')(x)
        x = Conv2D(8, 8, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.10)(x)
        x = Dense(64, activation='relu', kernel_regularizer='l2')(x)

        hour = Dense(24, activation='relu')(x)
        hour = Dropout(0.1)(hour)
        hour = Dense(48, activation='relu', kernel_regularizer='l2')(hour)
        hour = Dense(1, activation='sigmoid', name="hour_output")(hour)

        minute = Dense(60, activation='relu')(x)
        minute = Dropout(0.1)(minute)
        minute = Dense(120, activation='relu', kernel_regularizer='l2')(minute)
        minute = Dense(1, activation='sigmoid', name="minute_output")(minute)

        self.model = Model(inputs=inputs, outputs = [hour, minute], name="hhmm")


    def compile_model(self):
        nadam_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")

        huber = tf.keras.losses.Huber(delta=1.35, reduction="auto", name="huber_loss")

        self.model.compile(optimizer=nadam_optimizer,
                loss=huber,
                metrics=self.metrics)
        
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
