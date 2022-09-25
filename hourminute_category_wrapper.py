import tensorflow as tf
from keras.models import Model
from keras.layers import *

from tqdm.keras import TqdmCallback

from common import *
from wrapper import ModelWrapper

class HourMinuteCategoryModel(ModelWrapper):
    ds_mode = Mode.CATEGORICAL

    def __init__(self, ds, epochs):
        super().__init__("hhmmcat", "models/hhmmcat_model", ds, epochs)
        self.outputs = ["hour_output", "minute_output"]
        self.metrics = ["accuracy"]
        self.ds.mode = self.ds_mode


    def setup_model(self, img_size):
        inputs = Input(shape=(img_size[0], img_size[1], 3))
        x = Rescaling(1./255)(inputs)
        x = Conv2D(64, 8, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 8, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 8, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, 8, activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        #hour = Conv2D(32, 4, activation='relu')(x)
        #hour = MaxPool2D(pool_size=(4, 4), padding='valid')(hour)
        #hour = BatchNormalization()(hour)
        #hour = Flatten()(hour)
        hour = Dense(24, activation='relu')(x)
        hour = Dropout(0.1)(hour)
        hour = Dense(48, activation='relu', kernel_regularizer='l2')(hour)
        #hour = Dense(24, activation='relu')(hour)
        hour = Dense(24, activation='softmax', name="hour_output")(hour)

        #minute = Conv2D(32, 4, activation='relu')(x)
        #minute = MaxPool2D(pool_size=(4, 4), padding='valid')(minute)
        #minute = BatchNormalization()(minute)
        #minute = Flatten()(minute)
        minute = Dense(60, activation='relu')(x)
        minute = Dropout(0.1)(minute)
        minute = Dense(120, activation='relu', kernel_regularizer='l2')(minute)
        #minute = Dense(60, activation='relu')(minute)
        minute = Dense(60, activation='softmax', name="minute_output")(minute)

        self.model = Model(inputs=inputs, outputs = [hour, minute], name="hhmmcat")


    def compile_model(self):
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, epsilon = 1e-05, amsgrad = False, name = 'Adam')
        nadam_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")

        cg = tf.keras.losses.CategoricalCrossentropy()
        scg = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        bc = tf.keras.losses.BinaryCrossentropy(from_logits=False),

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
