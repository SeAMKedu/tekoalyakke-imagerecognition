import tensorflow as tf

from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    def __init__(self, name, savepath, ds, epochs):
        self.name = name
        self.model_savepath = savepath
        self.ds = ds
        self.epochs = epochs
        self.model = None
        self.history = None

    
    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_savepath)


    def setup(self):
        self.setup_model(self.ds.img_size)
        self.compile_model()

        tf.keras.utils.plot_model(self.model, to_file=f"./img/model_{self.name}.png", show_shapes=True)

    @abstractmethod
    def setup_model(self, img_size):
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def train_model(self, save = False):
        pass

