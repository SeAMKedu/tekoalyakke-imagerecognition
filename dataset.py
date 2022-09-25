from email.mime import image
import pathlib
import re

from datetime import datetime

import pandas as pd 

from PIL import Image

from common import *
from datagen import ClockImageDataGen


class DataSet:
    def __init__(self, root, sample_size, validation_fraction=0.2, test_fraction=0.1, batch_size=32, image_size=(128, 128)):
        self.root = root
        self.sample_size = sample_size
        self.df = None
        self.training = None
        self.validation = None
        self.test = None
        self.batch_size = batch_size
        self.img_size = image_size
        self._mode = Mode.SCALED
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction

        self.classify_images(self.root, self.sample_size)
        self.load_images()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self.training.mode = value
        self.validation.mode = value
        self.test.mode = value

    def classify_images(self, root, sample_size):
        image_directory = pathlib.Path(root)
        image_count = len(list(image_directory.glob('*/*.jpg')))

        if DEBUG: print(f"Images found {image_count}")

        images = []
        
        for f in image_directory.glob('*/*.jpg'):
            if self.img_size == None:
                im = Image.open(f)
                self.img_size = im.size

            ts_str = re.search(".*framif_(.*).jpg$", f.name).group(1)

            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").time()

            seconds = ts.hour * 60 * 60 + ts.minute * 60

            time = ts.hour * 60 + ts.minute

            images.append([f, ts.hour, ts.minute, seconds, time])

        self.df = pd.DataFrame(images, columns=["filepath", "hour", "minute", "secondsMidnight", "time"])

        self.df = self.df.sample(frac=sample_size)

        if DEBUG: print(f"Using {len(self.df)} images")

        # normalized to 24h 
        self.df["norm"] = self.df["secondsMidnight"] / 86400


    def load_images(self):
        df_train = self.df.sample(frac=(1.0 - self.test_fraction), random_state=SEED)
        df_test = self.df.drop(df_train.index)
        df_val = df_train.sample(frac=self.validation_fraction, random_state=SEED)
        df_train = df_train.drop(df_val.index)

        self.training = ClockImageDataGen(df_train, batch_size=self.batch_size, input_size=(self.img_size[0], self.img_size[1], 3))
        self.validation = ClockImageDataGen(df_val, batch_size=self.batch_size, input_size=(self.img_size[0], self.img_size[1], 3))
        self.test = ClockImageDataGen(df_test, batch_size=1, input_size=(self.img_size[0], self.img_size[1], 3))
