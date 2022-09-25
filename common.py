from enum import Enum

DEBUG = True
USE_CACHE = True
SEED = 1234
IMAGES = f"./cropped"

class Mode(Enum):
    SCALED = 1
    CATEGORICAL = 2
    TIME_LABEL = 3

