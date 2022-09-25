
import os
from pathlib import Path
from datetime import datetime, timedelta

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.model_selection import train_test_split

from common import *
from dataset import DataSet, Mode
from hourminute_wrapper import HourMinuteModel
from hourminute_category_wrapper import HourMinuteCategoryModel
from hourminute_time_wrapper import TimeCategoryModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # limit the amount of debug messages from tensorflow

print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))} ")

# matplotlib 3.6.0 has a new annoying warning, so hiding it for now,
# if you use newer, comment two lines below and fix the issue
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_dataset_sample(ds, batch, index):
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True) # A4 in inches
    gs = fig.add_gridspec(1, 2)
    gs_idx = 0

    ax = fig.add_subplot(gs[gs_idx])

    X, y  = ds[batch]
    #print(len(X), len(y), len(y[0]), len(y[1]))
    hour = int(y[0][index] * 24)
    minute = int(y[1][index] * 60)

    image = tf.keras.preprocessing.image.array_to_img(X[index])

    ax.imshow(image)
    
    gs_idx = 1
    ax = fig.add_subplot(gs[gs_idx])

    ax.text(0.5, 0.5, f"{hour}:{minute}", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    plt.show()


def plot_metric(wrapper, ax, title, y_label, lines):
    for l in lines:
        ax.plot(wrapper.history.history[l])

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.legend(["training", "validation"], loc='upper left')


def plot_training_results(wrapper):
    if DEBUG: print(wrapper.model.history.history.keys())

    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True) # A4 in inches
    
    # m * o = metric plots, o + 1 = loss plots
    plot_count = max((len(wrapper.metrics) + len(wrapper.outputs)), len(wrapper.outputs) + 1)
    gs = fig.add_gridspec(2, plot_count)
    gs_idx = 0

    ax = fig.add_subplot(gs[0, gs_idx])

    plot_metric(wrapper, ax, 'Loss', 'loss', ['loss', 'val_loss'])

    if len(wrapper.outputs) > 1:
        for output in wrapper.outputs:
            gs_idx = gs_idx + 1
            ax = fig.add_subplot(gs[0, gs_idx])
            plot_metric(wrapper, ax, "Loss", "Loss", [f"{output}_loss", f"val_{output}_loss"])

    gs_idx = 0

    for m in wrapper.metrics:
        if len(wrapper.outputs) > 1:
            for output in wrapper.outputs:
                ax = fig.add_subplot(gs[1, gs_idx])
                gs_idx = gs_idx + 1

                plot_metric(wrapper, ax, f"Metric {output}", m, [f"{output}_{m}", f"val_{output}_{m}"])
        else:
            ax = fig.add_subplot(gs[1, gs_idx])
            gs_idx = gs_idx + 1

            plot_metric(wrapper, ax, f"Metric", m, [f"{m}", f"val_{m}"])

    fig.tight_layout()

    plt.savefig(f"./img/training_result_{wrapper.name}.png")
    if DEBUG: plt.show()
    plt.close()


def test_model(wrapper, num):
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True) # A4 in inches

    gs = fig.add_gridspec(min(num, len(wrapper.ds.test)) + 1, 4)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"Image", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"Real", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"Prediction", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax = fig.add_subplot(gs[0, 3])
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"Error HH:MM:ss", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    for index in range(min(num, len(wrapper.ds.test))):
        X, y = wrapper.ds.test[index]

        prediction = wrapper.model.predict(X)

        if DEBUG: print(prediction)
        
        if wrapper.ds.mode == Mode.TIME_LABEL:
            ptime = argmax(prediction[0])
            phour = ptime // 60
            pminute = (ptime - phour * 60)

            mod = argmax(y)
            hour = mod // 60
            minute = (mod - hour * 60)
        elif wrapper.ds.mode == Mode.CATEGORICAL:
            phour = argmax(prediction[0])
            pminute = argmax(prediction[1])

            hour = argmax(y[0])
            minute = argmax(y[1])
        elif wrapper.ds.mode == Mode.SCALED:
            phour = int(prediction[0] * 24)
            pminute = int(prediction[1] * 60)

            hour = int(y[0] * 24)
            minute = int(y[1] * 60)
                        
        image = tf.keras.preprocessing.image.array_to_img(X[0])

        if DEBUG: print(f"real: {hour:02d}:{minute:02d} vs predicted: {phour:02d}:{pminute:02d}")
        
        ax = fig.add_subplot(gs[index + 1, 0])
        ax.set_axis_off()
        ax.imshow(image)

        ax = fig.add_subplot(gs[index + 1, 1])
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{hour:02d}:{minute:02d}", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        
        ax = fig.add_subplot(gs[index + 1, 2])
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{phour:02d}:{pminute:02d}", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        ax = fig.add_subplot(gs[index + 1, 3])
        ax.set_axis_off()
        rt = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M") 
        pt = datetime.strptime(f"{phour:02d}:{pminute:02d}", "%H:%M") 
        error = str(rt - pt if rt > pt else pt - rt)
        ax.text(0.5, 0.5, f"{error}", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    fig.tight_layout()

    plt.savefig(f"./img/test_result_{wrapper.name}.png")
    if DEBUG: plt.show()
    plt.close()


def run(ds, wrapper, tests):
    if not Path(wrapper.model_savepath).exists():
        wrapper.setup()
        wrapper.train_model(save=True)
        plot_training_results(wrapper)
    else:
        print("Using pretrained model")
        wrapper.load_model()

    test_model(wrapper, tests)

sample_size=1.0
epochs = 200
test_num = 10
batch_size=32

ds = DataSet(IMAGES, sample_size=sample_size, batch_size=batch_size, image_size=(128, 128))

# NOTE: When creating a new model, it sets the ds mode and can't be
# directly used in another model without setting it again correctly

run(ds, HourMinuteModel(ds, epochs = epochs), tests = test_num)
run(ds, HourMinuteCategoryModel(ds, epochs = epochs), tests = test_num)
run(ds, TimeCategoryModel(ds, epochs = epochs), tests = test_num)
