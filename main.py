# Import Libraries

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Data Preprocessing

# Training image preprocessing

trainingSet = tf.keras.utils.image_dataset_from_directory(
    "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

# Validation Image Processing
validationSet = tf.keras.utils.image_dataset_from_directory(
    "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

print(trainingSet)
for x, y in trainingSet:
    print(x, x.shape)
    print(y, y.shape)
    break