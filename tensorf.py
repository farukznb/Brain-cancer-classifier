import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

def build_model(num_classes, input_shape=(224, 224, 3), lr=0.0005, fine_tune=False):
    """Build MobileNetV2 model with transfer learning and optional fine-tuning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    
    if fine_tune:
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model