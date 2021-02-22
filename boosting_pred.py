import os
import cv2
import numpy as np

# configuation on the specific GPU ID used when running 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# configuration on the tensorflow LOG LEVEL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# libraries
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.models import load_model

from data_processing import get_data_orig
from utils import customized_loss_fn, dice_coef

# image size
width = 1024
height = 1024

# data preprocessing (histgram-equalization and stretching)
preprocessing = True
images, soma_labels, vessel_labels = get_data_orig('../data/Test',
                                                    size=(width, height),preprocessing=preprocessing)
images = np.expand_dims(images, axis=-1)
soma_labels = np.expand_dims(soma_labels, axis=-1)
vessel_labels = np.expand_dims(vessel_labels, axis=-1)

# network prediction
model = load_model('./weights/model-weights-epoch-10.hdf5')
out = model.predict([images, soma_labels, vessel_labels])
soma_labels_pred, vessel_labels_pred = out[0][0], out[0][1]

# network performance evaluation
soma_dice = dice_coef(np.squeeze(soma_labels),np.squeeze(soma_labels_pred))
vessel_dice = dice_coef(np.squeeze(vessel_labels),np.squeeze(vessel_labels_pred))
print("Boosting network performance for SOMA: ", np.mean(soma_dice))
print("Boosting network for VESSEL: ", np.mean(vessel_dice))
