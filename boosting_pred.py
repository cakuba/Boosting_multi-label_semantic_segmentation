import os
import cv2
import numpy as np

# 设置模型训练使用的GPU参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置debug信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 导入函数库
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.models import load_model

from data_processing import get_data_orig
from utils import customized_loss_fn, dice_coef

# 图像大小
width = 1024
height = 1024

# 数据预处理（直方图均衡化+拉伸）
preprocessing = True
images, soma_labels, vessel_labels = get_data_orig('../data/Test',
                                                    size=(width, height),preprocessing=preprocessing)
images = np.expand_dims(images, axis=-1)
soma_labels = np.expand_dims(soma_labels, axis=-1)
vessel_labels = np.expand_dims(vessel_labels, axis=-1)

# 模型预测
model = load_model('./weights/model-weights-epoch-10.hdf5')
out = model.predict([images, soma_labels, vessel_labels])
soma_labels_pred, vessel_labels_pred = out[0][0], out[0][1]

# 性能评估
soma_dice = dice_coef(np.squeeze(soma_labels),np.squeeze(soma_labels_pred))
vessel_dice = dice_coef(np.squeeze(vessel_labels),np.squeeze(vessel_labels_pred))
print("Boosting network performance for SOMA: ", np.mean(soma_dice))
print("Boosting network for VESSEL: ", np.mean(vessel_dice))