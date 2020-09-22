# 导入函数库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D,
                         Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization, Lambda)

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def real_dice_coef_loss(y_true, y_pred):
    """
    计算模型预测的分割mask与标注mask之间的Dice系数值
    
    输入:
       y_true - 二值化的图像人工标注，格式为[batch_size, width, height]
       y_pred - 模型预测的图像像素分类结果 ([0~1]), ，格式为[batch_size, width, height]
    
    """
    
    smooth = 1.0e-5
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred),axis=[1,2])
    y_true_sum = tf.reduce_sum(tf.multiply(y_true, y_true),axis=[1,2])
    y_pred_sum = tf.reduce_sum(tf.multiply(y_pred, y_pred),axis=[1,2])
    dice = (2.0*intersection+smooth)/(y_true_sum+y_pred_sum+smooth)
    
    return dice

# 自定义损失函数
def customized_loss_fn(target_soma, target_vessel, output_soma, output_vessel, mode=1, isWeighted=False, sample_weights=None):
    # 我们希望在多标签输出中，同一目标之间的dice尽量大，不同目标之间的dice尽量小
    soma_loss = real_dice_coef_loss(target_soma, output_soma)
    vessel_loss = real_dice_coef_loss(target_vessel, output_vessel)
    soma_vs_vessel_loss = real_dice_coef_loss(target_soma, output_vessel)
    vessel_vs_soma_loss = real_dice_coef_loss(target_vessel, output_soma)
      
    if mode == 1:
        # log求和
        if isWeighted:
            loss_soma = -math_ops.log(soma_loss)-math_ops.log(1-soma_vs_vessel_loss)
            loss_vessel = -math_ops.log(vessel_loss)-math_ops.log(1-vessel_vs_soma_loss)
            loss = tf.math.multiply(loss_soma,sample_weights['soma'])+tf.math.multiply(loss_vessel,sample_weights['vessel'])
        else:
            loss = -math_ops.log(soma_loss)-math_ops.log(vessel_loss)
            loss += -math_ops.log(1-soma_vs_vessel_loss)-math_ops.log(1-vessel_vs_soma_loss)
    elif mode == 2:
        # 借鉴交叉熵函数1
        loss = -tf.math.multiply(soma_loss,math_ops.log(1-soma_vs_vessel_loss))-math_ops.log(soma_loss)
        loss = -tf.math.multiply(vessel_loss,math_ops.log(1-vessel_vs_soma_loss))-math_ops.log(vessel_loss)
    else:
        # 借鉴交叉熵函数2
        loss = -tf.math.multiply(soma_loss,math_ops.log(1-soma_vs_vessel_loss))-tf.math.multiply(1+soma_vs_vessel_loss,math_ops.log(soma_loss))
        loss = -tf.math.multiply(vessel_loss,math_ops.log(1-vessel_vs_soma_loss))-tf.math.multiply(1+vessel_vs_soma_loss,math_ops.log(vessel_loss))                 
    return loss

def MultiLabel_UNet(img_width=256,img_height=256,dropout=0.2,seed=2020,show_summary=False,loss_mode=1,isWeighted=False,sample_weights=None):
    # 初始化权重
    kernel_initializer=initializers.he_normal(seed=seed)

    inputs = Input((img_width, img_height, 1),name='input')
    # -------------------------------------------------------------------------------------------------------------
    conv1_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1_0)
    bn1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2))(bn1)
    # -------------------------------------------------------------------------------------------------------------
    conv2_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2_0)
    bn2 = BatchNormalization()(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2))(bn2)
    # -------------------------------------------------------------------------------------------------------------
    conv3_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3_0)
    bn3 = BatchNormalization()(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2))(bn3)
    # -------------------------------------------------------------------------------------------------------------
    conv4_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4_0)
    bn4 = BatchNormalization()(conv4_1)
    drop4 = Dropout(dropout)(bn4)
    # 下采样结束
    # -------------------------------------------------------------------------------------------------------------
    # 开始上采样

    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3_1, up5], axis=3)
    conv5_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge5)
    conv5_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5_0)
    bn5 = BatchNormalization()(conv5_1)
    # -------------------------------------------------------------------------------------------------------------
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bn5))
    merge6 = concatenate([conv2_1, up6], axis=3)
    conv6_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6_0)
    bn6 = BatchNormalization()(conv6_1)
    # -------------------------------------------------------------------------------------------------------------
    up7 = Conv2D(64 , 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bn6))
    merge7 = concatenate([conv1_1, up7], axis=3)
    conv7_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7_0)
    bn7 = BatchNormalization()(conv7_1)
    conv7 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(bn7)
    # ----------------------------------------------------------------------------------------------------------
    output_soma = Conv2D(1, 1, activation='sigmoid', name='output_soma')(conv7)
    output_vessel = Conv2D(1, 1, activation='sigmoid', name='output_vessel')(conv7)

    # -------------------------------------------------------------------------------------------------------------
    # 定义标注数据作为模型输入，用于自定义损失函数的计算
    target_soma = Input((img_width, img_height, 1), name='input_soma')
    target_vessel = Input((img_width, img_height, 1), name='input_vessel')
    customized_loss = Lambda(lambda x: customized_loss_fn(*x,loss_mode,isWeighted,sample_weights), name="customized_loss")(
                     [target_soma, target_vessel, output_soma, output_vessel])

    model = Model(inputs=[inputs, target_soma, target_vessel], outputs=[[output_soma, output_vessel],customized_loss])
    #model = Model(inputs=inputs, outputs=[conv8,conv9])

    layer = model.get_layer("customized_loss")
    loss = tf.reduce_sum(layer.output, keepdims=True)
    model.add_loss(loss)
    
    if show_summary:
        model.summary()

    return model
