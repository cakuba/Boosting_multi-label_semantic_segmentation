import os
import time
import pickle
import numpy as np

# 设置模型训练使用的GPU参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 图像大小
width = 1024
height = 1024

# 保存模型参数
model_saved = True

# 模型预测结果的阈值；即大于0.5被认为是分割目标
pred_bin_thres = 0.5

# 提升轮数
epochs_of_boosting = 10

# 希望通过提升达到的dice阈值
boost_dice_soma_thres = 0.99
boost_dice_vessel_thres = 0.99
boost_soma_percent = 1.0
boost_vessel_percent = 1.0

# 数据融合的模型性能阈值；低于该性能的模型将发生不融合
fusion_model_soma_thres = 0.90
fusion_model_vessel_thres = 0.90
# 单个数据融合的dice阈值；大于该阈值的数据将"不"融合
fusion_dice_soma_thres = 0.99
fusion_dice_vessel_thres = 0.99

# 数据融合方法
from utils import annotation_fusion
fusion_mode = 3    # 2-以prediction为主； 3-以annotation为主

from data_processing import get_data_gold, get_data_orig
# 数据预处理（直方图均衡化+拉伸）
preprocessing = True

# 初始化训练所需要的数据 (规范化处理后)
gold_images, gold_soma_labels, gold_vessel_labels = get_data_gold('../data/gold',
                                                    size=(width, height),preprocessing=preprocessing)
train_images, train_soma_labels, train_vessel_labels = get_data_orig('../data/Train_aug',
                                                    size=(width, height),preprocessing=preprocessing)
valid_images, valid_soma_labels, valid_vessel_labels = get_data_orig('../data/Valid',
                                                    size=(width, height),preprocessing=preprocessing)
test_images, test_soma_labels, test_vessel_labels = get_data_orig('../data/Test',
                                                    size=(width, height),preprocessing=preprocessing)

nums_of_samples = len(train_images)
print("A total number of %d images fed into the network training..." % nums_of_samples)

# 样本权重初始化设置
#  模型接受的样本权重格式为
#     {'soma':np.array(), 'vessel':np.array()}
#
weights_of_samples = {'soma':np.ones(nums_of_samples,)/nums_of_samples,
                      'vessel':np.ones(nums_of_samples,)/nums_of_samples}

# 训练数据维度更新，适合模型输入
train_images = np.expand_dims(train_images, axis=-1)
train_soma_labels = np.expand_dims(train_soma_labels, axis=-1)
train_vessel_labels = np.expand_dims(train_vessel_labels, axis=-1)

valid_images = np.expand_dims(valid_images, axis=-1)
valid_soma_labels = np.expand_dims(valid_soma_labels, axis=-1)
valid_vessel_labels = np.expand_dims(valid_vessel_labels, axis=-1)

# 导入U-Net模型
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D,
                                    Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization, Lambda)

from UNet import MultiLabel_UNet, real_dice_coef_loss
from utils import sample_weights_boosted
from numpy.random import randint

# 自定义模型参数
img_width=1024
img_height=1024
dropout=0.5

lr = 5.e-5
batch_size = 6
epochs_of_training = 3000

soma_weight_of_epoch = []
vessel_weight_of_epoch = []
for epoch in range(epochs_of_boosting):
    start_time = time.time()

    print('\nCompiling and training the model...')
    K.clear_session()
    model = MultiLabel_UNet(img_width,img_height,dropout=dropout,seed=randint(10000),
                            loss_mode=1,isWeighted=True,sample_weights=weights_of_samples)
    model.compile(optimizer=Adam(lr=lr),loss=[None]*len(model.outputs))

    weights_file = 'model-weights-epoch-'+str(epoch+1).zfill(2)+'.hdf5'
    checkpoint = ModelCheckpoint('./weights/'+weights_file,monitor='val_loss',
                                verbose=1, mode='min', save_best_only=True)

    # 模型训练
    history = model.fit([train_images, train_soma_labels, train_vessel_labels],
                    batch_size=batch_size,
                    epochs=epochs_of_training,
                    shuffle=True,
                    validation_data=([valid_images, valid_soma_labels, valid_vessel_labels],None),
                    callbacks=[checkpoint],
                    sample_weight=weights_of_samples,
                    verbose=1)

    print('==============================================\n'*3)
    print('\nTraining Finished with %.2f seconds in boosting epoch %d!'% (time.time()-start_time, epoch))
    print('==============================================\n'*3)

    # 保存训练历史
    if model_saved:
        history_file = os.path.join('./weights', 'history-epoch-'+str(epoch+1).zfill(2)+'.dat')
        with open(history_file,'wb') as f:
            pickle.dump(history.history, f)
        print("Model training history successfully saved!")

    # 训练集的权重提升和人工标注数据融合
    train_soma_labels_pred = np.zeros(train_soma_labels.shape)
    train_vessel_labels_pred = np.zeros(train_vessel_labels.shape)

    # 模型预测 (note: limited by GPU memory)
    for i in range(int(nums_of_samples/10)):
        start = 10*i
        end = 10*(i+1)
        train_image = train_images[start:end,:,:,:]
        soma_labels = train_soma_labels[start:end,:,:,:]
        vessel_labels = train_vessel_labels[start:end,:,:,:]
        out = model.predict([train_image,soma_labels,vessel_labels])

        train_soma_labels_pred[start:end,:,:,:], train_vessel_labels_pred[start:end,:,:,:] = out[0][0], out[0][1]

    train_soma_dice = real_dice_coef_loss(np.squeeze(train_soma_labels),np.squeeze(train_soma_labels_pred))
    train_vessel_dice = real_dice_coef_loss(np.squeeze(train_vessel_labels),np.squeeze(train_vessel_labels_pred))
    print("Model performance on Training set for SOMA: ", np.mean(train_soma_dice))
    print("Model performance on Training set for VESSEL: ", np.mean(train_vessel_dice))

    num_fusion_soma = num_fusion_vessel = 0
    for i in range(nums_of_samples):
        # 数据融合; 某张图像结果不太好且整个模型的结果足够好
        if train_soma_dice[i] <= fusion_dice_soma_thres and np.mean(train_soma_dice) > fusion_model_soma_thres:
            pred_ = np.squeeze(train_soma_labels_pred)[i]
            pred_[pred_>=0.5] = 1
            pred_[pred_<0.5] = 0
            train_soma_labels[i,:,:,0] = annotation_fusion(np.squeeze(train_soma_labels)[i], pred_, mode=fusion_mode)
            num_fusion_soma += 1
        if train_vessel_dice[i] <= fusion_dice_vessel_thres and np.mean(train_vessel_dice) > fusion_model_vessel_thres:
            pred_ = np.squeeze(train_vessel_labels_pred)[i]
            pred_[pred_>=0.5] = 1
            pred_[pred_<0.5] = 0
            train_vessel_labels[i,:,:,0] = annotation_fusion(np.squeeze(train_vessel_labels)[i], pred_, mode=fusion_mode)
            num_fusion_vessel += 1
    print("A total number of %d (soma) and %d (vessel) annotated labels are merged with model predictions!"
                         %(num_fusion_soma, num_fusion_vessel))

    # 权重提升
    weights_of_samples['soma'], w = sample_weights_boosted(train_soma_dice, weights_of_samples['soma'],
                                                           percent=boost_soma_percent, thres=boost_dice_soma_thres)
    print("BOOSTED SOMA WEIGHTS \n",weights_of_samples['soma'])
    soma_weight_of_epoch.append(w)
    weights_of_samples['vessel'], w = sample_weights_boosted(train_vessel_dice, weights_of_samples['vessel'],
                                                           percent=boost_vessel_percent, thres=boost_dice_vessel_thres)
    print("BOOSTED VESSEL WEIGHTS \n",weights_of_samples['vessel'])
    vessel_weight_of_epoch.append(w)
    print("Trainig sample weights are successfully boosted! ")

    # 评估模型在测试集和金标数据上的性能，作为提升框架的性能评估指标
    nums_of_test_samples = len(test_images)
    test_soma_labels_pred = np.zeros(test_soma_labels.shape)
    test_vessel_labels_pred = np.zeros(test_vessel_labels.shape)

    # 模型预测 (note: limited by GPU memory; would be updated in better GPUs)
    for i in range(int(nums_of_test_samples/10)):
        start = 10*i
        end = 10*(i+1)
        test_image = test_images[start:end,:,:]
        soma_labels = test_soma_labels[start:end,:,:]
        vessel_labels = test_vessel_labels[start:end,:,:]
        out = model.predict([np.expand_dims(test_image, axis=-1), np.expand_dims(soma_labels, axis=-1),
                             np.expand_dims(vessel_labels, axis=-1)])

        test_soma_labels_pred[start:end,:,:], test_vessel_labels_pred[start:end,:,:] = np.squeeze(out[0][0]), np.squeeze(out[0][1])

    test_soma_dice = real_dice_coef_loss(test_soma_labels,test_soma_labels_pred)
    test_vessel_dice = real_dice_coef_loss(test_vessel_labels,test_vessel_labels_pred)

    # 金标数据10张，可直接预测
    out = model.predict([np.expand_dims(gold_images, axis=-1), np.expand_dims(gold_soma_labels, axis=-1),
                         np.expand_dims(gold_vessel_labels, axis=-1)])
    gold_soma_labels_pred, gold_vessel_labels_pred = out[0][0], out[0][1]
    gold_soma_dice = real_dice_coef_loss(gold_soma_labels, np.squeeze(gold_soma_labels_pred))
    gold_vessel_dice = real_dice_coef_loss(gold_vessel_labels, np.squeeze(gold_vessel_labels_pred))

    # 保存模型性能
    if model_saved:
        performance_file = os.path.join('./weights', 'performance-epoch-'+str(epoch+1).zfill(2)+'.dat')
        with open(performance_file,'wb') as f:
            pickle.dump((test_soma_dice,test_vessel_dice,gold_soma_dice,gold_vessel_dice),f)
                         #gold_vessel_dice,soma_weight_of_epoch,vessel_weight_of_epoch),f)
    print("Model performance on test/gold sets is successfully saved!")

    print('==============================================\n')
    print('\nSuccessfully acommplish the boosting epoch %d!'% epoch)
    print('==============================================\n\n')
