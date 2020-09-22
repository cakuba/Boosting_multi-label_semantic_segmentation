import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# 注意权重设置能够保证至少过半样本被认为是正确样本
def sample_weights_boosted(y_pred, weights, percent=0.25, thres=0.99, y_true=[]):
    """
    参考AdaBoost算法的权重更新算法，对目标分割样本进行权重更新

    y_pred: 模型预测的样本分割性能 (1-D)
    weights: 对应于每个样本的权重值
    percent: 提升样本的百分比；即性能排名低于percent的样本将被提升
    thres: 分割性能的目标阈值
    y_true: 默认为空

    返回值：更新后权重，模型集成的权重

    """

    assert len(y_pred) == len(weights), '样本数量和权重数量应一致'

    percent_thres = np.percentile(y_pred, percent*100)
    if percent_thres < thres:
        thres = percent_thres
    y_pred = np.array([1 if x >= thres else 0 for x in y_pred])
    if len(y_true)==0:
        y_true = np.ones(y_pred.shape)
    #print(y_pred, y_true)

    miss = [int(x) for x in (y_pred != y_true)]
    # Equivalent with 1/-1 to update weights
    miss2 = [x if x==1 else -1 for x in miss]
    #print(miss, miss2)
    # Error
    err_m = np.dot(weights, miss)/sum(weights)
    if (err_m==0):
        # 模型完美预测结果
        return weights, 10 
    if (err_m==1):
        return weights, 0

    # 如果是个弱分类器；即分类准确率>0.5
    if err_m < 0.5:
        # Alpha
        alpha_m = 0.5*np.log((1-err_m)/err_m)
        # New weights
        new_weights = np.multiply(weights, np.exp([float(x) * alpha_m for x in miss2]))
        new_weights = new_weights/sum(new_weights)

        #print('boosted param', alpha_m, new_weights)
    else:
        alpha_m = 0
        new_weights = weights
        print('WARNING: no sample weights are boosted!')

    # 归一化
    return new_weights, alpha_m

def find_neighbors(maps, target):
    """给定图像和目标像素值，找到图像里面所有目标区域（岛屿）"""

    # 用于记录发现的岛屿ID和对应的坐标; island_id:[(坐标),(坐标)]
    islands = {}
    # 发现岛屿的数量计数
    counter = 0

    # 坐标的属性; 坐标：island_id
    grid = {}
    
    # 第一步：获得全体目标区域坐标集合
    x, y = np.where(maps==target)
    pts = list(zip(x, y))
    
    # 第二步：获得目标坐标形成的岛屿
    for pt in pts:
        # 未访问的坐标
        if pt not in grid:
            cid = counter  # 当前岛屿ID
            islands[cid] = [pt]
            grid[pt] = cid
        
            # 岛屿ID增长1
            counter += 1  
        else:
            cid = grid[pt]
        
        # 寻找邻居并标识
        for m,n in [(0,-1), (0,1), (-1,0), (1,0)]:
            nx, ny = pt[0]+m, pt[1]+n
            # 坐标位于岛屿候选
            if (nx, ny) in pts:
                # 未访问的邻居坐标
                if (nx, ny) not in grid:
                    islands[cid].append((nx,ny))
                    grid[(nx,ny)] = cid
                else:
                    #　已访问的邻居坐标
                    oid = grid[(nx,ny)]
                    if cid != oid:
                        #  合并两个岛屿，更新坐标属性，然后删除被合并岛屿
                        #islands[cid].extend(islands[oid])
                        #for (i, j) in islands[oid]:
                        #    grid[(i,j)] = [cid, 1]
                        #del islands[oid]   
                        islands[oid].extend(islands[cid])
                        for (i, j) in islands[cid]:
                            grid[(i,j)] = oid
                        del islands[cid] 
                        cid = oid  
        
    return islands

# 给定岛屿坐标，判断该岛屿是否被指定像素包围
def is_island(maps, island, background):
    """
    输入：
       maps - 待分析的图像，numpy矩阵
       island - 岛屿的坐标点集合
       background - 背景像素值
    """
    
    row, col = maps.shape
    for (x, y) in island:
        if y-1>=0: 
            if (maps[x, y-1]!=background) and ((x,y-1) not in island):
                return False
    
        if y+1<=col-1:
            if (maps[x, y+1]!=background) and ((x,y+1) not in island):
                return False

        if x-1>=0:
            if (maps[x-1, y]!=background) and ((x-1, y) not in island):
                return False

        if x+1<=row-1: 
            if (maps[x+1, y]!=background) and ((x+1, y) not in island):
                return False
    
    return True

def annotation_fusion(annotation, prediction, mode=3):
    """
    基于语义分割的模型结果，融合标注数据，产生新的标注数据

    输入
        prediction: 语义分割的模型预测mask (0/1)
        annotation: 人工标注的分割mask     (0/1)
        mode: 不同的融合算法模式
            1 - 模型预测+漏掉的人工标注
            2 - 人工标注+漏掉的模型预测
            3 - 人工标注+模型预测的并集 (默认)

    返回
        fusion_ 融合后的mask
    """
    
    pred_ = prediction.astype(np.int)     # predicted mask
    label_ = annotation.astype(np.int)    # labeled mask

    if (mode==1):
        pred_[pred_==0] = 999
        label_[label_==0] = 999
        label_[label_==1] = -1
        fusion = pred_+label_
        
        target = 998
        bkg = 1998
        islands = find_neighbors(fusion, target)
        masks = np.zeros(fusion.shape)
        for i,j in enumerate(islands): 
            if is_island(fusion, islands[j], background=bkg):
                c = np.array([i for i,j in islands[j]])
                d = np.array([j for i,j in islands[j]])
                masks[tuple((c, d))] = 1
        fusion_ = prediction.astype(np.int)+masks
        
    elif (mode==2):
        pred_[pred_==0] = 999
        label_[label_==0] = 999
        label_[label_==1] = -1
        fusion = pred_+label_
        
        target = 1000
        bkg = 1998
        islands = find_neighbors(fusion, target)
        masks = np.zeros(fusion.shape)
        for i,j in enumerate(islands): 
            if is_island(fusion, islands[j], background=bkg):
                c = np.array([i for i,j in islands[j]])
                d = np.array([j for i,j in islands[j]])
                masks[tuple((c, d))] = 1
        fusion_ = annotation.astype(np.int)+masks
    
    else:
        # label ∪ pred
        xor_ = (pred_ ^ label_)
        pred_ = xor_ + pred_
        pred_[pred_>1] = 1
        xor_ = (pred_ ^ label_)
        fusion_ = xor_ + label_
        fusion_[fusion_>1] = 1

    return fusion_.astype(np.uint8)

def dice_coef(y_true, y_pred, threshold=0.5):
    """
    计算模型预测的分割mask与标注mask之间的Dice系数值
    
    输入:
       y_true - 二值化的图像人工标注，格式为[batch_size, width, height]
       y_pred - 模型预测的图像像素分类结果 ([0~1]), ，格式为[batch_size, width, height]
    
    """
    
    smooth = 1.0e-5
    y_pred = ops.convert_to_tensor_v2(y_pred)
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred>threshold, y_pred.dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred),axis=[1,2])
    y_true_sum = tf.reduce_sum(tf.multiply(y_true, y_true),axis=[1,2])
    y_pred_sum = tf.reduce_sum(tf.multiply(y_pred, y_pred),axis=[1,2])
    dice = (2.0*intersection+smooth)/(y_true_sum+y_pred_sum+smooth)
    
    return dice

"""
def dice_coef(y_true, y_pred, thres=0.5, morph=False, kernel_size=5):
    计算模型预测的分割mask与标注mask之间的Dice系数值

    输入:
       y_true - 二值化的图像人工标注（0/1）
       y_pred - 模型预测的图像像素分类结果 ([0~1])
       thres - 对模型预测结果进行二值化；即大于thres被认为是目标

    返回:
       Dice系数值 [0~1]

    assert len(y_true)==len(y_pred), "计算Dice系数值时y_true和y_pred应长度一致！"

    # 对y_pred的二值化预处理
    y_pred = np.array(y_pred)
    y_pred[y_pred>=thres] = 1
    y_pred[y_pred<thres] = 0
    y_pred = np.array(y_pred, dtype=np.uint8).flatten()

    y_true = np.array(y_true)
    smooth = y_true.shape[-1]**2*0.01

    if morph:
        kernel = (kernel_size,kernel_size)
        true_E = cv2.erode(y_true, kernel, iterations=2).flatten()
        true_D = cv2.dilate(y_true, kernel, iterations=2).flatten()
        y_true_coef = (true_E-true_D+1)
        intersection = np.sum(true_E*y_pred)

        dice = (2*intersection+smooth)/(np.sum(true_E)+np.sum(y_pred*y_true_coef)+smooth)
    else:
        y_true = y_true.flatten()
        intersection = np.sum(y_pred*y_true)
        dice = (2.*intersection+smooth)/(np.sum(y_true)+np.sum(y_pred)+smooth)

    return dice
"""

# 可视化辅助函数 - soma/vessel合成
def merge2obj(soma, vessel, thres=1):
    """
    Given the prediction of soma and vessel in binary format (width * height),
    this function merges soma and vessel into one single color image
    with vessels marked as red.
    
    Inputs - 
        thres: the binary threshold value in given images
    
    Return - 
        merged_img: the color image containing both soma and vessel
    """
    
    assert len(soma.shape) == len(vessel.shape), "soma and vessel image must have the same shape"
    assert len(soma.shape) == 2, 'soma image must be binary or gray format'
    assert len(vessel.shape) == 2, 'vesssel image must be binary or gray format'
    
    sc=cv2.cvtColor(soma,cv2.COLOR_GRAY2BGR)
    vc=cv2.cvtColor(vessel,cv2.COLOR_GRAY2BGR)

    sc[sc==thres] = 255
    ind = np.where(vc[:,:,0]!=0)
    for i in range(3):
        for j in range(len(ind[0])):
            x, y = ind[0][j], ind[1][j]
            if i == 0:
                vc[x,y,i] = 255
            else:
                vc[x,y,i] = 0
    
    return vc + sc

# 可视化辅助函数 - 原始图像/人工标注
def image_vis(image, soma_label, vessel_label):
    plt.figure(figsize=(14,6))

    plt.subplot(131)
    plt.imshow(image,cmap='gray')
    plt.title('Original image')

    plt.subplot(132)
    plt.imshow(soma_label,cmap='gray')
    plt.title('Annotated label for soma')

    plt.subplot(133)
    plt.imshow(vessel_label,cmap='gray')
    plt.title('Annotated label for vessel')

    plt.show()

# 可视化辅助函数 - 原始图像/人工标注/模型预测
def vis_image_pred(image, soma_label, vessel_label, soma_pred, vessel_pred, thres=0.5):

    soma_pred[soma_pred>=thres] = 1
    soma_pred[soma_pred<thres] = 0
    vessel_pred[vessel_pred>=thres] = 1
    vessel_pred[vessel_pred<thres] = 0


    plt.figure(figsize=(12,12))
    plt.subplot(231)
    plt.imshow(image,cmap='gray')
    plt.title('Original image')

    plt.subplot(232)
    plt.imshow(soma_label,cmap='gray')
    plt.title('Annotated label for soma')

    plt.subplot(233)
    plt.imshow(vessel_label,cmap='gray')
    plt.title('Annotated label for vessel')

    plt.subplot(235)
    plt.imshow(soma_pred,cmap='gray')
    plt.title('Predicted label for soma')

    plt.subplot(236)
    plt.imshow(vessel_pred,cmap='gray')
    plt.title('Predicted label for vessel')

    plt.show()

def vis_training(path, validation=False, acc="accuracy", epochs_of_training=100, epochs_of_boosting=10):
    """   
    提升过程中训练曲线的可视化；包括以下变量
    
    ['loss', 'soma_loss', 'vessel_loss', 'soma_'+acc, 'vessel_acc'+acc, 
     'val_loss', 'val_soma_loss', 'val_vessel_loss', 'val_soma_'+acc, 'val_vessel_'+acc]
     
    """

    import os
    import pickle

    loss = 'loss'
    soma_loss = 'soma_loss'
    vessel_loss = 'vessel_loss'
    soma_acc = 'soma_'+acc
    vessel_acc = 'vessel_'+acc
    title = 'Learning curve for TRAINING data'
    
    if validation:
        loss = 'val_'+loss
        soma_loss = 'val_'+soma_loss
        vessel_loss = 'val_'+vessel_loss
        soma_acc = 'val_'+soma_acc
        vessel_acc = 'val_'+vessel_acc
        title = 'Learning curve for VALIDATION data'
    
    plt.figure(figsize=(12,20))
    for epoch in range(epochs_of_boosting):
        file = os.path.join(path, 'history-epoch-'+str(epoch+1).zfill(2)+'.dat')
        with open(file, 'rb') as f:
            history = pickle.load(f)
            num = len(history[loss])
            start = int(epochs_of_training/num)
            x = range(start, epochs_of_training+1, start)

            plt.subplot(np.ceil(epochs_of_boosting/3),np.ceil(epochs_of_boosting/3),epoch+1)
            plt.plot(x, np.squeeze(history[loss]),'r')
            plt.plot(x, np.squeeze(history[soma_loss]),'b')
            plt.plot(x, np.squeeze(history[vessel_loss]),'g')

            plt.plot(x, np.squeeze(history[soma_acc]),'b--')
            plt.plot(x, np.squeeze(history[vessel_acc]),'g--')
            if epoch<np.ceil(epochs_of_boosting/3)+1:
                plt.title(title)
            if epoch>6:
                plt.xlabel('Number of epochs')
            plt.ylim([0, 1])
            plt.xlim([0, epochs_of_training])
            plt.grid()
            plt.legend(['Total Loss','Loss (soma)','Loss (vessel)','ACC (soma)', 'ACC (vessel)'])
        
    plt.show()

def vis_boosting(path, epochs_of_boosting=10):
    """
    提升过程中模型性能的可视化；包括在测试集和金标数据上的表现

    """    

    import os
    import pickle

    test_soma = np.array([])
    test_vessel = np.array([])
    gold_soma = np.array([])
    gold_vessel = np.array([])
    
    for epoch in range(epochs_of_boosting):
        file = os.path.join(path, 'performance-epoch-'+str(epoch+1).zfill(2)+'.dat')
        with open(file, 'rb') as f:
            test_soma_dice,test_vessel_dice,gold_soma_dice,gold_vessel_dice, sw, vw = pickle.load(f)
            
            if len(test_soma):
                test_soma = np.vstack([test_soma, np.array(test_soma_dice)])
                test_vessel = np.vstack([test_vessel, np.array(test_vessel_dice)])
                gold_soma = np.vstack([gold_soma, np.array(gold_soma_dice)])            
                gold_vessel = np.vstack([gold_vessel, np.array(gold_vessel_dice)])
            else:
                test_soma = np.array(test_soma_dice)
                test_vessel = np.array(test_vessel_dice)
                gold_soma = np.array(gold_soma_dice)           
                gold_vessel = np.array(gold_vessel_dice)
    
    # 可视化曲线
    x = range(1,11)
    plt.figure(figsize=(12,16))
    
    plt.subplot(3,1,1)
    plt.plot(x, np.mean(gold_soma,axis=1),'b')
    plt.plot(x, np.mean(gold_vessel,axis=1),'g')  
    plt.plot(x, np.mean(test_soma,axis=1),'b--')
    plt.plot(x, np.mean(test_vessel,axis=1),'g--')   
    plt.title("Performance of Boosting Framework (Dice Coef)")
    plt.xlabel('Number of boosting epochs')
    plt.ylim([0, 1])
    plt.xlim([x[0], x[-1]])
    plt.grid()
    plt.legend(['Gold(soma)','Gold(vessel)','Test(soma)','Test(vessel)'])
    
    plt.subplot(3,2,3)
    plt.plot(x, gold_soma)
    plt.title("Gold set (SOMA)")
    plt.ylim([0, 1])
    plt.xlim([x[0], x[-1]])
    plt.grid()
    
    plt.subplot(3,2,4)
    plt.plot(x, gold_vessel)
    plt.title("Gold set (VESSEL)")
    plt.ylim([0, 1])
    plt.xlim([x[0], x[-1]])
    plt.grid()
    
    plt.subplot(3,2,5)
    plt.plot(x, test_soma)
    plt.title("Test set (SOMA)")
    plt.ylim([0, 1])
    plt.xlim([x[0], x[-1]])
    plt.xlabel('Number of boosting epochs')
    plt.grid()
    
    plt.subplot(3,2,6)
    plt.plot(x, test_vessel)
    plt.title("Test set (VESSEL)")
    plt.ylim([0, 1])
    plt.xlim([x[0], x[-1]])
    plt.xlabel('Number of boosting epochs')
    plt.grid()    
    
    plt.show()