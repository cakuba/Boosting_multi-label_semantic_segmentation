import os
import cv2
import numpy as np

def get_data_gold(path,size=(256,256),normalized=True, preprocessing=False):
    """
    获得金标准数据；其中
    图像格式(num_images, weight, height)
    标注格式(num_images, weight, height)，像素值为0/1
    
    输入：
    path - 指定的数据目录
    size - 输出的图像数据大小
    normalized - 图像数据是否归一化
    
    注：金标数据目录结构如下
    gold/ 
        images/ 
        labels/ 
            soma/ 
            vessel/
    """
    
    files = os.listdir(os.path.join(path, 'images'))
    files.sort()
    gold_images = np.zeros([len(files),size[0],size[1]])
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, 'images', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        if preprocessing:
            img = image_histEqu(img)   
        if normalized:
            gold_images[i,:,:] = img/255
        else:
            gold_images[i,:,:] = img
    
    files = os.listdir(os.path.join(path, 'labels/soma'))
    files.sort()
    gold_label_soma = np.zeros([len(files),size[0],size[1]],dtype=np.uint8)
    for i, file in enumerate(files):
        #print(file, gold_images.shape)
        img = cv2.imread(os.path.join(path, 'labels/soma', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        gold_label_soma[i,:,:] = img_bin
        
    files = os.listdir(os.path.join(path, 'labels/vessel'))
    files.sort()
    gold_label_vessel = np.zeros([len(files),size[0],size[1]],dtype=np.uint8)
    for i, file in enumerate(files):
        #print(file, gold_images.shape)
        img = cv2.imread(os.path.join(path, 'labels/vessel', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        gold_label_vessel[i,:,:] = img_bin
    
    return gold_images, gold_label_soma, gold_label_vessel

def get_data_orig(path, size=(256,256),normalized=True, preprocessing=False):
    """
    获得模型学习所需要的数据；
    其中图像格式(num_images, weight, height)
    标注格式(num_images, weight, height)，像素值为0/1

    注：训练数据目录结构如下
    path/
        images/
            0/
        labels/
            soma/
            vessel/
    """

    files = os.listdir(os.path.join(path, 'images/0'))
    files.sort()
    images = np.zeros([len(files),size[0],size[1]])
    for i, file in enumerate(files):
        #print(file, gold_images.shape)
        img = cv2.imread(os.path.join(path, 'images/0', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        if preprocessing:
            img = image_histEqu(img) 
        if normalized:
            images[i,:,:] = img/255
        else:
            images[i,:,:] = img

    files = os.listdir(os.path.join(path, 'labels/soma'))
    files.sort()
    label_soma = np.zeros([len(files),size[0],size[1]],dtype=np.uint8)
    for i, file in enumerate(files):
        #print(file, gold_images.shape)
        img = cv2.imread(os.path.join(path, 'labels/soma', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        label_soma[i,:,:] = img_bin

    files = os.listdir(os.path.join(path, 'labels/vessel'))
    files.sort()
    label_vessel = np.zeros([len(files),size[0],size[1]],dtype=np.uint8)
    for i, file in enumerate(files):
        #print(file, gold_images.shape)
        img = cv2.imread(os.path.join(path, 'labels/vessel', file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        label_vessel[i,:,:] = img_bin

    return images, label_soma, label_vessel

def image_histEqu(img, climit=4.0):
    """ 自适应的图像直方图均衡化+灰度拉伸处理
    
    输入：
        img - 图像矩阵（numpy格式）
    """
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8,8))
    img_hist = clahe.apply(img)
    
    # 灰度拉伸
    img_min = np.min(img_hist)
    img_max = np.max(img_hist)
    
    return (img_hist-img_min)/(img_max-img_min)*255