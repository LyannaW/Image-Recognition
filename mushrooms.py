'''
env：
python3
numpy，pandas，fastai，torch，zipfile，matplotlib，time，os，PIL
（除了fastai外为最新版本）
借助云服务器运行
'''
from fastai.vision import *
from fastai.metrics import error_rate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import pandas as pd
import zipfile

import matplotlib.pyplot as plt
import time
import os

from fastai.callbacks import ActivationStats
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

classes = ['Boletus','Entoloma','Russula','Suillus','Lactarius','Amanita','Agaricus','Hygrocybe','Cortinarius']#蘑菇的类别

path = Path('/kaggle/input')#输出到kaggle输出文件夹
dest = path
dest.mkdir(parents=True, exist_ok=True)#创造目录

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.35,
                                  ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#读入数据并进行数据预处理
data.classes#显示种类
data.show_batch(rows=3, figsize=(7,8))#显示部分数据

data.classes, data.c, len(data.train_ds), len(data.valid_ds), len(data.test_ds)#显示训练集信息

learn = cnn_learner(data, models.resnet50, metrics=accuracy)#建立模型

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

learn.fit_one_cycle(8, wd=0.9)#初始训练
learn.unfreeze()

learn.lr_find(start_lr = slice(1e-5),end_lr=slice(1))
learn.recorder.plot()#绘制损失和学习率关系图

learn.fit_one_cycle(4, max_lr=slice(1e-5, 2e-3), pct_start=0.8, wd=0.9)
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-8), pct_start=0.8, wd=0.9)

interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))#列出损失最大样本
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)#绘制混淆矩阵

preds_test,y_test, losses_test= learn.get_preds(ds_type=data.test_ds, with_loss=True)
print("Accuracy on test set: ", accuracy(preds_test,y_test).item())#测试结果