import re
import os
import json
import h5py
import glob
import torch
import random
import zipfile
import requests
import matplotlib
from torch import is_signed
matplotlib.use('Agg')

import numpy as np
import nibabel as nib
import nibabel.processing
import nibabel as nibabel 
# import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from utils import logger
from utils.functions import convert_state_dict
from collections import deque, OrderedDict
from scipy.ndimage.measurements import label as getComponents
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


def setlogs(root_path, model_name, file_name, discription):

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    model_path=os.path.join(root_path,model_name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    log_path=os.path.join(model_path,file_name)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    val_loss_path=os.path.join(log_path,"val_loss.txt")
    train_loss_path=os.path.join(log_path,"train_loss.txt")

    if not os.path.exists(val_loss_path):
        val_file = open(val_loss_path, "w") 
        val_file.write(discription) 
        val_file.close() 

    if not os.path.exists(train_loss_path):
        train_file=open(train_loss_path,"w")
        train_file.write(discription)
        train_file.close()
    
    return model_path, log_path, val_loss_path, train_loss_path



def set_model_parallelism(opt, model):
    if opt['use_multi_gpus']:
        model = torch.nn.DataParallel(model, device_ids=opt['gpu_ids'])
        model = model.cuda(opt["gpu_ids"][0])
    else:
        model.cuda()
    return model

def set_loss_device(opt, loss):
    if opt['use_multi_gpus']:
        loss = loss.cuda(opt["gpu_ids"][0])
    else:
        loss.cuda() 
    return loss


def loadTrainModel(opt, model, log_path,PATH):

    if os.path.exists(PATH):
        states = convert_state_dict(torch.load(PATH,map_location='cpu'), is_multi=opt['use_multi_gpus'])
        model.load_state_dict(states)
        
        file_list = glob.glob(os.path.join(log_path, '*epoch*.pt'))
        init_epoch = 0
        if file_list:
            for file_ in file_list:
                init_epoch += 10
        init_epoch = init_epoch - 10
    else:
        init_epoch=0

    return model, init_epoch


def loadPretrainedModel(opt, model, PATH):
    if opt['use_multi_gpus']:
        model = model.cuda(opt["gpu_ids"][0]) 
    else:
        model = model.cuda()
    states = convert_state_dict(torch.load(PATH,map_location='cpu'), is_multi=False)
    model.load_state_dict(states)
    return model
        


def saveTrainModel(model_state_dict,PATH,log_path,epoch,save_freq):
    torch.save(model_state_dict,PATH)
    if epoch%save_freq==0:
        PATH_epoch=os.path.join(log_path,"epoch_%d.pt"%(epoch))
        torch.save(model_state_dict,PATH_epoch)


def loadTestModel(opt, model ,PATH):
    if opt['use_multi_gpus']:
        model = model.cuda(opt["gpu_ids"][0]) 
    else:
        model = model.cuda()
    states = convert_state_dict(torch.load(PATH,map_location='cpu'), is_multi=opt['use_multi_gpus'])
    model.load_state_dict(states)
    return model
    
def getLogPath(root_path, model_name, file_name):

    model_path=os.path.join(root_path,model_name)

    log_path=os.path.join(model_path,file_name)

    return log_path
    
