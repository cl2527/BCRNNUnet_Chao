import re
import os
import glob
import torch
import shutil
import numpy as np
import torch.optim as optim

import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

from utils import logger
from utils.functions import  metrics, modelSaver, msgSender, convert_state_dict
from loader.BCRNNUnet_Chao_loader import PDBrainDataset
from loader.BrainMotion import BrainMotionDataset
from loader.BrainMotion_stacked_Unet import BrainMotionDataset_StackUnet
from fineTuneCodes.FineTune_loader import BrainMotionFineTuneDataset

import pickle


def loadDataset(opt, batch_size, augmentations = None, split = 'train'):

    name = opt['dataset']
    correction_type = opt['correction_type']
    data_dir = os.path.join(opt['datasets_path'],split)
    data_info_dir_root = opt['data_info_dir_root']
    data_info_dir = os.path.join(data_info_dir_root, split+'.pkl')

    with open(data_info_dir, "rb") as f:
        data_info_list = pickle.load(f)
    

    if name == 'Brain_motio':
        loader = BrainMotionDataset(data_dir, data_info_list, correction_type)
    elif name == 'fine_tune_train':
        loader = BrainMotionFineTuneDataset(data_dir, data_info_list)
    elif name == 'Brain_motion_AF':
        loader = BrainMotionDataset(data_dir, data_info_list, correction_type)
    elif name == 'Brain_motion_AF_Norm':
        loader = BrainMotionDataset(data_dir, data_info_list, correction_type)
    else:
        raise ValueError('Unkown datasets: please define proper dataset name')
    
    if opt['model']=='stacked_unet':
        loader = BrainMotionDataset_StackUnet(data_dir, data_info_list, correction_type)
    
    return loader   

def getcrossEntropyWeight(loader, opt):

    if "lesion_change" in opt["dataset"]:
        return torch.tensor([0.1, 0.9])
    if 'bicams' in opt["dataset"] or '178' in opt["dataset"]:
        return torch.tensor([0.2, 0.8])

    class_rate = loader.getLabelStats()['rateMean']
    logger.print_log("----->>>> Dataset class rate: %.10f" % (class_rate))
    weight = torch.tensor([np.exp(class_rate), np.exp(1-class_rate)])

    return weight

def getDataLoader(opt, augmentations=None, split = 'train', is_ddl=False):
    if(split == 'train'):
        batch_size = opt['batch_size']
        data_shuffle = True
        num_workers = 16
        opt["is_mem"] = True
    else:
        batch_size = 1
        data_shuffle = False
        num_workers = 1
        if "isbi" in opt["dataset"]:
            opt["is_mem"] = False
    
    logger.print_log("----->>>> Loading %s dataset ..." % (split))
    dataset = loadDataset(opt, batch_size, augmentations, split)
    if is_ddl and split == "train":
        # setting distributed learning parameters    
        kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': data_shuffle}
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        data_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=opt["hvd_size"], rank=opt["hvd_rank"])
        loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, **kwargs)
        logger.print_log("----->>>> DDL with %s batch size: %d, # of %s iterations per epoch: %d" %  (split, batch_size, split, int(len(dataset) / batch_size)))
    
    else:
        loader = DataLoader(dataset = dataset,
                            num_workers = num_workers,  
                            batch_size = batch_size,
                            pin_memory = True,
                            shuffle = data_shuffle)
        logger.print_log("----->>>> %s batch size: %d, # of %s iterations per epoch: %d" %  (split, batch_size, split, int(len(dataset) / batch_size)))

    #if split == 'train':
    #    weight = getcrossEntropyWeight(dataset, opt)
    #else:
    weight = None

    return loader, weight

def getOptimizer(opt, model):

    lr = opt['lr']
    total_epochs = opt['epochs']
    scheduler_type = opt['scheduler_type']
    w_decay = opt['weight_decay']
    lr_ed = opt['lr_ed']

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = w_decay)
    logger.print_log("----->>>> Adam optimizer, lr: %.5f, w_decay: %.5f" % (lr, w_decay))

    if scheduler_type == 'MultiStepLR':

        # ms = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]
        ms = [0.5, 0.7, 0.9]
        ms = [np.floor(m * total_epochs).astype(int) for m in ms]
        # gamma = 0.447
        gamma = 0.2

        scheduler = MultiStepLR(optimizer, milestones = ms, gamma = gamma)
        logger.print_log("----->>>> %s scheduler, milestones: %s gamma: %.2f" % (scheduler_type, ms, gamma))

    elif scheduler_type == 'LambdaLR':

        rate = 0.9
        lr_st = 1.0 - lr_ed
        lambdaS = lambda epoch: ((lr_st * (1.0 - epoch / total_epochs) + lr_st) ** (rate))

        scheduler = LambdaLR(optimizer, lr_lambda = lambdaS)
        logger.print_log("----->>>> %s scheduler, %.5f * (1 - epoch / %d)^%.2f " % (scheduler_type, lr, total_epochs, rate))

    return optimizer, scheduler

def getTorchDevice(opt=None):

    if torch.cuda.is_available():
        if opt and opt['use_multi_gpus']:
            device = torch.device('cuda:{}'.format(opt["gpu_ids"][0]))
        else:
            device = torch.device('cuda')
        logger.print_log("----->>>> CUDA is used for training ...")
    else:
        device = torch.device("cpu")
        logger.print_log("----->>>> CPU is used for training ...")

    return device

def getCriterion(opt, w, device):

    ignore_lbl = opt['ignore_label']
    cri_names = opt['loss']
    ws = opt['loss_ws']
    if 'is_voxelweighted' in opt:
        is_voxelweighted = opt['is_voxelweighted']
    else:
        is_voxelweighted = False
    criterion = []
    if 'rand_bce' in cri_names or 'rand_dice_loss' in cri_names:
        w =  torch.tensor([0.9, 0.99])

    for idx, cri_name in enumerate(cri_names):
        if cri_name == 'mse_loss':
            criterion.append(MSE())
        else:
            logger.print_log('Unkown criterion: please define proper criterion name')
        """
        elif cri_name == 'd_mlp_loss':
            criterion.append(DisBCEDiceLoss(loss_w = ws[idx]))
        elif cri_name == 'bce_ortho':
            criterion.append(bceOrtho(weight = w, ignore_index = ignore_lbl, loss_w = ws[idx]))
        elif cri_name == 'bce_dice_ortho':
            criterion.append(bceDiceOrtho(weight = w, ignore_index = ignore_lbl, loss_w = ws[idx]))
        elif cri_name == 'm_bce_dice':
            criterion.append(multiBCEDice(weight = w, loss_w = ws[idx]))
        elif cri_name == '4_bce_dice':
            criterion.append(fourBCEDice(weight = w, loss_w = ws[idx]))
        elif cri_name == 'region_object':
            criterion.append(regionObjLoss(weight = w, loss_w = ws[idx]))
        elif cri_name == 'sim_loss':
            criterion.append(localSimLoss(weight = w, loss_w = ws[idx]))
        elif cri_name == 'sobel_3axis_loss':
            criterion.append(sobel3AxisLoss(ignore_index = ignore_lbl, loss_w = ws[idx], device = device))
        elif cri_name == 'sobel_loss':
            criterion.append(sobelLoss(ignore_index = ignore_lbl, loss_w = ws[idx], device = device))
        elif cri_name == 'lap_loss':
            criterion.append(lapLoss(ignore_index = ignore_lbl, loss_w = ws[idx], device = device))
        elif cri_name == 'dice_loss':
            criterion.append(diceLoss(ignore_index = ignore_lbl, loss_w = ws[idx]))
        elif cri_name == 'rand_bce':
            criterion.append(randBCE(weight = w, ignore_index = ignore_lbl, loss_w = ws[idx]))
        elif cri_name == 'rand_dice_loss':
            criterion.append(randForegroundDice(weight = w, loss_w = ws[idx]))
        elif cri_name == 'focal_loss':
            criterion.append(focalLoss(ignore_index = ignore_lbl, loss_w = ws[idx]))
        elif cri_name == 'focal_mc':
            criterion.append(focalMCLoss(loss_w = ws[idx]))
        elif cri_name == 'd_loss':
            criterion.append(DiscriminativeLoss(loss_w = ws[idx]))
        elif cri_name == 'kd_loss':
            criterion.append(kLesionDiscriminativeLoss(loss_w = ws[idx]))
        elif cri_name == 'hd_loss':
            criterion.append(hdLoss(loss_w = ws[idx]))
        elif cri_name == 'boundary_loss':
            criterion.append(boundaryLoss(loss_w = ws[idx]))
        elif cri_name == 'lap_int_loss':
            criterion.append(lapIntegrationLoss(ignore_index = ignore_lbl, loss_w = ws[idx], device = device))
        elif cri_name == 'vw_bce_loss':
            criterion.append(voxelWeightedBCE(weight = w, loss_w = ws[idx], is_vw = is_voxelweighted))
        elif cri_name == 'vw_dice_loss':
            criterion.append(voxelWeightedDice(weight = w, loss_w = ws[idx], is_vw = is_voxelweighted))
        """

    logger.print_log("----->>>> Using loss criterion %s with weights %s..." % (cri_names, ws))

    return criterion

def getAugmentations(opt):

    aug_list = opt['aug_list']
    augmentations = []

    for aug_names in aug_list:
        augs = []
        for aug_name in aug_names:
            if aug_name == 'rand_crop':
                augs.append(randomCrop3D(opt['crop_size']))
            elif aug_name == 'elastic_grid':
                augs.append(elasticDeformationGrid())
            elif aug_name == 'shift':
                augs.append(randomIntensityShift())
            elif aug_name == 'elastic_pixel':
                augs.append(elasticDeformationPixel())
        augmentations.append(composeAug(augs))

    n_augs = len(augmentations)
    logger.print_log("----->>>> %d sequences of augmentations have been used ..." % n_augs)

    for i in range(n_augs):
        aug_names = aug_list[i]
        print("----->>>>", aug_names)

    return augmentations

def getMetricComputer(opt, val_loader, split = 'train'):

    val_length = len(val_loader)
    is_logged = True if split == 'train' else False
    is_test_print = opt['print_score'] if split != 'train' else False
    
    if "metrics" in opt.keys():
        metrics_ = opt["metrics"]
    else:
        metrics_ = ['dice', 'ldice', 'lfpr', 'ltpr', 'precision', 'sensitivity']
    
    metric = metrics(
        metrics = metrics_,
        sample_length = val_length,
        is_logged = is_logged,
        is_test_print = is_test_print,
        )

    return metric

def getModelSaver(opt):

    model_saver = modelSaver(opt['log'], opt['save_freq'], opt['n_checkpoints'])

    return model_saver

def getMsgSender(opt):

    msg_sender = msgSender(opt['enable_msg'], opt['sp_msg'])

    return msg_sender

def findLastCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("net_epoch_(.*)_score_.*.pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        init_epoch = max(epochs_exist)
    else:
        init_epoch = 0

    score = None
    if init_epoch > 0:
        for file_ in file_list:
            file_name = "net_epoch_" + str(init_epoch) + "_score_(.*).pth.*"
            result = re.findall(file_name, file_)
            if result:
                score = result[0]
                break

    return_name = None
    if init_epoch > 0:
        return_name =  "net_epoch_" + str(init_epoch) + "_score_" + score + ".pth"

    return init_epoch, score, return_name

def findBestCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("best_score_(.*)_net_epoch_.*.pth.*", file_)
            if result:
                epochs_exist.append(result[0])
        score = max(epochs_exist)

        for file_ in file_list:
            file_name = "best_score_" + str(score) + "_net_epoch_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return_name = result[0]
                file_name = "best_score_" + str(score) + "_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                epoch = result[0]
                return epoch, score, return_name

    raise ValueError("can't find checkpoints")

def findCheckpointByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "net_epoch_" + str(epoch) + "_score_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def findBestDiceByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "best_score_.*_net_epoch_" + str(epoch) + ".pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def getTrainModelWithCheckpoints(opt):

    model = getModel(opt)
    init_epoch, score, file_name = findLastCheckpoint(opt['log'])

    if init_epoch > 0:
        logger.print_log("----->>>> Resuming model by loading epoch %s with dice %s" % (init_epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)), is_multi=opt['use_multi_gpus'])
        model.load_state_dict(states)

    return model, init_epoch

def getTestModelWithCheckpoints(opt):

    model = getModel(opt)
    file_name = 'unknown'
    epoch = 'unknown'
    score = '0'
    which_model = 'unknow'

    if opt['load_ckpt'] == 'best':
        epoch, score, file_name = findBestCheckpoint(opt['log'])
        which_model = 'best'
    elif opt['load_ckpt'] == 'last':
        epoch, score, file_name = findLastCheckpoint(opt['log'])
        which_model = 'last'
    elif "epoch" in opt['load_ckpt']:
        epoch = opt['load_ckpt'].split('_')[1]
        file_name = findCheckpointByEpoch(opt['log'], epoch)
        which_model = str(epoch) + 'th'
    elif int(opt['load_ckpt']):
        file_name = findBestDiceByEpoch(opt['log'], opt['load_ckpt'])
        if file_name:
            epoch = str(opt['load_ckpt'])
    else:
        raise ValueError("Not either best, last or epoch")

    if file_name:
        logger.print_log("----->>>> Resuming the %s model by loading epoch %s with dice %s" % (which_model, epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)), is_multi=opt['use_multi_gpus'])
        model.load_state_dict(states)

    info = {
        "file_name": file_name,
        "epoch": int(epoch),
        "score": float(score),
    }

    return model, info


def getDiscriminativeLoss():

    return DiscriminativeLoss()
