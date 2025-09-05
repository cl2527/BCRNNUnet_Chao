import os
import torch
import shutil
import random

import numpy as np

from utils import logger

def setGPU(opt, hvd = None):

    if opt['use_gpu']:

        if "enable_ddp" in opt and opt['enable_ddp']:
            
            torch.cuda.set_device(hvd.local_rank())
            torch.set_num_threads(1)
            logger.print_log("----->>>> Distributed training, local rank [%s/%s]..." % (hvd.local_rank(),hvd.size()))

            opt["hvd_local_rank"] = hvd.local_rank()
            opt["hvd_size"] = hvd.size()
            opt["hvd_rank"] = hvd.rank()
        
        elif opt['use_multi_gpus']:
            
            gpu_ids = ''
            for idx, gpu_id in enumerate(opt['gpu_ids']):
                if idx == 0:
                    n_add = str(gpu_id)
                else:
                    n_add = "," + str(gpu_id)
                gpu_ids += n_add

            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            logger.print_log("----->>>> Multiple GPUs %s are set up ..." % gpu_ids)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']

            if torch.cuda.is_available():
                logger.print_log("----->>>> GPU %s is set up ..." % opt['gpu_id'])

def setFoldersLoggers(opt, split_id = None):

    if 'isbi' in opt['dataset']:
        opt['data_path'] = os.path.join(opt['datasets_path'], opt['dataset'][:-3])
    else:
        opt['data_path'] = os.path.join(opt['datasets_path'], opt['dataset'])

    if opt['mode'] == 'ensemble':
        opt['log'] = os.path.join(opt['logs_path'], opt['dataset'], (opt['model']), str(split_id))
    else:
        if 'isbi' in opt['dataset']:
            opt['log'] = os.path.join(opt['logs_path'], opt['dataset'], (opt['model']), str(opt['rand_split_id']))
        else: 
            opt['log'] = os.path.join(opt['logs_path'], opt['dataset'], (opt['model']), str(opt['rand_split_id']))

    if opt['remove_logs'] and os.path.exists(opt['log']):
        shutil.rmtree(opt['log'])

    os.makedirs(opt['log'], exist_ok = True)

    setLoggers(opt['log'], opt['rand_split_id'])

    if split_id is not None:
        logger.print_log("----->>>> Log folder %s is created ..." % opt['log'])
    logger.print_log("----->>>> Data set path %s is set ..." % opt['data_path'])

def setExcelResultPath(opt):
        
    opt['excel_file_path'] = os.path.join(opt['logs_path'], opt['dataset'], (opt['model']))
    os.makedirs(opt['excel_file_path'], exist_ok = True)
    if opt['model'] == 'lst' or "prob" in opt:
        opt['excel_file_name'] = os.path.join(opt['excel_file_path'], 'result_%.1f.xls' % opt["prob"])
    else:
        opt['excel_file_name'] = os.path.join(opt['excel_file_path'], 'result.xls')

    setLoggers(opt['excel_file_path'])

    print("----->>>> Excel result folder is %s ..." % opt['excel_file_path'])

def setLoggers(log_path, split_id = "None"):
 
    logger.init(log_path, split_id)

def setCropSize(opt):

    crop_size = []
    size_list = opt['crop_size'][1:-1].split(',')

    for size_ in size_list:
        crop_size.append(int(size_))

    opt['crop_size'] = crop_size    

    print("----->>>> Crop size for MRI img is %s ..." % opt['crop_size'])

def setSeed(seed = 10):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def setPrintModelStructure(opt, model):

    is_print = opt['is_print_model']

    if is_print:
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print(model)
        logger.print_log('----->>>> Total number of parameters: %d' % num_params)

def setLaplacianFilterLoss(opt):

    if opt['is_sobel']:
        sobel_type = opt["sobel_type"]
        opt["loss"].append("sobel_loss")
        opt["loss_ws"].append(sobel_type)

    if opt['is_laplacian']:
        lap_coef = float(opt["lap_coef"])
        opt["loss"].append("lap_loss")
        opt["loss_ws"].append(lap_coef)