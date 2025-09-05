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
# import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from utils import logger
from collections import deque, OrderedDict
from scipy.ndimage.measurements import label as getComponents
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier




def computeLoss(criterion, outs, lbls, cur_epoch, opt):

    n_cris = len(criterion)
    enable_alpha_loss = opt["enable_alpah_loss"]
    total_epochs = opt["epochs"]

    if enable_alpha_loss:
        one_minus_alpha = (1.0 - 0.01) / total_epochs * cur_epoch + 0.01
        alpha = 1.0 - one_minus_alpha
    else:
        one_minus_alpha = 1.0 
        alpha = 1.0

    if n_cris == 1:
        loss = criterion[0](outs, lbls)
    elif n_cris == 2:
        loss_1 = criterion[0](outs, lbls)
        loss_2 = criterion[1](outs, lbls)
        loss = loss_1 + loss_2
    elif n_cris == 3:
        loss_1 = criterion[0](outs, lbls)
        loss_2 = criterion[1](outs, lbls)
        loss_3 = criterion[2](outs, lbls)
        loss = alpha * (loss_1 + loss_2) + one_minus_alpha * loss_3
    elif n_cris == 4:
        loss_1 = criterion[0](outs, lbls)
        loss_2 = criterion[1](outs, lbls)
        loss_3 = criterion[2](outs, lbls)
        loss_4 = criterion[3](outs, lbls)
        loss = alpha * (loss_1 + loss_2) + one_minus_alpha * (loss_3 + loss_4)
    else:
        raise ValueError("---->>>> Not supported for more than two losses chosen ...")

    return loss


class msgSender(object):

    def __init__(self, enable, sp_msg):

        self.enable = enable
        self.sp_msg = sp_msg
        self.webhook_url = 'https://hooks.slack.com/services/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'

        if self.enable:
            logger.print_log("----->>>> Msg sender enabled with special message: %s.\n" % sp_msg)
        else:
            logger.print_log("----->>>> Msg sender not enabled.")

    def send(self, msg):

        if self.sp_msg == '':
            slack_data = {'text': msg}
        else:
            sp_msg = self.sp_msg + '\n'
            slack_data = {'text': (sp_msg + msg)}

        if self.enable:
            try:
                response = requests.post(
                    self.webhook_url,
                    data = json.dumps(slack_data),
                    headers = {'Content-Type': 'application/json'}
                )

                if response.status_code != 200:
                    logger.print_log('Request to slack returned an error %s, the response is:\n%s' % (response.status_code, response.text))
            except:
                print("-----------------------------------------------------------------------")
                print("Error! There is no internet connection for sending the msg...")
                print("-----------------------------------------------------------------------")

class metrics(object):

    def __init__(self,
        metrics = ['dice', 'ldice', 'lfpr', 'ltpr', 'precision', 'sensitivity'],
        sample_length = 1,
        is_logged = True,
        is_test_print = False
    ):

        self.supported_metrics = {
            'dice': self.dice,
            'ldice': self.ldice,
            'precision': self.precision,
            'sensitivity': self.sensitivity,
            'iou': self.iou,
            'lfpr': self.lfpr,
            'ltpr': self.ltpr,
        }

        self.vox_val = {'po': 0, 'tr': 0, 'in': 0,'or': 0}
        self.lesion_val = {'ltp':0, 'lfp':0, 'rl': 0, 'pl': 0}

        self.eps = 1e-8
        self.sample_length = sample_length # The length of training, validation or testing data
        self.metrics = [m for m in metrics if m in self.supported_metrics]
        self.is_logged = is_logged
        self.is_test_print = is_test_print
        self.comp_filter = np.ones((3, 3, 3), dtype=np.int)

        self.resetScores()

    def getVoxelWiseValues(self, x, y):

        self.Po = np.sum(x) * 1.0
        self.Tr = np.sum(y) * 1.0
        self.In = np.sum(np.logical_and(x, y)) * 1.0
        self.Or = self.Po + self.Tr - self.In

        self.vox_val['po'] += self.Po
        self.vox_val['tr'] += self.Tr
        self.vox_val['in'] += self.In
        self.vox_val['or'] += self.Or

    def getLesionWiseValues(self, pre, ref):

        pre_labels, n_pre_lesions = getComponents(pre, self.comp_filter)
        ref_labels, n_ref_lesions = getComponents(ref, self.comp_filter)

        self.Pl = n_pre_lesions
        self.Rl = n_ref_lesions
        self.lesion_val['pl'] += self.Pl
        self.lesion_val['rl'] += self.Rl

        self.Ltp = 0
        self.tp_mask = np.zeros(pre.shape)
        self.fn_mask = ref.copy()
        for i in range(1, n_ref_lesions+1):
            pos = (ref_labels == i)
            ref_label_i = pos.astype(int)
            if np.sum(ref_label_i * pre) > 0:
                self.Ltp += 1
                self.tp_mask[pos] = 1
                self.fn_mask[pos] = 0

        self.Lfp = 0
        self.fp_mask = np.zeros(pre.shape)
        for i in range(1, n_pre_lesions+1):
            pos = (pre_labels == i)
            pre_label_i = pos.astype(int)
            if np.sum(pre_label_i * ref) == 0:
                self.Lfp += 1
                self.fp_mask[pos] = 1

        self.lesion_val['ltp'] += self.Ltp
        self.lesion_val['lfp'] += self.Lfp

    def getLesionWiseTPMask(self):
        
        return self.tp_mask.astype(float)
    
    def getLesionWiseFPMask(self):
        
        return self.fp_mask.astype(float)
    
    def getLesionWiseFNMask(self):

        return self.fn_mask.astype(float)

    def getPredMaskWithTPFP(self):

        return self.tp_mask + self.fp_mask * 2 + self.fn_mask * 3

    def computeScore(self, pre, lbl, img_size, folder_num = None):

        if type(pre) == np.ndarray:
            x = np.reshape(pre, img_size).astype(int)
        else:
            x = (pre.view(img_size)==1).cpu().detach().numpy().astype(int)
        y = (lbl.view(img_size)==1).cpu().detach().numpy().astype(int)

        if "ltpr" in self.metrics or "lfpr" in self.metrics:
            self.getLesionWiseValues(x, y)

        self.getVoxelWiseValues(x, y)

        self.scores = []
        self.scoreDict = {}

        if folder_num:
            info = 'Image Folder: ' + str(folder_num) + ', '
        else:
            info = ''

        for k, metric in enumerate(self.metrics):

            func = self.supported_metrics[metric]
            score = func()
            self.scores.append(score)
            self.scoreDict[metric] = score
            self.avg_scores[k] += score / self.sample_length

            info += metric + ': %.5f, '

        if self.is_logged:
            logger.print_log((info % tuple(self.scores)))
        else:
            if self.is_test_print:
                print((info % tuple(self.scores)))

        return self.scoreDict

    def getScoreByName(self, name):

        pos = self.metrics.index(name)

        return self.avg_scores[pos]
    
    def getWeightedScore(self):

        dice_val = self.avg_scores[self.metrics.index("dice")]
        lfpr_val = 1.0 - self.avg_scores[self.metrics.index("lfpr")]
        ltpr_val = self.avg_scores[self.metrics.index("ltpr")]
        prec_val = self.avg_scores[self.metrics.index("precision")]
        sens_val = self.avg_scores[self.metrics.index("sensitivity")]

        voxel_wise_score = 1.0 / 6 * (dice_val + prec_val + sens_val)
        lesion_wise_score = 1.0 / 4 * (lfpr_val + ltpr_val)

        overall_score = voxel_wise_score + lesion_wise_score

        logger.print_log("Overall validation score is %.4f" % overall_score)

        return overall_score

    def getAvgScores(self, total = 0):

        info = ''
        for metric in self.metrics:
            info += 'Avg ' + metric + ': %.5f, '

        if total:
            info = 'total ' + info
            scores = []
            for metric in self.metrics:
                func = self.supported_metrics[metric]
                score = func(total = total)
                scores.append(score)

            if self.is_logged:
                logger.print_log((info % tuple(scores)))
            else:
                if self.is_test_print:
                    print((info % tuple(scores)))

            return info, scores

        info = 'normal ' + info
        if self.is_logged:
            logger.print_log((info % tuple(self.avg_scores)))
        else:
            if self.is_test_print:
                print((info % tuple(self.avg_scores)))

        return info, self.avg_scores

    def resetScores(self):

        self.avg_scores = []
        for _ , _ in enumerate(self.metrics):
            self.avg_scores.append(0.0)

        for k, _ in self.vox_val.items():
            self.vox_val[k] = 0.0

        for k, _ in self.lesion_val.items():
            self.lesion_val[k] = 0.0

    def ltpr(self, total = 0):

        if total:
            return self.lesion_val['ltp'] / (self.lesion_val['rl'] + self.eps)
        else:
            return self.Ltp / (self.Rl + self.eps)

    def lfpr(self, total = 0):

        if total:
            return self.lesion_val['lfp'] / (self.lesion_val['pl'] + self.eps)
        else:
            return self.Lfp / (self.Pl + self.eps)

    def ldice(self, total = 0):

        if total:
            return (2.0 * self.lesion_val['ltp']) / (self.lesion_val['pl'] + self.lesion_val['rl'] + self.eps)
        else:
            return (2.0 * self.Ltp) / (self.Pl + self.Rl + self.eps)

    def dice(self, total = 0):

        if(total):
            return (2.0 * self.vox_val['in']) / (self.vox_val['po'] + self.vox_val['tr'] + self.eps)
        else:
            return (2.0 * self.In) / (self.Po + self.Tr + self.eps)

    def precision(self, total = 0):

        if(total):
            return self.vox_val['in'] / (self.vox_val['po']+ self.eps)
        else:
            return self.In / (self.Po + self.eps)

    def sensitivity(self, total = 0):

        if(total):
            return self.vox_val['in'] / (self.vox_val['tr'] + self.eps)
        else:
            return self.In / (self.Tr + self.eps)

    def iou(self, total = 0):

        if(total):
            return self.vox_val['in'] / (self.vox_val['or'] + self.eps)
        else:
            return self.In / (self.Or + self.eps)

class modelSaver():

    def __init__(self, save_path, save_freq, n_checkpoints = 10):

        self.save_path = save_path
        self.save_freq = save_freq
        self.best_score = -1e6
        self.best_loss = 1e6
        self.n_checkpoints = n_checkpoints
        self.epoch_fifos = deque([])
        self.score_fifos = deque([])
        self.loss_fifos = deque([])

        self.initModelFifos()

    def initModelFifos(self):

        epoch_epochs = []
        score_epochs = []
        loss_epochs  = []

        file_list = glob.glob(os.path.join(self.save_path, '*epoch*.pth'))
        if file_list:
            for file_ in file_list:
                file_name = "net_epoch_(.*)_score_.*.pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    epoch_epochs.append(int(result[0]))

                file_name = "best_score_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    score_epochs.append(int(result[0]))

                file_name = "best_loss_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    loss_epochs.append(int(result[0]))


        score_epochs.sort()
        epoch_epochs.sort()
        loss_epochs.sort()

        if file_list:
            for file_ in file_list:
                for epoch_epoch in epoch_epochs:
                    file_name = "net_epoch_" + str(epoch_epoch) + "_score_.*.pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.epoch_fifos.append(result[0])

                for score_epoch in score_epochs:
                    file_name = "best_score_.*_net_epoch_" + str(score_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.score_fifos.append(result[0])
                
                for loss_epoch in loss_epochs:
                    file_name = "best_loss_.*_net_epoch_" + str(loss_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.loss_fifos.append(result[0])

        logger.print_log("----->>>> BEFORE: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)), is_print_file = False)

        self.updateFIFOs()

        logger.print_log("----->>>> AFTER: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)), is_print_file = False)

    def saveModel(self, model, epoch, avg_score, loss=None):

        torch.save(model.state_dict(), os.path.join(self.save_path, 'net_latest.pth'))

        if epoch % self.save_freq == 0:

            file_name = ('net_epoch_%d_score_%.4f.pth' % (epoch, avg_score))
            self.epoch_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if avg_score >= self.best_score:

            self.best_score = avg_score
            file_name = ('best_score_%.4f_net_epoch_%d.pth' % (avg_score, epoch))
            self.score_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if loss is not None and loss <= self.best_loss:

            self.best_loss = loss
            file_name = ('best_loss_%.4f_net_epoch_%d.pth' % (loss, epoch))
            self.loss_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        self.updateFIFOs()

    def updateFIFOs(self):

        while(len(self.epoch_fifos) > self.n_checkpoints):
            file_name = self.epoch_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            os.remove(file_path)



def constructTrainingMsg(opt):

    model = opt['model']
    dataset = opt['dataset']
    split = opt['rand_split_id']
    epochs = opt['epochs']

    msg = ("%s %s split %s finished %d epochs training." % (model, dataset, split, epochs))

    return msg

def constructAllSplitTestingMsg(opt, df):

    data = list(df.mean())
    model = opt['model']
    dataset = opt['dataset']

    msg_fragments = []

    msg_fragments.append(("%s %s results:\n" % (model, dataset)))

    # based on best validation epoch
    msg_fragments.append(("Best Val Dice: %.4f.\n" % data[0]))
    msg_fragments.append(("Best # Epochs: %d.\n" % data[1]))
    msg_fragments.append(("Best Test Dice: %.4f.\n" % data[2]))
    msg_fragments.append(("Best Test LFPR: %.4f.\n" % data[3]))
    msg_fragments.append(("Best Test LTPR: %.4f.\n" % data[4]))
    msg_fragments.append(("Best Test Prec: %.4f.\n" % data[5]))
    msg_fragments.append(("Best Test Sens: %.4f.\n" % data[6]))

    # based on the lastest epoch
    msg_fragments.append(("Last Val Dice: %.4f.\n" % data[7]))
    msg_fragments.append(("Last # Epochs: %d.\n" % data[8]))
    msg_fragments.append(("Last Test Dice: %.4f.\n" % data[9]))
    msg_fragments.append(("Last Test LFPR: %.4f.\n" % data[10]))
    msg_fragments.append(("Last Test LTPR: %.4f.\n" % data[11]))
    msg_fragments.append(("Last Test Prec: %.4f.\n" % data[12]))
    msg_fragments.append(("Last Test Sens: %.4f.\n" % data[13]))

    msg = ''.join(msg_fragments)

    return msg

def constructSingleSplitTestingMsg(opt, n_scores, n_epoch=-1):

    model = opt['model']
    dataset = opt['dataset']
    split = opt['rand_split_id']

    msg_fragments = []

    if "loss" in opt['load_ckpt']:
        s_type = "loss"
    else:
        s_type = "Dice"

    msg_fragments.append(("%s %s split %s finished training.\n" % (model, dataset, split)))
    msg_fragments.append(("Model with best %s is at Epoch %d used for evaluation.\n" % (s_type, n_epoch)))
    msg_fragments.append("Dice: %.5f \n" % (n_scores[0]))
    msg_fragments.append("LDice: %.5f \n" % (n_scores[1]))
    msg_fragments.append("LFPR: %.5f \n" % (n_scores[2]))
    msg_fragments.append("LTPR: %.5f \n" % (n_scores[3]))
    msg_fragments.append("Prec: %.5f \n" % (n_scores[4]))
    msg_fragments.append("Sens: %.5f \n" % (n_scores[5]))

    # ['dice', 'precision', 'sensitivity', 'iou']
    msg = ''.join(msg_fragments)

    return msg


def readTrainValTestIds(root_dir = '.', file_id = 0, split = 'train'):

    file_name = os.path.join(root_dir, "data_split_" + str(file_id) + '.json')

    params = None
    with open(file_name) as x:
        params = json.load(x)

    return params[split]

def getFoldersById(folder_ids, root_dir = '.'):

    folder_paths = []

    for folder_id in folder_ids:

        folder_path = root_dir + '/' + folder_id
        folder_paths.append(folder_path)

    return folder_paths

def listFolders(root_dir = '.'):

    return [os.path.join(root_dir, file_name)
        for file_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, file_name))]

def listFilesWithSuffix(rootDir = '.', suffix = None):

    if suffix:
        res = [os.path.join(rootDir, file_name)
            for file_name in os.listdir(rootDir)
                if os.path.isfile(os.path.join(rootDir, file_name)) and file_name.endswith(suffix)]
    else:
        res = [os.path.join(rootDir, file_name)
            for file_name in os.listdir(rootDir)
                if os.path.isfile(os.path.join(rootDir, file_name))]

    return res

def savePredictionsWithHeader(opt, data, img_num):

    dataset_name = opt['dataset']
    dataset_path = opt['datasets_path']
    tgt_path = os.path.join(opt['log'], "preds", opt["load_ckpt"])
    os.makedirs(tgt_path, exist_ok=True)

    num = str(img_num).zfill(4)
    header_file_name = None

    iso_path = os.path.join(dataset_path, dataset_name, "data", num)
    files = os.listdir(iso_path)
    iso_file_path = None
    for file_ in files:
        if "T2_to_T2FLAIR_brain" in file_:
            header_file_name = file_
            iso_file_path = os.path.join(iso_path, file_)
            break
    temp = nib.load(iso_file_path)
    iso_affine = temp.affine
    iso_header = temp.header

    pics = header_file_name.split("_")
    new_name = "_".join(pics[:4]) + "_mask_cnn.nii.gz"
    tgt_path = os.path.join(tgt_path, new_name)

    out = nib.Nifti1Image(data.astype(float), iso_affine, iso_header)
    nib.save(out, tgt_path)

    print("%s is saved..." % tgt_path)

def savePredictions(opt, data, img_num, score = 0, score_name = '', prob_save = False):

    dataset_name = opt['dataset']
    log_path = opt['log']

    if 'isbi_15' in dataset_name or "FMRI" in dataset_name:

        num = str(img_num).zfill(4)
        img_num_str = str(num)[:2]+ '_' + str(num)[2:]
        teamname = 'YOLO'
        file_format = 'nii'
        file_name = ("test%s_%s.%s" % (img_num_str, teamname, file_format))
        result_path = os.path.join(opt['log'], 'pred')
        os.makedirs(result_path, exist_ok = True)
        file_path = os.path.join(result_path, file_name)

        saveNii(data, file_path)

    else:

        img_num = str(img_num)
        if prob_save:
            file_name = ("%s_test_prob%s_%.6f.nii.gz" % (img_num, score_name, score))
        else:
            if "prob" in opt.keys():
                file_name = ("%s_test_%s_%.3f_prob_%.2f.nii.gz" % (img_num, score_name, score, opt["prob"]))
            else:
                file_name = ("%s_test_%s_%.6f.nii.gz" % (img_num, score_name, score))
       
        file_path = os.path.join(log_path, file_name)
        saveNii(data, file_path)

def loadNii(file_name):

    return nib.load(file_name).get_data()

def saveNii(data, file_name, file_name_sample=''):

    if file_name_sample:
        nib.save(nib.Nifti1Image(data, None, nib.load(file_name_sample).header), file_name)
    else:
        nib.save(nib.Nifti1Image(data, None, None), file_name)

def zipDir(opt, file_name = 'output_seg.zip'):

    dataset_name = opt['dataset']

    if 'isbi_15' in dataset_name:

        path = os.path.join(opt['log'], 'pred')
        os.makedirs(path, exist_ok = True)
        zip_path = os.path.join(opt['log'], file_name)
        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        len_dir_path = len(path)
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path , file_path[len_dir_path:])
                print('Zipping subject: ', str(file))
        zipf.close()

def loadH5(file_name):

    h5f = h5py.File(file_name, 'r')
    data = h5f['data'][:]
    h5f.close()

    return data

def saveH5(data, file_name, file_name_sample=''):

    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('data', data=data)
    h5f.close()


def lowPrecisionVoting(pred, thresh = 0.5):

    size_predicts = len(pred)
    size_samples = len(pred[0])
    seg = [0 for i in range(size_samples)]
    
    for idx in range(size_samples):
        img_size = pred[0][idx].shape
        sample = torch.zeros(img_size)
        sample = sample.view(-1, 2)
        for j in range(size_predicts):
            tmp = torch.tensor(pred[j][idx]).view(-1, 2)
            sample += F.softmax(tmp, dim = 1)
        
        sample = sample[:,1] / size_predicts
        sample = (sample > thresh).int()
        sample = sample.cpu().detach().numpy()
        sample = np.reshape(sample, img_size[:-1]) 
        # seg[idx] = np.argmax(sample, axis = -1)
        seg[idx] = np.transpose(sample, (1, 2, 0))
        seg[idx] = seg[idx].astype(float)

    return seg


def majorityVoting(pred):

    size_predicts = len(pred)
    size_samples = len(pred[0])
    seg = [0 for _ in range(size_samples)]

    for idx in range(size_samples):
        sample = 0
        for j in range(size_predicts):
            sample += pred[j][idx]
        sample = sample / size_predicts
        sample = sample.sigmoid().numpy()
        seg[idx] = np.transpose(sample>=0.5, (1, 2, 0)).astype(float)

    return seg

# def plotHeatmap(data):

#     file_name = 'attention.pdf'
#     os.makedirs('./plots', exist_ok = True)
#     save_path = os.path.join('./plots', file_name)

#     fig= plt.subplots()
#     ax = sns.heatmap(data) # sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#     plt.savefig(save_path)

def convert_state_dict(state_dict, is_multi = False):
    
    new_state_dict = OrderedDict()

    if is_multi:
        if next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is a DataParallel model_state

        for k, v in state_dict.items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
    else:

        if not next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is not a DataParallel model_state

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    
    return new_state_dict

def reshapeToOriSizeWithCrops(preds, crops):

    out_preds = []
    for idx, param in enumerate(crops):
        w,h,d = param[:3]
        img = np.zeros([w,h,d])
        sx, sy, sz, ex, ey, ez = param[3:]
        img[sx:ex+1, sy:ey+1, sz:ez+1] = preds[idx]
        out_preds.append(img)

    return out_preds

def getLesionWeights(mask):

    labels, n_lesions = getComponents(mask, np.ones((3, 3, 3), dtype=np.int))

    weights = np.ones(mask.shape).astype(int)

    for i in range(1, n_lesions+1):
        label_i = (labels == i)
        lesion_size = np.sum(label_i)
        if lesion_size < 20:
            weights[label_i] = 2
        elif lesion_size < 40: 
            weights[label_i] = 1.5

    return np.uint8(weights)

def constructKnnDatabse(feas, lbls):

    _, n_feas, h, w, d = feas.size()
    feas = feas.contiguous().view(n_feas, h*w*d)
    lbls = lbls.contiguous().view(h*w*d)
    locs_0 = (lbls==0)
    locs_1 = (lbls==1)

    feas_0 = feas[:,locs_0]
    feas_1 = feas[:,locs_1]

    feas_0_size = locs_0.sum()
    down_size = locs_1.sum() 
    
    indice = random.sample(range(int(feas_0_size)), int(down_size))
    indice = torch.tensor(indice).to(feas.device)

    feas_0 = feas_0[:,indice]

    out_feas = torch.cat([feas_0, feas_1], dim=1)
    out_lbls = torch.cat([torch.zeros([down_size]), torch.ones([down_size])])

    return out_feas, out_lbls

def knnBatchFuse(feas, lbls):
    
    feas = torch.cat(feas, dim=1)
    lbls = torch.cat(lbls)

    return feas, lbls

def knnPredict(out, knn_model):
    
    n_feas, h, w, d = out.size()[1:]    
    out = out.view(n_feas, h*w*d).detach().cpu().numpy()

    mask = knn_model.predict(np.transpose(out, [1,0]))


    return np.reshape(mask, [h, w, d])

def getLesions(mask):

    mask_cpu = mask.cpu().numpy()
    lbls, n_lesions = getComponents(mask_cpu, np.ones((3, 3, 3), dtype=np.int))

    return lbls, n_lesions

def kLesionFuse(outs, masks):

    n_samples = len(outs)

    knn_feas = []
    for idx in range(n_samples):
        n_features, h, w, d = outs[idx][0,:].size()
        out = outs[idx].view(n_features, h*w*d)
        lbl = masks[idx].view(h, w, d)
        mask_cpu, n_lesions = getLesions(lbl)
        
        mask = torch.tensor(mask_cpu).to(outs[0].device).view(h*w*d)
        c_means = torch.zeros(n_features, n_lesions).to(outs[0].device)
        for j in range(1, n_lesions+1):
            locs = (mask==j)
            c_means[:,j-1] = out[:,locs].mean(dim=1)
        
        knn_feas.append(c_means)

    knn_feas = torch.cat(knn_feas, dim=1)
    print("---->>>> Knn database constructed %d lesion centers with feature dim %d" % (knn_feas.size()[1], knn_feas.shape[0]))
    
    total_lesions = knn_feas.size()[1] 
    feas = []
    lbls = []
    print("---->>>> Geting training Knn features...")
    for idx in range(n_samples):
        n_features, h, w, d = outs[idx][0,:].size()
        out = outs[idx].view(n_features, h*w*d)
        mask = masks[idx].view(h*w*d)
        locs_0 = (mask == 0)
        locs_1 = (mask == 1)

        fea_0 = out[:,locs_0]
        fea_1 = out[:,locs_1]
        lbl_0 = mask[locs_0]
        lbl_1 = mask[locs_1]

        feas_0_size = locs_0.sum()
        down_size = 10*locs_1.sum() 
        
        indice = random.sample(range(int(feas_0_size)), int(down_size))
        indice = torch.tensor(indice).to(outs[0].device)

        fea_0 = fea_0[:,indice]
        lbl_0 = lbl_0[indice]
        out = torch.cat([fea_0, fea_1], dim=1)
        lbl = torch.cat([lbl_0, lbl_1])
        fea = torch.zeros(total_lesions, out.size()[1])
        for j in range(total_lesions):
            fea[j,:] = torch.norm(out-knn_feas[:,j].view(n_features, 1),2,dim=0).cpu()
        
        feas.append(fea)
        lbls.append(lbl)
    
    feas = torch.cat(feas, dim=1)
    feas = np.transpose(feas.cpu().numpy(), [1,0])
    lbls = torch.cat(lbls).cpu().numpy()

    print("---->>>> Fitting RandomForestClassifier with (%d, %d)..." % (feas.shape[0], feas.shape[1]))
    clf = RandomForestClassifier()
    clf.fit(feas, lbls)
    print("---->>>> RandomForestClassifier fit with the data...")
    

    return knn_feas, clf

def rfPredict(out, knn_feas, clf):
    
    total_lesions = knn_feas.size()[1]
    n_feas, h, w, d = out.size()[1:]    
    out = out.view(n_feas, h*w*d)
    print("---->>>> Testing: got raw features")
    fea = torch.zeros(total_lesions, h*w*d)
    for j in range(total_lesions):
        fea[j,:] = torch.norm(out.view(n_feas, h*w*d)-knn_feas[:,j].view(n_feas, 1),2,dim=0).cpu()
    print("---->>>> Testing: got Knn features")
    fea = np.transpose(fea.cpu().numpy(), [1,0])
    mask = clf.predict(fea)
    print("---->>>> fea: got prediction")

    return np.reshape(mask, [h, w, d])

def eliminateSmallLesion(mask, t_size=5):

    labels, n_lesions = getComponents(mask, np.ones((3, 3, 3), dtype=np.int))

    for i in range(1, n_lesions+1):
        pos = (labels == i)
        if np.sum(pos) < t_size:
            mask[pos] = 0
    
    return mask

def growLesions(sims, mask, lbl, opt):

    # if "NMS" in opt["model"]:
    #     mask = mask.float()
    # else:
    mask = (mask.sigmoid()>=0.5).float()
    # mask = mask.float() * lbl.float()
    # mask = mask.float()
    sims = [(sim.sigmoid()>=0.01).float() for sim in sims]
    sims = [sim.squeeze(1) for sim in sims]
    mask = mask.squeeze(1)

    if opt["reduce_fa"] == 1:
        elim = torch.zeros(mask.shape).to(mask.device)
        for i in range(6):
            elim += sims[i] * mask
        pos = (elim==0) 
        mask[pos] = 0

    while(1):
        old_mask = mask.clone()
        sim = sims[0] * mask
        mask[:,:-1,:,:] += sim[:,1:,:,:]
        mask = (mask>=1).float()

        sim = sims[1] * mask
        mask[:,1:,:,:] += sim[:,:-1,:,:]
        mask = (mask>=1).float()

        sim = sims[2] * mask
        mask[:,:,:-1,:] += sim[:,:,1:,:]
        mask = (mask>=1).float()

        sim = sims[3] * mask
        mask[:,:,1:,:] += sim[:,:,:-1,:]
        mask = (mask>=1).float()

        sim = sims[4] * mask
        mask[:,:,:,:-1] += sim[:,:,:,1:]
        mask = (mask>=1).float()

        sim = sims[5] * mask
        mask[:,:,:,1:] += sim[:,:,:,:-1]
        mask = (mask>=1).float()

        if (mask - old_mask).sum() == 0:
            break

    return mask