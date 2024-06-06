# coding:utf-8
import random

import numpy as np
import torch
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve,auc
import argparse
from os.path import join
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class ROCMetric():

    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    # if len(output.shape) == 3:
    #     output = np.expand_dims(output.float(), axis=1)

    # print("o",output.shape)
    # print("t", target.shape)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    # predict = (output > 0).float()##P
    predict = output.float()  ##P
    pixel_labeled = (target > 0).float().sum()  ###T
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()  ##TP

    ##根据P、T、TP值计算IoU
    T_num = pixel_labeled
    P_num = (output > 0).float().sum()
    TP_num = pixel_correct

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled, T_num, P_num, TP_num


def batch_intersection_union(output, target, nclass):
    mini = 1
    maxi = 1
    nbins = 1
    # predict = (output > 0).float()
    predict = output.float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled, T_num, P_num, TP_num = batch_pix_accuracy(preds, labels)
        # print(correct)
        # print(labeled)
        inter, union = batch_intersection_union(preds, labels, self.nclass)

        ####图像累积计算IoU值
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        ####单张图像计算IoU值
        # self.total_correct = correct
        # self.total_label = labeled
        # self.total_inter = inter
        # self.total_union = union

        self.single_IOU = TP_num * 1.0 / (T_num + P_num - TP_num)

    def get(self):
        pixAcc = 1.0 * self.total_correct.cpu() / (np.spacing(1) + self.total_label.cpu()).float()
        # print(pixAcc)
        # print(self.total_correct)
        # print(self.total_union)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # print(IoU)
        mIoU = IoU.mean()

        single_IOU = self.single_IOU
        return pixAcc, mIoU, single_IOU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.single_IOU = 0


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC   = ROCMetric(1, 100)
        # self.PD_FA = PD_FA(1,255)
        # self.PD_FA = PD_FA(1,10)
        self.mIoU = mIoU(1)
        # self.save_prefix = '_'.join([args.model, args.dataset])

        #gt_dir = './NUAA_teTCD/gt5'
        #pred_dir = './NUAA_teTCD/pred5'
        gt_dir = './IRSTD_teTCD/gt52'
        pred_dir = './IRSTD_teTCD/pred52'
        gt_dirs = os.listdir(gt_dir)
        #gt_dirs.sort(key=lambda x: int(x[:-4]))
        pred_dirs = os.listdir(pred_dir)
        #pred_dirs.sort(key=lambda x: int(x[:-4]))


        sin_IOU = []


        for ind in range(len(pred_dirs)):  # 读取每一个（图片-标签）对
            transf = transforms.ToTensor()
            # pred = Image.open(pred[ind])
            # print(pred.shape)
            # img=Image.open(preds[ind]+'.png')
            img = Image.open(pred_dir + '/' + pred_dirs[ind])
            # print(pred_dirs[ind])
            #img = img.convert("RGB")
            img = img.convert("L")
            pred = transf(img)
            pred = np.expand_dims(pred.float(), axis=1)
            pred = torch.from_numpy(pred)
            # print("p",pred.shape)

            # label=Image.open(labels[ind]+'.png')
            label = Image.open(gt_dir + '/' + gt_dirs[ind])
            # print(gt_dirs[ind])
            #label = label.convert("RGB")
            label = label.convert("L")
            label = transf(label)
            label = np.expand_dims(label.float(), axis=1)
            label = torch.from_numpy(label)
            # print("l",label.shape)
            # print("pred:", type(pred))
            # print("label:", type(label))

            self.mIoU.update(pred, label)
            self.ROC.update(pred, label)
            #self.PD_FA.update(pred, label)
            _, mean_IOU, singleIOU = self.mIoU.get()
            print('ind:', ind)
            print('mean_IOU:', mean_IOU)
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()


            sin_IOU.append(singleIOU)
        #FA, PD = self.PD_FA.get(len(pred_dirs))
        nn_iou = np.mean(sin_IOU)
        print('mean_IOU:', mean_IOU)
        print("nn_iou", nn_iou)
        # print("false_positive_rate", false_positive_rate)
        # print("ture_positive_rate", ture_positive_rate)
        #
        #
        # plt.figure(1)
        # plt.title('ROC Curve')
        # false_positive_rate2=[t*0.0001 for t in false_positive_rate]
        # plt.xlabel('false_positive_rate')
        # plt.ylabel('ture_positive_rate')
        # my_x_ticks = np.arange(0, 10, 2)
        # plt.xticks(my_x_ticks)
        # plt.plot(false_positive_rate2,ture_positive_rate)
        # plt.show()
        # plt.savefig('ROC.png')



def main(args):
    trainer = Trainer(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')#设置gt_dir参数，存放验证集分割标签的文件夹
    # parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')#设置pred_dir参数，存放验证集分割结果的文件夹
    # parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')#设置devikit_dir文件夹，里面有记录图片与标签名称及其他信息的txt文件
    args = parser.parse_args()
    main(args)  # 执行主函数