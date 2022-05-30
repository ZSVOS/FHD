import os
import os.path as osp
import cv2
import argparse
from skimage import io
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class F1_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        return tp, tn, fp, fn


def eval(pred_path, gt_path, use_jpg):
    gt_dir = osp.join(gt_path, 'label')
    gt_list = os.listdir(gt_dir)
    pred_dir = osp.join(pred_path, 'label')

    tp_All, tn_All, fp_All, fn_All = 0, 0, 0, 0
    pbar = tqdm(gt_list)
    for im_name in pbar:
        pbar.set_description(f"Now get {im_name}")
        if use_jpg:    
            predfile = osp.join(pred_dir, im_name[:-3]+'jpg')
            gtfile = osp.join(gt_dir, im_name)
            pred = cv2.imread(predfile)[:, :, 0].flatten()
            gt = io.imread(gtfile).flatten()
        else:
            predfile = osp.join(pred_dir, im_name)
            gtfile = osp.join(gt_dir, im_name)
            pred = cv2.imread(predfile)[:, :, 0].flatten()
            gt = cv2.imread(gtfile)[:, :, 0].flatten()
        
        pred[pred != 0] = 1
        gt[gt != 0] = 1
        f1_loss = F1_Loss().cuda()

        tp, tn, fp, fn = (f1_loss(torch.from_numpy(pred),
                                      torch.from_numpy(gt)
                                      )
                              )
        tp_All += tp
        fp_All += fp
        tn_All += tn
        fn_All += fn

    precision= tp_All / (tp_All + fp_All)
    recall = tp_All / (tp_All + fn_All)
    f1 = 2 * (precision * recall) / (precision + recall)
    iou = tp_All / (tp_All + fn_All + fp_All)
    oa = (tp_All + tn_All) / (tp_All + tn_All + fn_All + fp_All)

    print('Precision: {}, Recall: {}, F1-Score: {}'.format(precision, recall, f1))
    print('IoU: {}, OA: {}'.format(iou, oa))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluation Change Detection.')
    parser.add_argument('--pred_path', help='test file path',
                        default='./output_path/'
                        )
    parser.add_argument('--gt_path', help='test file path',
                        default='./data/DSIFN/test/'
                        )
    parser.add_argument('--use_jpg', help='jpg format GT',
                        default=False
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_dir = './output_path/'
    # LEVIR
    # gt_dir = './data/LEVIR-CD_256x256/test/'
    eval(args.pred_path, args.gt_path, args.use_jpg)
