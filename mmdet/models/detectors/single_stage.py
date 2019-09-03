import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result
import numpy as np
from nms import nms
from mmdet.core.post_processing.bbox_nms import multiclass_nms
import torch

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        # raise NotImplementedError
        bbox_results_ret=[]

        for i, img in enumerate(imgs):
            x = self.extract_feat(img)
            outs = self.bbox_head(x)  

            bbox_inputs = outs + (img_metas[i], self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]

            bbox_results_ret.append(bbox_results[0][0])
 
        
        ret = np.concatenate(bbox_results_ret, axis=0)

        bboxes = ret[:,:4].copy()
        bboxes[:,2] = ret[:,2] - ret[:, 0]
        bboxes[:,3] = ret[:,3] - ret[:, 1]
        # scores = ret[:,4].tolist()

        multiscores = ret[:,4].reshape(ret[:,4].shape[0],1)
        zeros_dim = np.zeros(multiscores.shape)

        multiscores = np.concatenate((zeros_dim, multiscores), axis=1)

        det_bboxes, det_labels = multiclass_nms(torch.Tensor(ret[:,:4]), torch.Tensor(multiscores),self.test_cfg.score_thr, self.test_cfg.nms,
                                                    self.test_cfg.max_per_img)

        bboxes = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)]

        return bboxes[0]
