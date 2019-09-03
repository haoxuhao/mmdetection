'''
合并分割图片测试结果为一个结果
输入：results.pkl 分割图像的测试结果是依次存放的 
[result_image_l_left, result_image_1_right, result_image_2_left, result_image_2_right,...]
调用mmdetection 的nms 去除重复框
'''

from tqdm import tqdm
import pickle
import os
import os.path as osp
import numpy as np
import sys
sys.path.append("/root/mmdetection")
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdet.core import bbox2result
import torch


def read_pkl(pkl_file):
    results = pickle.load(open(pkl_file, 'rb'), encoding='utf-8')
    return results
 
test_images_id_file = "/root/datasets/testset/ImageSets/Main/test_split.txt"
input_results_file = "results_test_leftright.pkl"
output_results_file = "final_results_test_leftright.pkl"

input_results = read_pkl(input_results_file)
with open(test_images_id_file, "r") as f:
    imageids = [file.strip() for file in f.readlines()]
    
score_thresh = 0.4
score_desent = 0.42
nms_cfg = dict(type="soft_nms", iou_thr=0.3)

final_results = []
for i in tqdm(range(0, len(imageids), 2)):
    right_imageid = imageids[i+1]
    left_imageid = imageids[i]
    x_offset_right = int(right_imageid.split("_")[-1])
    x_offset_left = int(left_imageid.split("_")[-1])
    left_results = input_results[i][0]
    right_results = input_results[i+1][0]
    
    #remove FP on split line
    episod=5
    index_on_right_line = np.where(np.fabs(left_results[:,0]-x_offset_right) < episod)
    print(index_on_right_line)
    left_results[index_on_right_line, 4]-= score_desent
    
    
    right_results[:, 0] = right_results[:, 0]+x_offset_right 
    right_results[:, 2] = right_results[:, 2]+x_offset_right
    
    index_on_left_line = np.where(np.fabs(right_results[:,2]-x_offset_left) < episod)
    print(index_on_left_line)
    right_results[index_on_left_line, 4]-= score_desent
    
    merge_results = np.concatenate((left_results, right_results), axis=0)
    
    #do nms
    multiscores = merge_results[:,4].reshape(merge_results[:,4].shape[0],1)
    zeros_dim = np.zeros(multiscores.shape)

    multiscores = torch.Tensor(np.concatenate((zeros_dim, multiscores), axis=1))

    det_bboxes, det_labels = multiclass_nms(torch.Tensor(merge_results[:,:4]),multiscores ,\
                                            score_thresh, nms_cfg, 2000)

    bboxes = bbox2result(det_bboxes, det_labels, 2)

    final_results.append(bboxes)

    
with open(output_results_file, "wb") as f:
    pickle.dump(final_results, f, pickle.HIGHEST_PROTOCOL)

print("done.")    
