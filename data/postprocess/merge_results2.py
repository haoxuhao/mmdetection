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
from PIL import Image


def read_pkl(pkl_file):
    results = pickle.load(open(pkl_file, 'rb'), encoding='utf-8')
    return results

# dataset_root = "/mnt/nfs/hangkongbei/voc-style"
dataset_root = '/root/datasets/testset'

org_images_id_file = osp.join(dataset_root, "ImageSets/Main/test.txt")
org_images_dir = osp.join(dataset_root, "JPEGImages")
test_images_id_file = osp.join(dataset_root, "ImageSets/Main/test_split_new.txt") #split_dataset_new


input_results_file = "pkl_results/results_testset_resampled_finetune.pkl"
out_results_name = "results_testset_resampled_finetune_merged.pkl"

pkl_save_dir = "./pkl_results_merge"
if not osp.exists(pkl_save_dir):
    os.makedirs(pkl_save_dir)
    
output_results_file = osp.join(pkl_save_dir, out_results_name)

input_results = read_pkl(input_results_file)
with open(test_images_id_file, "r") as f:
    imageids = [file.strip() for file in f.readlines()]
    
with open(org_images_id_file, "r") as f:
    org_imageids = [file.strip() for file in f.readlines()]
    
score_thresh = 0.4
nms_cfg = dict(type="soft_nms", iou_thr=0.3)

final_results = []
wind_size = 1080
overlap = 360
stride = wind_size-overlap
episod=10
score_desent = 0.61

# org_imageids = ["00013868"]
#print(org_imageids[2])
for i in tqdm(range(len(org_imageids))):
    org_imageid = org_imageids[i]
    
    split_results = []
    org_w, org_h = Image.open(osp.join(org_images_dir, org_imageid+".jpg")).size
    
    
    for j in range(len(imageids)):
        split_imageid, offset_x, offset_y = imageids[j].split("_")
        offset_x, offset_y = int(offset_x), int(offset_y)
        #print(split_imageid, org_imageid)
        if split_imageid == org_imageid:
            split_result = input_results[j][0]
            #print(split_result.shape, offset_x, offset_y)
            #print("equal", split_imageid, org_imageid)
            #remove box on left line
            if offset_x != 0:
                index_on_left_line = np.where(np.fabs(split_result[:,0] - 0) < episod)
                split_result[index_on_left_line, 4] -= score_desent
            #remove up on up line 
            if offset_y != 0:
                index_on_up_line = np.where(np.fabs(split_result[:,1] - 0) < episod)
                split_result[index_on_up_line, 4] -= score_desent
                
            #remove up on right line 
            if offset_x + wind_size != org_w:
                index_on_right_line = np.where(np.fabs(split_result[:, 0] - wind_size) < episod)
                split_result[index_on_right_line, 4] -= score_desent
            #remove up on up line 
            if offset_y + wind_size != org_h:
                index_on_bottom_line = np.where(np.fabs(split_result[:, 1] - wind_size) < episod)
                split_result[index_on_bottom_line, 4] -= score_desent
            
            
            split_result[:, 0:4:2] += offset_x
            split_result[:, 1:4:2] += offset_y
            split_results.append(split_result)
        else:
            continue
            
    #print("merge")
    merge_results = np.concatenate(split_results, axis=0)
    #do nms
    multiscores = merge_results[:,4].reshape(merge_results[:,4].shape[0],1)
    zeros_dim = np.zeros(multiscores.shape)

    multiscores = torch.Tensor(np.concatenate((zeros_dim, multiscores), axis=1))

    det_bboxes, det_labels = multiclass_nms(torch.Tensor(merge_results[:,:4]),multiscores ,\
                                            score_thresh, nms_cfg, 2000)

    bboxes = bbox2result(det_bboxes, det_labels, 2)

    final_results.append(bboxes)
    #break
    
with open(output_results_file, "wb") as f:
    pickle.dump(final_results, f, pickle.HIGHEST_PROTOCOL)

print("done.")