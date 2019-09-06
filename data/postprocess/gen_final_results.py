from tqdm import tqdm
import pickle
import os
import os.path as osp
import numpy as np
import cv2


def draw_boxes(img, boxes, labels=None, thresh=0.5):
    '''
    boxes [[x1, y1, x2, y2], ...]
    '''
    for i in range(boxes.shape[0]):
        box = boxes[i,:4]
        score = boxes[i,4]
        if score > thresh:
            cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
    return img
def read_pkl(pkl_file):
    results = pickle.load(open(pkl_file, 'rb'), encoding='utf-8')
    return results

def generate_final_results(imageid_list, result_file, output_dir="./results/final_results_esemble_thresh06", score_thresh=0.5):
    '''
    Args:
        images_list: list of image ids, 顺序与测试时的顺序一致
        result_file: mmdetection 测试时生成的结果文件
        output_dir: 保存结果文件的目录，没有会自动创建一个
    '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    category = "Vehicle"
    results = read_pkl(result_file)
    
    for i, imageid in tqdm(enumerate(imageid_list)):
        with open(osp.join(output_dir, imageid+".txt"), "w") as f:
            result = results[i][0]
            for j in range(result.shape[0]):
                if result[j,4] > score_thresh:
                    box = result[j,:4]
                    f.write("%d,%d,%d,%d,%s\n"%(box[0], box[1], box[2], box[3], category))
                    

def show_all_images(results_txt_dir, image_dir, imageids, save_dir="results/testset_images_split_leftright_retinanet101_vis"):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for imageid in tqdm(imageids):
        file_txt = open(osp.join(results_txt_dir, "%s.txt"%imageid))
        lines = file_txt.readlines()
        if len(lines) == 0:
            img_file = osp.join(image_dir, '%s.jpg'%imageid)
            img = cv2.imread(img_file)
            save_image_path = osp.join(save_dir, "%s.jpg"%imageid)
            cv2.imwrite(save_image_path, img)
            continue

        boxes = np.array([line.strip().split(",")[:4] for line in lines ],dtype=np.float32)
        scores = np.ones((boxes.shape[0],1))
        results = np.concatenate((boxes, scores), axis=1)
                        
        img_file = osp.join(image_dir, '%s.jpg'%imageid)
        img = cv2.imread(img_file)
        show_img = draw_boxes(img.copy(), results, thresh=thresh)
        
        save_image_path = osp.join(save_dir, "%s.jpg"%imageid)
        cv2.imwrite(save_image_path, show_img)
        

# dataset_root = "/mnt/nfs/hangkongbei/voc-style"
# dataset_root = '/root/datasets/testset'

inputdata_root = '/openbayes/input/input0/'
dataset_root = '/openbayes/home/mmdetection/data/preprocess/testset'

image_dir = osp.join(inputdata_root, "JPEGImages")
val_set_path = osp.join(dataset_root, "ImageSets/Main/minival.txt")

result_file = "./pkl_results_merge/minival_results_merge.pkl"
txt_file_save_dir = "./results/minival_results_merge_txt_dir"
vis_results = True
vis_results_save_dir = "./results/minival_results_merge_txt_dir_vis"
                        
#global
thresh = 0.5

with open(val_set_path, "r") as f:
    imageids = [file.strip() for file in f.readlines()]
    
print(imageids[:5])

generate_final_results(imageids, result_file, output_dir = txt_file_save_dir, score_thresh=thresh)

print("generate final results done.")

if vis_results:
    print("begin generate vis results")
    show_all_images(txt_file_save_dir, image_dir, imageids, save_dir = vis_results_save_dir)

    print("generate final vis results done.")
