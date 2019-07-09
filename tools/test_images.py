from mmdet.apis import init_detector, inference_detector, show_result
import os.path as osp
import os
import json
from tqdm import tqdm

def vis_images(model, imgs, save_dir="./data/tmp", score_thr=0.25):
    # test a list of images and write the results to image files
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for img_path in tqdm(imgs):
        result = inference_detector(model, img_path)
        save_path=osp.join(save_dir, (osp.basename(img_path)).split(".")[0]+"_result.jpg")
        show_result(img_path, result, model.CLASSES, out_file=save_path)

if __name__=="__main__":
    config_file = 'configs/yaogan_det/mask_rcnn_r50_fpn_1x.py'
    checkpoint_file = './work_dirs/mask_rcnn_r50_fpn_1x/latest.pth'
    
    dataset_root = "/root/datasets/yaogan"
    image_dir = osp.join("/root/datasets/yaogan/", "val", "images")

    with open(osp.join(dataset_root, "val", "minival.json")) as f:
        images = json.load(f)["images"]
        img_paths = [osp.join(image_dir, file["file_name"]) for file in images]

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    vis_images(model, img_paths)

