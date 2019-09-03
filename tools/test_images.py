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
        save_path=osp.join(save_dir, (osp.basename(img_path)).split(".")[0]+".jpg")
        show_result(img_path, result, model.CLASSES, score_thr=score_thr, out_file=save_path)

def get_image_paths(image_dir, imageids=None, image_suffix=".jpg"):
    if not osp.exists(image_dir):
        raise FileExistsError("no such file or directory: %s"%image_dir)
    imagefile_flags = [".jpg", ".jpeg", ".png", "PNG"]
    if imageids is None:
        image_paths = [osp.join(image_dir,file) for file in os.listdir(image_dir) if file in imagefile_flags]
        return image_paths
    else:
        image_paths = [osp.join(image_dir, id+image_suffix) for id in imageids]
        return image_paths
        
if __name__=="__main__":
    config_file = './work_dirs/ga_retinanet_x100_32x4d_fpn_1x_multi_scale_split_new_all_images/ga_retinanet_x101_32x4d_fpn_1x_split_new_random_crop.py'
    checkpoint_file = './work_dirs/ga_retinanet_x100_32x4d_fpn_1x_multi_scale_split_new_all_images/epoch_8.pth'
    
    # dataset_root = "/root/datasets/yaogan"
    # image_dir = osp.join("/root/datasets/yaogan/", "val", "images")

    # with open(osp.join(dataset_root, "val", "minival.json")) as f:
    #     images = json.load(f)["images"]
    #     img_paths = [osp.join(image_dir, file["file_name"]) for file in images]

    # data_root = '/mnt/nfs/hangkongbei/test_images'

    data_root = "/root/datasets/testset/split_images_new"
    image_dir = data_root
    #image_dir = osp.join(data_root, "voc-style/split_dataset/JPEGImages")
    
    # with open(osp.join(data_root, "voc-style/split_dataset/ImageSets/Main/val.txt")) as f:
    #     imageids = [file.strip() for file in f.readlines() if file.strip()!=""]
    org_imageids = ["00013868","00013873", "00013874", "00013875", "00013876", "00100122"]
    split_testset_path = "/root/datasets/testset/ImageSets/Main/test_split_new.txt"
    with open(split_testset_path, "r") as f:
        split_imageids = [line.strip() for line in f.readlines()]

    imageids = []
    for imageid in org_imageids:
        imageids += [line for line in split_imageids if line.split("_")[0] == imageid]

    # imageids = ["00010030_0_0",\
    #      "00010030_1440_720","00010030_720_0","00010030_0_720","00010030_2160_0","00010030_720_720","00010030_1440_0",\
    #          "00010030_2160_720"]#'00000029_left','00000069_left','10001204_left','00010835_right','00010030_left','10004210_left','00010023_right','00010763_right','00010757_right','00010763_left','10001156_left','00010884_right','10004202_left','00010853_right','00010030_right','00000053_right','00010031_left','00010803_left','00010787_left','00010846_right','00000028_left','00010835_left','00010757_left','00010852_right','00000029_right','00010784_right','00010023_left','10001503_left','00010883_left',]

    img_paths = get_image_paths(image_dir, imageids=imageids)

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    vis_images(model, img_paths, save_dir="./data/results/testset_vis", score_thr=0.6)

