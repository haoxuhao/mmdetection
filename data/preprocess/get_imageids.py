import os
import os.path as osp

dataset_root = "/openbayes/input/input0"
image_dir = osp.join(dataset_root, "JPEGImages")
ann_dir = osp.join(dataset_root, "Annotations")

testset_root = "./testset/ImageSets"
if not osp.exists(testset_root):
    os.makedirs(testset_root)

imageids = [file.strip().split(".")[0] for file in os.listdir(image_dir)]

with open(osp.join(testset_root, "val.txt"), "w") as f:
    for imageid in imageids:
        f.write(imageid+"\n")
