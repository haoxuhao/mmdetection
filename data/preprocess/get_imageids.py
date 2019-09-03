import os
import os.path as osp

image_dir = "./test"
test_set_root = "./testset"


imageids = [file.strip().split(".")[0] for file in os.listdir(image_dir)]

with open("test.txt", "w") as f:
    for imageid in imageids:
        f.write(imageid+"\n")
