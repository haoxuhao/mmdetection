from PIL import Image  
import os
import os.path as osp
from tqdm import tqdm

image_dir = "/root/datasets/testset/JPEGImages"
val_set_path = "/root/datasets/testset/ImageSets/Main/test.txt"

test_images_id_file = "/root/datasets/testset/ImageSets/Main/test_split.txt"
test_image_save_dir = "/root/datasets/testset/split_images"

if not osp.exists(test_image_save_dir):
    os.makedirs(test_image_save_dir)
    
with open(val_set_path, "r") as f:
    image_paths = [osp.join(image_dir, file.strip()+".jpg") for file in f.readlines()]
    
ext="jpeg"
margin = 160

outfile = open(test_images_id_file, "w")

for img_path in tqdm(image_paths):
    img = Image.open(img_path)
    imageid = osp.basename(img_path).split(".")[0]
    w,h = img.size
    if w <= 2 * h :
        #print('图片切割为两个正方形')
        box = (0, 0, h+margin, h)#设置左、上、右、下的像素
        left_image_path = osp.join(test_image_save_dir, imageid+"_left_%d.jpg"%(box[2]))
        img.crop(box).save(left_image_path, ext)
        box = (w-h-margin, 0, w, h)#设置左、上、右、下的像素
        right_image_path = osp.join(test_image_save_dir, imageid+"_right_%d.jpg"%(box[0]))
        img.crop(box).save(right_image_path, ext)
        outfile.write(osp.basename(left_image_path).split(".")[0]+"\n")
        outfile.write(osp.basename(right_image_path).split(".")[0]+"\n")
        
    elif w > 2 * h :
        #print('w > 2 * h 图片切割为两部分')
        box = (0, 0, w//2+margin, h)#设置左、上、右、下的像素
        left_image_path = osp.join(test_image_save_dir, imageid+"_left_%d.jpg"%(box[2]))
        
        img.crop(box).save(left_image_path, ext)
        box = (w//2-margin, 0, w, h)#设置左、上、右、下的像素
        right_image_path = osp.join(test_image_save_dir, imageid+"_right_%d.jpg"%(box[0]))
        img.crop(box).save(right_image_path, ext)
        
        outfile.write(osp.basename(left_image_path).split(".")[0]+"\n")
        outfile.write(osp.basename(right_image_path).split(".")[0]+"\n")
        
outfile.close()
        