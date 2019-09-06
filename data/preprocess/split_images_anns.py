from PIL import Image  
import os
import os.path as osp
from tqdm import tqdm
import xml.dom.minidom
from multiprocessing.dummy import Pool as ThreadPool
import threading


dataset_root = "/mnt/nfs/hangkongbei"
dataset_root = "/root/datasets/testset"
input_root = "/openbayes/input/input0"
output_dataset = "./testset"

# images_dir = "/root/datasets/testset/JPEGImages"
# imagesets_dir = "/root/datasets/testset/ImageSets/Main"
# anns_dir = "/root/datasets/testset/Annotations"

images_dir = osp.join(input_root, "JPEGImages")
imagesets_dir = osp.join(output_dataset, "ImageSets/Main")
anns_dir = osp.join(input_root, "Annotations")

# split_dataset_root = osp.join(dataset_root, "split_dataset_new")
split_dataset_root = output_dataset

split_images_dir = osp.join(split_dataset_root, "split_images_new")
split_imagesets_dir = osp.join(split_dataset_root, "ImageSets/Main")
split_anns_dir = osp.join(split_dataset_root, "Annotations")

to_process_set_name = "minival.txt"
out_set_name = "minival_split_new.txt"

if not osp.exists(split_images_dir):
    os.makedirs(split_images_dir)
if not osp.exists(split_imagesets_dir):
    os.makedirs(split_imagesets_dir)
if not osp.exists(split_anns_dir):
    os.makedirs(split_anns_dir)



to_process_set_path = osp.join(imagesets_dir, to_process_set_name)

output_set_path = osp.join(split_imagesets_dir, out_set_name)

with open(to_process_set_path, "r") as f:
    imageids = [line.strip() for line in f.readlines()]
    
print(imageids[:4])


def change_xml(src_xml, box, out_xml, size_thresh=15):
    dom = xml.dom.minidom.parse(src_xml)
    root = dom.documentElement
    
    imgid = osp.basename(out_xml).split(".")[0]
    imagename = imgid+".jpg"
    w,h = box[2]-box[0], box[3]-box[1]
    
    x1,y1 = box[0], box[1]
    x2,y2 = box[2], box[3]
    
    #set filename
    file_name = root.getElementsByTagName('filename')
    file_name[0].text = imagename
    
    #set size
    size = root.getElementsByTagName('size')[0]
    size.getElementsByTagName('width')[0].firstChild.data=str(w)
    size.getElementsByTagName('height')[0].firstChild.data=str(h)
    
    #set objects
    objects = root.getElementsByTagName('object')
    count_in = 0
    
    for object in objects:
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin_node = bndbox.getElementsByTagName('xmin')[0].childNodes[0]
        ymin_node = bndbox.getElementsByTagName('ymin')[0].childNodes[0]
        xmax_node = bndbox.getElementsByTagName('xmax')[0].childNodes[0]
        ymax_node = bndbox.getElementsByTagName('ymax')[0].childNodes[0]
        
        xmin = xmin_node.data
        ymin = ymin_node.data
        xmax = xmax_node.data
        ymax = ymax_node.data
        
        #judge if this object in this window
        if int(float(xmin)) >= x1 and int(float(xmin)) < x2 \
            and int(float(ymin)) >= y1 and int(float(ymin)) < y2:
                is_on_edge = False
                if int(float(xmax)) > x2:
                    xmax = x2
                    is_on_edge = True
                if int(float(ymax)) > y2:
                    ymax = y2
                    is_on_edge = True
                    
                #remove small box on edge
                if is_on_edge and ((int(float(xmax))-int(float(xmin))) < size_thresh or \
                                   (int(float(ymax))-int(float(ymin))) < size_thresh):
                    root.removeChild(object)
                    continue
                    
                #add offset
                xmin_node.data = str(int(float(xmin) - x1))
                ymin_node.data = str(int(float(ymin) - y1))
                xmax_node.data = str(int(float(xmax) - x1))
                ymax_node.data = str(int(float(ymax) - y1))
        else:
             root.removeChild(object)
        
    with open(out_xml, "w") as f:
        dom.writexml(f)     
    

wind_size = 1080
overlap = 360
stride = wind_size-overlap
split_xml = False
save_image = False

outfile = open(output_set_path,"w")
ext = "jpeg"
count=0

def process_single_image(imageid):
    img = Image.open(osp.join(images_dir, imageid+".jpg"))
    #print(imageid)
    xml_path = osp.join(anns_dir, imageid+".xml")
    w, h = img.size
    #print(w,h)
    for i in range(0, w-wind_size+1, stride):
        for j in range(0, h-wind_size+1, stride):
            box = [i, j, i+wind_size, j+wind_size]
            split_imageid = imageid + "_%d_%d"%(i,j)
            outfile.write(split_imageid+"\n")
            split_imagefile = osp.join(split_images_dir, split_imageid+".jpg")
            split_xmlfile = osp.join(split_anns_dir, split_imageid+".xml")
            #print(box)
            #output split image
            if save_image:
                img.crop(box).save(split_imagefile, ext)
            #change xml
            if split_xml:
                change_xml(xml_path, box, split_xmlfile)
    
    #right
    i = w-wind_size if w>wind_size else 0;
    for j in range(0, h-wind_size+1, stride):
        box = [i, j, i+wind_size, j+wind_size]
        box[2] = w if box[2] > w else box[2]
        box[3] = h if box[3] > h else box[3]
        split_imageid = imageid + "_%d_%d"%(i,j)
        outfile.write(split_imageid+"\n")
        split_imagefile = osp.join(split_images_dir, split_imageid+".jpg")
        split_xmlfile = osp.join(split_anns_dir, split_imageid+".xml")
        #print(box)
        #output split image
        if save_image:
            img.crop(box).save(split_imagefile, ext)
        #change xml
        if split_xml:
            change_xml(xml_path, box, split_xmlfile)
            
    #down
    j = h-wind_size if h>wind_size else 0;
    
    for i in range(0, w-wind_size+1, stride):
        box = [i, j, i+wind_size, j+wind_size]
        box[2] = w if box[2] > w else box[2]
        box[3] = h if box[3] > h else box[3]
        split_imageid = imageid + "_%d_%d"%(i,j)
        outfile.write(split_imageid+"\n")
        split_imagefile = osp.join(split_images_dir, split_imageid+".jpg")
        split_xmlfile = osp.join(split_anns_dir, split_imageid+".xml")
        #print(box)
        #output split image
        if save_image:
            img.crop(box).save(split_imagefile, ext)
        #change xml
        if split_xml:
            change_xml(xml_path, box, split_xmlfile)
            
    #rightdown
    i = w-wind_size if w>wind_size else 0;
    j = h-wind_size if h>wind_size else 0;
    box = [i, j, i+wind_size, j+wind_size]
    box[2] = w if box[2] > w else box[2]
    box[3] = h if box[3] > h else box[3]

    split_imageid = imageid + "_%d_%d"%(i,j)
    outfile.write(split_imageid+"\n")
    split_imagefile = osp.join(split_images_dir, split_imageid+".jpg")
    split_xmlfile = osp.join(split_anns_dir, split_imageid+".xml")
    #print(box)
    #output split image
    if save_image:
        img.crop(box).save(split_imagefile, ext)
    #change xml
    if split_xml:
        change_xml(xml_path, box, split_xmlfile)
    
    
    
    #lock.acquire()
    global count
    count+=1
    #print("%d/%d"%(count,len(imageids)), end='\n', flush=False)
    
    #lock.release()

# run processes 
# single thread is more faster
# imageids = ["00013868"]
for imageid in tqdm(imageids):
    process_single_image(imageid)

# multi thread is more slower
# threads = 8
# pool = ThreadPool(threads) 
# results = pool.map(process_single_image, imageids)
# pool.close() 
# pool.join()

outfile.close()

#show output results
with open(output_set_path,"r") as f:
    lines = f.readlines()
print(lines[:5])
