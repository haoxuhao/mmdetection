# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 18:19:09 2018
@author: Administrator
"""
 
import os
from PIL import Image  
import os.path  
import xml.dom.minidom
# import cv2
def changeXML(parent, filepath, location,dirRootNew):
    xmlPath = os.path.join(parent,filepath)
    #eg:F:/ObjectDetection/mstardata/test_AAA2/test_AAA2_xml/000001.xml
    #print(xmlPath)
    xmlName = os.path.splitext(filepath)
    #eg；000001
    print(xmlName[0])
    dom = xml.dom.minidom.parse(xmlPath)
    root = dom.documentElement

    #获取图片名
    file_name = root.getElementsByTagName('filename')
    p0 = file_name[0]
    print(p0.firstChild.data)
    #获取图片尺寸信息
    size = root.getElementsByTagName('size')
    size0 = size[0]
    width0 = size0.getElementsByTagName('width')
    height0 = size0.getElementsByTagName('height')
    width = int(width0[0].firstChild.data)
    height = int(height0[0].firstChild.data)
    print(width)
    print(height)
    #获取标注信息
    objects = root.getElementsByTagName('object')

    for object in objects:
        print("*****Object*****")
        bndbox = object.getElementsByTagName('bndbox')[0]
        Nodexmin = bndbox.getElementsByTagName('xmin')[0]
        Nodeymin = bndbox.getElementsByTagName('ymin')[0]
        Nodexmax = bndbox.getElementsByTagName('xmax')[0]
        Nodeymax = bndbox.getElementsByTagName('ymax')[0]
        xmin = int(Nodexmin.childNodes[0].data)
        ymin = int(Nodeymin.childNodes[0].data)
        xmax = int(Nodexmax.childNodes[0].data)
        ymax = int(Nodeymax.childNodes[0].data)
        print ("xmin: %s" % xmin)
        print ("ymin: %s" % ymin)
        print ("xmax: %s" % xmax)
        print ("ymax: %s" % ymax)
    # 标签处理
        if(width <= 2 * height):
          print("正常处理")
          if(location == 'left'):
            print("Left")
            if(xmax > height):
              print("删掉这个object")
              root.removeChild(object)
          else:#right
            print("Right")
            if(xmin < width - height):
              print("删掉这个object")
              root.removeChild(object)
            else:#需要将所有坐标向左移动
              Nodexmin.childNodes[0].data = xmin - (width - height)
              Nodexmax.childNodes[0].data = xmax - (width - height)
        else:
          print("特殊处理")
          if(location == 'left'):
            print("Left")
            if(xmax > width/2):
              print("删掉这个object")
              root.removeChild(object)
          else:#right
            print("Right")
            if(xmin < width/2):
              print("删掉这个object")
              root.removeChild(object)
            else:#需要将所有坐标向左移动
              Nodexmin.childNodes[0].data = xmin - width/2
              Nodexmax.childNodes[0].data = xmax - width/2

    #保存修改到xml文件中

    with open(os.path.join(dirRootNew,xmlName[0]+'_'+location+'.xml'),'w') as fh:  
        dom.writexml(fh)  
        print('写入xml OK!')  

# 切割图片
def splitimage(src, dstpath):
    parent=r'./Annotations/'
    #filepath=r'10000116.xml'
    #location='left'
    dirRootNew = r'./SplitAnnotations/'
    #changeXML(parent, filepath, location,dirRootNew)

    img = Image.open(src)
    w, h = img.size
    if w <= 2 * h :
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('图片切割为两个正方形')
 
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]
        if ext == 'jpg':
            ext = 'jpeg'
#Left
        box = (0, 0, h, h)#设置左、上、右、下的像素
        img.crop(box).save(os.path.join(dstpath, basename + '_' + 'left' + '.' + "jpg"), ext)
        changeXML(parent, basename+'.xml' , 'left' ,dirRootNew)

#Right
        box = (w-h, 0, w, h)#设置左、上、右、下的像素
        img.crop(box).save(os.path.join(dstpath, basename + '_' + 'right' + '.' +  "jpg"), ext)
        changeXML(parent, basename+'.xml' , 'right' ,dirRootNew)

    if w > 2 * h :
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('w > 2 * h 图片切割为两部分')
 
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]
        if ext == 'jpg':
            ext = 'jpeg'
#Left
        box = (0, 0, w/2, h)#设置左、上、右、下的像素
        img.crop(box).save(os.path.join(dstpath, basename + '_' + 'left' + '.' +  "jpg"), ext)
        changeXML(parent, basename+'.xml' , 'left' ,dirRootNew)
#Right
        box = (w/2, 0, w, h)#设置左、上、右、下的像素
        img.crop(box).save(os.path.join(dstpath, basename + '_' + 'right' + '.' +  "jpg"), ext)
        changeXML(parent, basename+'.xml' , 'right' ,dirRootNew)

 
# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
 
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print (path+' 创建成功')
        return True
    else:
        print (path + ' 目录已存在')
        return False
 
 
folder = r'./JPEGImages' # 存放图片的文件夹
mkdir(r'./SplitAnnotations/') #存放标签结果
path = os.listdir(folder)
# print(path)
 
for each_bmp in path: # 批量操作
        first_name, second_name = os.path.splitext(each_bmp)
        each_bmp = os.path.join(folder, each_bmp)
        src = each_bmp
        print(src)
        print(first_name)
        # 定义要创建的目录
        #mkpath = r'./SplitJPEGImages/'+ first_name
        mkpath = r'./SplitJPEGImages/'

        # 调用函数
        mkdir(mkpath)

        
        if os.path.isfile(src):
            dstpath = mkpath
            if (dstpath == '') or os.path.exists(dstpath):
                splitimage(src, dstpath)
            else:
                print('图片保存目录 %s 不存在！' % dstpath)
        else:
            print('图片文件 %s 不存在！' % src)
