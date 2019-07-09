from mmdet.apis import init_detector, inference_detector, show_result
import os.path as osp
import os

config_file = 'configs/mask_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'data/tmp/0214.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES,score_thr=0.5, out_file="data/tmp/%s_result.jpg"%osp.basename(img).split(".")[0])

# test a list of images and write the results to image files
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))