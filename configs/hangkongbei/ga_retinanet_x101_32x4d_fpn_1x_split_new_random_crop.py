# model settings
model = dict(
    type='RetinaNet',
    #pretrained='modelzoo://resnet50',
    #backbone=dict(
    #    type='ResNet',
    #    depth=50,
    #    num_stages=4,
    #    out_indices=(0, 1, 2, 3),
    #    frozen_stages=1,
    #    style='pytorch'),
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='GARetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[1.0, 1.0, 1.0, 1.0],
        target_means=(.0, .0, .0, .0),
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    ga_assigner=dict(
        type='ApproxMaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.4,
        ignore_iof_thr=-1),
    ga_sampler=dict(
        type='RandomSampler',
        num=256,
        pos_fraction=0.5,
        neg_pos_ub=-1,
        add_gt_as_proposals=False),
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    center_ratio=0.2,
    ignore_ratio=0.5,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type='nms', iou_thr=0.35),
    max_per_img=250)
# dataset settings
dataset_type = 'HangkongbeiDataset'
data_root = '/mnt/nfs/hangkongbei/'
# data_root = '/root/datasets/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# data = dict(
#     imgs_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'voc-style/split_dataset/ImageSets/Main/train.txt',
#         img_prefix=data_root + 'voc-style/split_dataset/',
#         img_scale=(800, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0.5,
#         with_mask=False,
#         with_crowd=False,
#         with_label=True),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'voc-style/split_dataset/ImageSets/Main/val.txt',
#         img_prefix=data_root + 'voc-style/split_dataset/',
#         img_scale=(800, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0,
#         with_mask=False,
#         with_crowd=False,
#         with_label=True),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'voc-style/split_dataset/ImageSets/Main/val.txt',
#         img_prefix=data_root + 'voc-style/split_dataset/',
#         img_scale=(800, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0,
#         with_mask=False,
#         with_crowd=False,
#         with_label=False,
#         test_mode=True))
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',  # to avoid reloading datasets frequently
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'voc-style/split_dataset_new/ImageSets/Main/trainval.txt',
                #data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'voc-style/split_dataset_new'],
            img_scale=[(800, 800), (1080, 1080), (1280,1280)],#(1080, 1080)
            multiscale_mode="range",
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True,
            #data aug
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.8, 1.2),
                    saturation_range=(0.8, 1.2),
                    hue_delta=18),
                    expand=dict(
                        mean=img_norm_cfg['mean'],
                        to_rgb=img_norm_cfg['to_rgb'],
                        ratio_range=(1, 2)),
                    random_crop=dict(
                        min_ious=(0.0000001, 0.0000002), min_crop_size=0.8)
                ),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'voc-style/split_dataset_new/ImageSets/Main/val.txt',
        img_prefix=data_root + 'voc-style/split_dataset_new/',
        img_scale=(1080, 1080),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
       ),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'voc-style/ImageSets/Main/val.txt', #split_dataset_new
        # img_prefix = data_root+"voc-style/",#split_dataset_new
	    ann_file = data_root+"testset/ImageSets/Main/test_split_new.txt",
        img_prefix = data_root + 'testset/',
        img_scale=(1080, 1080),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False
       ))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[3, 6])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 8
device_ids = range(0)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ga_retinanet_x100_32x4d_fpn_1x_multi_scale_split_new_random_crop'
load_from = None
resume_from = None
workflow = [('train', 1)]#[('train', 1)] #
