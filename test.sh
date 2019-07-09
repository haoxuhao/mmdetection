#!/bin/sh 

config="configs/yaogan_det/mask_rcnn_r50_fpn_1x.py"
work_dir="./work_dirs/mask_rcnn_r50_fpn_1x"
checkpoints="${work_dir}/latest.pth"

python tools/test.py ${config} \
    ${checkpoints} \
    --out ${work_dir}"/results.pkl" --eval bbox segm