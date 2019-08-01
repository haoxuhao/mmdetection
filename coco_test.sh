#!/bin/sh 

config="configs/hangkongbei/ga_retinanet_x101_32x4d_fpn_1x.py"
work_dir="./work_dirs/ga_retinanet_x101_32x4d_fpn_1x"
checkpoints="${work_dir}/latest.pth"

python tools/test.py ${config} \
    ${checkpoints} \
    --out ${work_dir}"/results.pkl" --eval bbox