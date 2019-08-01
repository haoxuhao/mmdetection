#!/bin/sh

mode=$1
echo "mode: "${mode}

subworkdir=ga_retinanet_x101_32x4d_fpn_1x
config="ga_retinanet_x101_32x4d_fpn_1x.py"

echo "workdir: "${subworkdir}

# python tools/test.py ./configs/hangkongbei/${config} \
#     work_dirs/${subworkdir}/latest.pth \
#      --out work_dirs/${subworkdir}/results.pkl


#evaluate
python ./tools/voc_eval.py work_dirs/${subworkdir}/results.pkl "./configs/hangkongbei/${config}"