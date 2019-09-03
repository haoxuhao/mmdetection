#!/bin/sh

subworkdir=$1
config=$2
weights=$3
out_file=$4


# subworkdir=ga_retinanet_x101_32x4d_fpn_1x_multi_scale
# config="ga_retinanet_x101_32x4d_fpn_1x.py"

echo "workdir: "${subworkdir}

# python tools/test.py ${config} \
#     ${weights} \
#      --out ${subworkdir}/results.pkl


#evaluate
python ./tools/voc_eval.py ${subworkdir}/results.pkl $config
