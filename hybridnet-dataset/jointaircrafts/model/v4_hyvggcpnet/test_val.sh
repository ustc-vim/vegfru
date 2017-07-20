#!/bin/bash

# first fine all layers

SNAPSHOT=ft_all_iter_30000.caffemodel
./build/tools/caffe test \
	-model "data/jointaircrafts/model/v4_hyvggcpnet/ft_all.prototxt" \
	-solver "data/jointaircrafts/model/v4_hyvggcpnet/ft_all.solver" \
	-weights "data/jointaircrafts/model/v4_hyvggcpnet/snapshot/${SNAPSHOT}" \
	-gpu 1 \
    -iterations 3333 2>&1 | tee "data/jointaircrafts/model/v4_hyvggcpnet/jointaircrafts_hyvggcpnet_test_val_log.txt.${SNAPSHOT}"

# batch_size for test phase: 4 
# num of images in val set: 3333
