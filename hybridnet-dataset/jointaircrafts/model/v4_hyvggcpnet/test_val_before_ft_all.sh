#!/bin/bash

# first fine all layers

SNAPSHOT=ft_last_layer_iter_60000.caffemodel
./build/tools/caffe test \
	-model "data/jointaircrafts/model/v4_hyvggcpnet/ft_last_layer.prototxt" \
	-solver "data/jointaircrafts/model/v4_hyvggcpnet/ft_last_layer.solver" \
	-weights "data/jointaircrafts/model/v4_hyvggcpnet/snapshot/${SNAPSHOT}" \
	-gpu 4 \
    -iterations 3333 2>&1 | tee "data/jointaircrafts/model/v4_hyvggcpnet/jointaircrafts_hyvggcpnet_test_val_before_ft_all_log.txt.${SNAPSHOT}"

# batch_size for test phase: 4 
# num of images in val set: 3333
