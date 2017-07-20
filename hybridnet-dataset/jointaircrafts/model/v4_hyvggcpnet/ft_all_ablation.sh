#!/bin/bash

# first fine all layers

./build/tools/caffe train \
	-model "data/jointaircrafts/model/v4_hyvggcpnet/ft_all_ablation.prototxt" \
	-solver "data/jointaircrafts/model/v4_hyvggcpnet/ft_all_ablation.solver" \
	-weights "data/jointaircrafts/model/v4_hyvggcpnet/snapshot/ft_last_layer_iter_60000.caffemodel,data/jointaircrafts/model/v4_hyvggcpnet/aircrafts_cpnet_ft_all_iter_20000.caffemodel_fineaux,data/jointaircrafts/model/v4_hyvggcpnet/supaircrafts_cpnet_ft_all_iter_20000.caffemodel_coarseaux" \
	-gpu all 2>&1 | tee "data/jointaircrafts/model/v4_hyvggcpnet/jointaircrafts_hyvggcpnet_ft_all_ablation_log.txt"
