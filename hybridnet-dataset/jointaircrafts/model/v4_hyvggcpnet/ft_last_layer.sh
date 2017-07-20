#!/bin/bash

# first fine tune the last layer only

./build/tools/caffe train \
	-model "data/jointaircrafts/model/v4_hyvggcpnet/ft_last_layer.prototxt" \
	-solver "data/jointaircrafts/model/v4_hyvggcpnet/ft_last_layer.solver" \
    -weights "data/jointaircrafts/model/v4_hyvggcpnet/aircrafts_cpnet_ft_all_iter_20000.caffemodel_fine,data/jointaircrafts/model/v4_hyvggcpnet/supaircrafts_cpnet_ft_all_iter_20000.caffemodel_coarse" \
	-gpu all 2>&1 | tee "data/jointaircrafts/model/v4_hyvggcpnet/jointaircrafts_hyvggcpnet_ft_last_layer_log.txt"
	
