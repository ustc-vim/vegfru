#!/bin/bash

./build/tools/caffe test \
	-model "examples/compact_bilinear/ft_last_layer.prototxt" \
	-solver "examples/compact_bilinear/ft_last_layer.solver" \
	-weights "examples/compact_bilinear/snapshot/ft_last_layer_iter_60000.caffemodel" \
	-gpu 2 \
    -iterations 1450 2>&1 | tee "examples/compact_bilinear/cub_compact_bilinear_test_test_before_ft_all_log.txt"
