#!/bin/bash

# first fine all layers

./build/tools/caffe test \
	-model "examples/compact_bilinear/ft_all.prototxt" \
	-solver "examples/compact_bilinear/ft_all.solver" \
	-weights "examples/compact_bilinear/snapshot/ft_all_iter_20000.caffemodel" \
	-gpu 2 \
    -iterations 1450 2>&1 | tee "examples/compact_bilinear/cub_compact_bilinear_test_test_log.txt"
