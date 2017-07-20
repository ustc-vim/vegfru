#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=examples/fru92/data
EXAMPLE=data/fru92/data
DATA=data/fru92/data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/fru92_train_lmdb \
  $DATA/fru92_mean.binaryproto

echo "Done."
