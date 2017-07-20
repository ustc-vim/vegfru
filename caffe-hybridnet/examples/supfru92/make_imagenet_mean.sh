#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=examples/supfru92/data
EXAMPLE=data/supfru92/data
DATA=data/supfru92/data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/supfru92_train_lmdb \
  $DATA/supfru92_mean.binaryproto

echo "Done."
