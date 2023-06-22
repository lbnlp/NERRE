#!/bin/bash

# Set the experiments dir
EXPERIMENTS_DIR=$1

# Set the experiment name
EXPERIMENT_NAME=$2

for i in ` ls $EXPERIMENTS_DIR ` ; do
    t="${EXPERIMENTS_DIR}/$i/train.jsonl"
    v="${EXPERIMENTS_DIR}/$i/val.jsonl"
    openai api fine_tunes.create -t $t -v $v -m "davinci" --suffix $EXPERIMENT_NAME --no_follow
done

