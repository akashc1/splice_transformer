#!/usr/bin/env bash

CONTEXT_LENGTHS=(80 400 2k 10k)
echo "running training!"
for cl in ${CONTEXT_LENGTHS[@]}; do
    for i in {1..5};
    do
        echo "Running training for context length ${cl}, iteration ${i}!"
        python train.py --workdir checkpoints/baseline_context${cl}_v${i} --config configs/base_${cl}.py
    done
done
