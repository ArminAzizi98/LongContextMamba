python -u eval.py \
    --tokenized PY007/tokenized_proof_pile_test_neox \
    --dataset-min-tokens 32768 \
    --samples 20 \
    --split test \
    --min-tokens 32768 \
    --max-tokens 32768 \
    --tokens-step 32768 \
    --truncate \
    --delta_ratio 1.0 \
    -m state-spaces/mamba-1.4b
python plot.py --xmax 12288 --ymax 20 figure
