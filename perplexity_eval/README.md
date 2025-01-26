# Run MambaExtend on Pile
<pre> <code>python -u eval.py \
    --tokenized PY007/tokenized_proof_pile_test_neox \
    --eval-length 16384 \
    --samples 100 \
    --calib-samples 20 \
    --split test \
    --output-file figure \
    --truncate \
    --delta_ratio 1.0 \
    -m state-spaces/mamba-130m </code> </pre>

**Note:** You can set the desired evaluation context length and the number of evaluation samples directly  by modifying the --eval-length and --samples parameters, respectively.
