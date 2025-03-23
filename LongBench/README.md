# Run MambaExtend on LongBench
<pre> <code>python -u pred.py --model state-spaces/mamba-130m --e --task qasper --mambaextend </code> </pre>

**Note:** You can specify the desired dataset using `--task` and enable MambaExtend zeroth-order calibration with `--mambaextend`.

![Accuracy comparison on LongBnech](./assets/LongBench.png)
