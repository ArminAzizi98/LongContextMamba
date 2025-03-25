# MambaExtend
This is the official code repo for the **ICLR 2025** paper **MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba**

Paper link: https://openreview.net/pdf?id=LgzRo1RpLS

# Contrbutiors
1. Seyedarmin Azizi
2. Souvik Kundu
3. Mohammad Erfan Sadeghi
4. Massoud Pedram

# Environment Setup
<pre><code>conda env create -f env.yml
conda activate mambaextend</code></pre>

Alternatively, you can only install the dependencies:
<pre><code>pip install -r requirements.txt</code></pre>


# Tasks
MambaExtend is evaluated across three sets of tasks: perplexity evaluation (ProofPile and PG-19), passkey retrieval, and LongBench.



## 📚 Citation

If you use this code or refer to it in your work, please cite our paper:

```bibtex
@inproceedings{azizi2025mambaextend,
  title     = {MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba},
  author    = {Seyedarmin Azizi, Souvik Kundu, Mohammad Erfan Sadeghi, Massoud Pedram},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=LgzRo1RpLS}
}
