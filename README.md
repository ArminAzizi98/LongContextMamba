# MambaExtend
This is the official code repo for the **ICLR 2025** paper **MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba**

Paper link: https://openreview.net/pdf?id=LgzRo1RpLS

# Contrbutiors
1. Seyedarmin Azizi
2. Souvik Kundu
3. Mohammad Erfan Sadeghi
4. Massoud Pedram

# Environment Setup
<pre><code>conda env create -f env.yaml
conda activate mambaextend</code></pre>

Alternatively, you can only install the dependencies:
<pre><code>pip install -r requirements.txt</code></pre>


# Tasks
MambaExtend is evaluated across three sets of tasks: perplexity evaluation (ProofPile and PG-19), passkey retrieval, and LongBench. Please navigate to the corresponding directory for the codebase. 



## 📚 Citation

If you use this code or refer to it in your work, please cite our paper:

```bibtex
@inproceedings{azizi2025mambaextend,
  title={MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba},
  author={Azizi, Seyedarmin and Kundu, Souvik and Sadeghi, Mohammad Erfan and Pedram, Massoud},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
