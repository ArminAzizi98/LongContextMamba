# Run MambaExtend on Passkey Retrieval
<pre> <code>python -u finetune_ssm.py </code> </pre>

**Note**: This code is based on the DeciMamba implementation for the passkey retrieval task. To run MambaExtended, set the mambaextend argument to true in the ./configs/finetune_ssm_config.json file. To use the original Mamba, set it to false. You can also configure other hyperparameters, attributes, and the model name in this file. 

![Perplexity comparison on Passkey Retrieval.](./assets/Passkey.png)
