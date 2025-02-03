# Load model directly
from huggingface_hub import login
login('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')
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("state-spaces/mamba2-2.7b")
