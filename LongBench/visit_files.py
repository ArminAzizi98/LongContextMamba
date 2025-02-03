import json
with open('pred_e/state-spaces/mamba-1.4b/result.json') as f:
    d = json.load(f)
    print(d)
