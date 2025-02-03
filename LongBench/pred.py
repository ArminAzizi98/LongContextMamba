import os
from datasets import load_dataset
import shutil
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM ,LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from mamba_ssm.utils.generation import InferenceParams
import torch.distributed as dist
import torch.multiprocessing as mp
from modeling.mamba_lm import MambaLMHeadModel
from modeling.mamba_module import Mamba
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
#from mamba_ssm.modules.mamba2 import Mamba2




def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k","state-spaces/mamba-1.4b", "state-spaces/mamba-1.4b", "state-spaces/mamba2-780m", "state-spaces/mamba-130m"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--mambaextend', action='store_true', help="Use MambaExtend calibration")
    parser.add_argument('--task', type = str, default = None, choices = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"])
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def set_min(array, val1 = 0.01, val2=1.0):
    #for j in  range(array.shape[0]):
      for i,x in enumerate(array):
        if x<val1:
            array[i] = val1
        #elif x>val2:
        #    array[j,i] = val2
      return array


def set_model(loaded, vec):
    counter = 0
    for pname, p in loaded.named_modules():  
        if (isinstance(p, Mamba)):
            p.mamba_scale = torch.nn.Parameter(vec[counter], requires_grad = False)
            counter = counter + 1
    return loaded


def compute_perturb(x, t):
        import numpy
        torch.cuda.empty_cache()
        c = 0.2/((1+x)**0.1)
        beta = 0.9
        import math
        alpha = 0.005
        #* math.cos(x * math.pi/(99 * 2))
        delta = torch.tensor(numpy.random.choice([-1,+1],size=24)).cuda()
        t_p = set_min(t + c *delta)
        t_m = set_min(t - c *delta)
        return t_p, t_m, delta, c, alpha

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, mambaextend):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    import numpy
    #t= torch.tensor(numpy.array([0.0100, 0.0100, 1.7016, 1.1313, 1.4559, 1.5175, 1.8600, 0.3568, 0.4267,
    #    1.7901, 0.8959, 2.4366, 1.3380, 0.3765, 1.7525, 1.3178, 3.4849, 0.0100,
    #    1.6071, 0.3584, 1.0389, 0.1946, 0.7703, 0.0100, 1.6112, 1.0778, 0.1517,
    #    0.1435, 1.0265, 1.0852, 0.7021, 1.2663, 0.1792, 0.3393, 1.2366, 1.8142,
    #    1.6014, 1.7782, 2.7835, 1.4366, 1.4562, 0.5032, 1.3063, 2.5018, 0.4267,
    #    1.4954, 0.7166, 0.6338])).cuda()
    #t = torch.tensor(numpy.random.rand(48)/4 + 0.75).cuda()
    #t = torch.tensor(numpy.array([0.5776, 0.3189, 0.6392, 0.6927, 0.7072, 1.4100, 0.6354, 0.4411, 1.2929,
    #    1.4418, 0.4413, 0.9434, 1.1256, 0.6092, 0.7682, 1.0391, 0.5790, 0.7380,
    #    0.7558, 0.1685, 0.5206, 1.3184, 0.5636, 0.5186, 0.7503, 0.7939, 0.5169,
    #    0.7245, 0.7423, 0.7900, 1.2273, 1.2442, 0.9297, 0.6359, 0.3824, 0.3940,
    #    0.6271, 0.7331, 1.0948, 1.0054, 1.3502, 0.2990, 0.7926, 0.2176, 0.4179,
    #    1.0421, 1.0478, 1.2017])).cuda() # finalized
    #t = torch.tensor(numpy.array([0.7648, 0.3779, 1.0807, 0.5091, 0.3780, 1.7541, 0.5315, 0.2343, 1.4614,
    #    1.9082, 0.2484, 0.6208, 1.2934, 0.0248, 1.1877, 1.1492, 0.3983, 1.0729,
    #    0.8388, 0.0871, 0.7610, 0.6818, 0.5557, 0.6803, 0.5769, 0.8180, 0.8442,
    #    0.8572, 0.5958, 0.7516, 0.8611, 1.0093, 0.9234, 0.5632, 0.1779, 0.0153,
    #    1.1015, 0.4842, 0.6423, 0.6943, 1.6962, 0.0100, 0.7031, 0.0921, 0.0844,
    #    0.8501, 0.2619, 1.5276])).cuda() # finalized 780
    #t = torch.tensor(numpy.array([0.9691, 0.0100, 0.5147, 0.6906, 1.1221, 1.2855, 0.0100, 1.1731, 0.5609,
    #    1.4898, 0.0100, 0.6049, 1.1010, 0.5082, 0.7661, 0.5741, 0.8950, 0.8625,
    #    0.4173, 0.0100, 0.9355, 1.6334, 0.8561, 0.2026, 0.7524, 0.0619, 0.5649,
    #    0.6000, 1.0573, 0.4750, 1.1028, 1.2688, 0.4647, 0.7604, 0.3344, 0.0100,
    #    0.9656, 0.2681, 0.6532, 1.0075, 1.7918, 0.1803, 1.5481, 0.1803, 0.1803,
    #    1.4336, 1.5128, 1.1771])).cuda() # LCC
    #t = torch.tensor(numpy.array([1.3846, 1.8379, 0.0114, 0.0100, 0.0100, 0.4874, 0.3629, 1.2641, 1.0224,
    #    1.9399, 0.6759, 1.5571, 0.1660, 0.5062, 0.6090, 1.1965, 1.1858, 1.1046,
    #    0.8383, 0.2274, 1.1836, 0.8032, 0.9696, 0.6715, 0.7960, 1.4410, 1.1821,
    #    0.7258, 0.1824, 0.5803, 2.6876, 2.8612, 0.9936, 0.5074, 0.6428, 0.4307,
    #    0.7008, 2.3750, 1.6528, 0.6649, 1.0521, 0.7215, 1.4350, 0.4734, 0.5541,
    #    0.9644, 1.2101, 0.6937])).cuda() # 2wiki
    t = torch.tensor(numpy.random.rand(24)).cuda() #
    for x in range(100):
        #inference_params = InferenceParams(max_seqlen=None, max_batch_size=1)
        if os.path.isfile(out_path):
            os.remove(out_path)
        t_p, t_n, delta,c, alpha = compute_perturb(x, t)
        if mambaextend:
            model = set_model(model, t_p)
        counter = 0
        for json_obj in tqdm(data):
            if json_obj['length'] > 8000 or json_obj['length'] < 4000:
               continue
            counter = counter + 1
            if counter > 20 and mambaextend:
                break

            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt")
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    input['input_ids'],
                    max_length=max_gen,
                    temperature=1.0,
                    #min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:

                  output = model.generate(
                  input['input_ids'],
                  max_length=max_gen,
                  temperature=1.0,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')


        score_p = print_result(dataset, model_name ,True)
        print(t_p)
        print(f"Upper score is {score_p}")
 
        import subprocess
        #command =  "rm -r pred_e1/state-spaces/mamba-1.4b;mkdir pred_e1/state-spaces/mamba-1.4b"
        #subprocess.run(command, shell=True, check=True)

        if not mambaextend:
             return
        os.remove(out_path)

        model = set_model(model, t_n)
        counter = 0
        for json_obj in tqdm(data):
            if json_obj['length'] < 4000 or json_obj['length'] > 8000:
               continue
            counter = counter + 1
            if counter > 20:
                break
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    input['input_ids'],
                    max_length=max_gen,
                    temperature=1.0,
                    #min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],

                )[0]
            else:
                output = model.generate(
                    input['input_ids'],
                    max_length=max_gen,
                    temperature=1.0,
                 )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')

        score_n = print_result(dataset, model_name ,True)
        print(t_n)
        print(f"Lower score is {score_n}")
        
        #print(t_n)
        #command =  "rm -r pred_e1/state-spaces/mamba-1.4b;mkdir pred_e1/state-spaces/mamba-1.4b"
        #subprocess.run(command, shell=True, check=True)
        os.remove(out_path)
        g = (score_p - score_n)/(2*c * delta)
        t = set_min(t + alpha * g)
        model = set_model(model, t)
        counter = 0
        for json_obj in tqdm(data):
            if json_obj['length'] < 4000 or json_obj['length'] > 8000:
               continue
            counter = counter + 1

            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    input['input_ids'],
                    max_length=max_gen,
                    temperature=1.0,
                    #min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],

                )[0]
            else:
                output = model.generate(
                    input['input_ids'],
                    max_length=max_gen,
                    temperature=1.0,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')

        score = print_result(dataset, model_name, True)

        print(f"score at iteration {x} is {score}")
        #command =  "rm -r pred_e1/state-spaces/mamba-1.4b;mkdir pred_e1/state-spaces/mamba-1.4b"
        #subprocess.run(command, shell=True, check=True)
        if score > 10:
            print('The scaling factors are: ' + str(t))
        if score < score_p:
           score = score_p
           t = t_p

        if score < score_n:
           score = score_n
           t = t_n

        if dist.is_initialized():
            dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    else:
        model = MambaLMHeadModel.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        print(model)
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        tokenizer.pad_token = tokenizer.eos_token

    model = model.eval()
    return model, tokenizer


import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def print_result(dataset, model_path, e):
  scores = dict()
  for x in range(1):
    if e:
        path = f"pred_e{x}/{model_path}/"
    else:
        path = f"pred{x}/{model_path}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    score_4_8 = scores[dataset]["4-8k"]
    return score_4_8
  
if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [args.task]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    for iter in range(1):
      if not os.path.exists(f"pred{iter}"):
        os.makedirs(f"pred{iter}")
      if not os.path.exists(f"pred_e{iter}"):
        os.makedirs(f"pred_e{iter}")
      for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e{iter}/{model_name}"):
                os.makedirs(f"pred_e{iter}/{model_name}")
            else:
                shutil.rmtree(f"pred_e{iter}/{model_name}", ignore_errors = False)
                os.makedirs(f"pred_e{iter}/{model_name}")
            out_path = f"pred_e{iter}/{model_name}/{dataset}.jsonl"

        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args.mambaextend))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
