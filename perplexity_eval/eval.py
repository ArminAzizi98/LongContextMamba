import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer
from tqdm import tqdm
from modeling.mamba_lm import MambaLMHeadModel
from modeling.mamba_module import Mamba
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
#from mamba_ssm.modules.mamba2 import Mamba2
#from mamba_ssm.modules.mamba_simple import Mamba
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import numpy
import math

def set_model(loaded, vec):
    counter = 0
    for pname, p in loaded.named_modules():  
        if (isinstance(p, Mamba)):
#            print('setting the model')
            p.armin_ratio = torch.nn.Parameter(torch.tensor([vec[counter]]).cuda(), requires_grad = False)
            counter = counter + 1
    return loaded

def set_min(array, val1 = 0.01, val2=1.0):
#    for j in  range(array.shape[0]):
      for i,x in enumerate(array):
        if x<val1:
            array[i] = val1
#        #elif x>val2:
#        #    array[j,i] = val2
      return array

def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, device=None, max_length=None, sliding_window=256, truncate=False, aggressive_memory=False, hide_progress=False, delta_ratio=None
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]
    

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=hide_progress)
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)
            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                # only get the logits for the last 1024 tokens:
                logits = model(input_ids, delta_ratio=delta_ratio).logits[..., :-1, :].contiguous()
                target_ids = target_ids[..., 1:].contiguous()
                print(input_ids.shape)
                neg_log_likelihood = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='mean')
            
            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.pad_token = tokenizer.eos_token
#    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")


    if args.tokenized:
            #input_texts = datasets.load_from_disk(args.tokenized)
            input_texts_calib = datasets.load_from_disk("./my_dataset")
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)

    else:
        input_texts = datasets.load_dataset("mmosbach/pile-validation")

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize, num_proc=4)
        input_texts_calib = input_texts_calib.map(tokenize, num_proc=4)
        if args.save_tokenized:
            from datasets import DatasetDict
            dataset = DatasetDict({"test": input_texts})
            dataset.push_to_hub(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:

        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens, num_proc=4)
        input_texts_calib = input_texts_calib.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens, num_proc=4)
    if args.samples:
        input_texts = input_texts[:args.samples]
    input_texts_calib = input_texts_calib[:args.calib_samples]
    if args.tokens_step:
        tokens = [x for x in range(
            args.min_tokens, args.max_tokens + 1, args.tokens_step)]
    else:
        tokens = [args.min_tokens]
        while args.min_tokens < args.max_tokens:
            point = tokens[-1] * 2
            if point <= args.max_tokens:
                tokens.append(point)
            else:
                break

    results = []
    vectors = []
    ppl = None
    iter = 0
    for model in tqdm(models, desc="Model", leave=False, disable=args.hide_progress):
      loaded = MambaLMHeadModel.from_pretrained(model, dtype=torch.bfloat16).to("cuda")
      print(loaded)
      loaded.eval()
#      t = numpy.array([
#      0.84518524, 0.87056195, 0.32920591, 0.16116334, 0.82215493, 0.80103385,
#      0.23010394, 0.70949769, 0.67413583, 0.78306602, 0.43663559, 0.33801703,
#      0.98078202, 0.30016948, 0.31640601, 0.95040514, 0.32520981, 0.25505764,
#      0.44414539, 0.05, 0.92785918, 0.86185007, 0.23557336, 0.99902242,
#      0.51519949, 0.18902727, 0.94836321, 0.97048861, 0.82548838, 0.99901873,
#      0.56320603, 0.15848667, 0.06337906, 0.95856838, 0.65514387, 0.85362217,
#      0.89041023, 0.8941483, 0.58430139, 0.05820116, 0.3614314, 0.50283315,
#      0.98084944, 0.22268181, 0.87325258, 0.28364548, 0.31033875, 0.79487263
#      ])
      t = numpy.random.rand(48)/4 + 0.75
      #t = numpy.array([
      #0.89984392, 0.977997,   0.8815897,  0.97730368, 1.00134224, 0.78043555,
      #1.04885112, 0.76973544, 0.91439183, 0.79723754, 0.91133445, 0.77088535,
      #0.85674372, 0.97822917, 0.93009127, 0.90448091, 0.77194169, 0.98618461,
      #0.83719668, 0.93634083, 0.79392251, 0.97520798, 0.85131216, 0.89692465,
      #0.73222719, 0.81826012, 0.99916794, 0.75185384, 0.93275284, 0.79111716,
      #0.84819277, 0.90472468, 0.80400207, 0.80411821, 0.8844608,  0.88787405,
      #0.92398218, 0.92560383, 0.77703873, 0.96746474, 0.83979701, 0.9437035,
      #0.86621843, 0.75672654, 0.78661807, 0.91611292, 0.89322591, 0.8221567])  
#      t = numpy.array([0.4917, 1.3064, 0.6692, 0.5107, 0.9111, 0.8782, 0.8039, 0.3174, 1.7046,
#        1.2679, 2.8062, 0.9354, 0.2809, 0.9637, 0.1081, 0.0773, 2.0758, 0.4019,
#        1.1145, 2.0735, 0.5589, 0.7694, 0.8399, 0.8908]) # from pg-19 70k
#      t = numpy.array([1.0663, 0.9476, 0.7780, 0.4936, 0.9110, 0.5419, 1.3521, 1.2470, 1.5686,
#        1.3175, 2.3640, 0.5392, 0.8938, 0.9619, 0.2627, 0.2432, 1.6837, 0.7308,
#        0.8478, 2.1218, 1.0165, 1.0778, 0.9045, 0.8876]) # from pg-19 16k
#      t = numpy.array([0.3444, 1.1201, 0.8093, 0.7494, 0.8681, 0.6419, 0.9452, 0.2323, 1.4122,
#        1.2875, 2.4757, 0.9242, 0.3421, 1.4046, 0.1348, 0.0600, 1.7062, 0.5206,
#        0.9418, 1.8392, 0.7080, 0.9140, 0.9318, 0.9288]) # from pg-19 64k

      for x in range(50):
        print(model)
        torch.cuda.empty_cache()
        c = 0.1/((1+x)**0.1)
        alpha = 0.001 * math.cos(x * math.pi/(99 * 2))
        delta = numpy.random.choice([-1,1], size=(48))
        t_p = set_min(t + c *delta)
        t_m = set_min(t - c *delta)
        result = []
        for max_length in tokens:
            loaded = set_model(loaded, t_p)
            ppl1 = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts_calib,
                                     add_start_token=tokenizer.bos_token is not None, max_length=max_length,
                                     sliding_window=args.sliding_window, truncate=args.truncate,
                                     aggressive_memory=args.aggressive_memory, hide_progress=args.hide_progress, delta_ratio=args.delta_ratio)['mean_perplexity']

            print('max length is:' + str(max_length))
            print("__________________________________________")

            loaded = set_model(loaded, t_m)
            ppl2 = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts_calib,
                                     add_start_token=tokenizer.bos_token is not None, max_length=max_length,
                                     sliding_window=args.sliding_window, truncate=args.truncate,
                                     aggressive_memory=args.aggressive_memory, hide_progress=args.hide_progress, delta_ratio=args.delta_ratio)['mean_perplexity']

            g = (ppl1-ppl2) * delta/(2*c)
            t = set_min(t - alpha * g)
            loaded = set_model(loaded, t)

            ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
                                     add_start_token=tokenizer.bos_token is not None, max_length=max_length,
                                     sliding_window=args.sliding_window, truncate=args.truncate,
                                     aggressive_memory=args.aggressive_memory, hide_progress=args.hide_progress, delta_ratio=args.delta_ratio)['mean_perplexity']

            if ppl > ppl1:
               t = t_p
               ppl = ppl1
            if ppl > ppl2:
               t = t_m
               ppl = ppl2

            print(f"{model}: , x = {x}, {max_length}={ppl}")
            if ppl < 6:
                 print(f" for ppl = {ppl}, scales are: " + str(t))
            result.append(ppl1)

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--calib-samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--aggressive-memory", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--delta_ratio", type=float)
    main(parser.parse_args())
