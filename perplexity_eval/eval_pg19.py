from utils import *
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import json
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import pickle
import argparse
from modeling.mamba_lm import MambaLMHeadModel
from modeling.mamba_module import Mamba
from tabulate import tabulate
import math
import numpy

def set_model(loaded, vec):
    counter = 0
    for pname, p in loaded.named_modules():
        if (isinstance(p, Mamba)):
            p.mamba_scale = torch.nn.Parameter(torch.tensor([vec[counter]]).cuda(), requires_grad = False)
            counter = counter + 1
    return loaded

def set_min(array, val1 = 0.01, val2=1.0):

    for i,x in enumerate(array):
        if x<val1:
            array[i] = val1
    return array

def evaluate_validation_set_ppl_test(model, dataset_val, config, args):
    minimal_stride = 10
    max_amount_of_windows = config['ppl_test_num_windows_per_context_len_eval']
    ce_loss = CrossEntropyLoss()
    max_amount_of_windows = 1
    context_lengths = config['ppl_test_context_lens_eval']
    context_lengths = [args.min_tokens]
    n_layers = len(model.backbone.layers)
    ppl_per_context_length = []
    for i_ctx_len, window_size in enumerate(context_lengths):
        trg_len = config['ppl_test_pred_len']
        print(f'testing perplexity with context length of {window_size}, windows per sample = {max_amount_of_windows}, {trg_len} labels per window')
        t = torch.tensor(numpy.random.rand(n_layers)/4 + 0.5).cuda()

        for x in range(100):
          c = 0.1/((1+x)**0.1)
          alpha = 0.01
          delta = torch.tensor(numpy.random.normal(size=(n_layers))).cuda()
          t_p = (t + c *delta).clamp(min=0.01)
          t_m = (t - c *delta).clamp(min=0.01)
          model = set_model(model, t_p)
          nlls = []
          counter = 0

          for i, sample in enumerate(tqdm(dataset_val)):
            counter = counter + 1
            if counter ==args.calib_samples:
                break

            seq_len = sample['input_ids'].size(1)
            #if seq_len < window_size:
            #    print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')

            stride = (seq_len-window_size)//max_amount_of_windows
            if stride < minimal_stride:
                stride = minimal_stride

            for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(config['model_device'])
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs = model(input_ids, num_last_tokens=trg_len+1)
                    logits = outputs.logits
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())

                nlls.append(neg_log_likelihood)

                if end_loc == seq_len:
                    break

          ppl_p = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
          print(f'calculated up perplexity: {ppl_p:.2f}')
          ppl_per_context_length.append(ppl_p)
          model = set_model(model, t_m)
          nlls = []
          counter = 0
          for i, sample in enumerate(tqdm(dataset_val)):    
            counter = counter + 1
            if counter == args.calib_samples:
                break


            seq_len = sample['input_ids'].size(1)
            #if seq_len < window_size:
            #    print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')

            stride = (seq_len-window_size)//max_amount_of_windows
            if stride < minimal_stride:
                stride = minimal_stride

            for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(config['model_device'])
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs= model(input_ids, num_last_tokens=trg_len+1) # FIXME i added the +1 here, see if it makes sense
                    logits = outputs.logits
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())

                nlls.append(neg_log_likelihood)

                if end_loc == seq_len:
                    break

          ppl_m = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
          print(f'calculated down perplexity: {ppl_m:.2f}')
          ppl_per_context_length.append(ppl_m)

          g = (ppl_p - ppl_m)/(2*c*delta)
          t = (t - alpha * g).clamp(min=0.01)
          nlls = []
          model = set_model(model, t)
          counter = 0
          for i, sample in enumerate(tqdm(dataset_val)):    
            counter = counter + 1
            if counter == 5:
                break


            seq_len = sample['input_ids'].size(1)
            #if seq_len < window_size:
            #    print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')

            stride = (seq_len-window_size)//max_amount_of_windows
            if stride < minimal_stride:
                stride = minimal_stride

            for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(config['model_device'])
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs = model(input_ids, num_last_tokens=trg_len+1) # FIXME i added the +1 here, see if it makes sense
                    logits = outputs.logits
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())

                nlls.append(neg_log_likelihood)

                if end_loc == seq_len:
                    break

          ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
          print(f'calculated perplexity: {ppl:.2f}')
          if ppl >= ppl_p:
               ppl = ppl_p
               t = t_p
          if ppl >= ppl_m:
               ppl = ppl_m
               t = t_m
          print("The selected scaling factors are:")
          print(t)
          print("---------------------------------------------------------------")


    nlls = []
    model = set_model(model, t)
    counter = 0
    for i, sample in enumerate(tqdm(dataset_val)):    
      counter = counter + 1
      if counter == 5:
          break


      seq_len = sample['input_ids'].size(1)
            #if seq_len < window_size:
            #    print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')

      stride = (seq_len-window_size)//max_amount_of_windows
      if stride < minimal_stride:
          stride = minimal_stride

      for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(config['model_device'])
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs = model(input_ids, num_last_tokens=trg_len+1) # FIXME i added the +1 here, see if it makes sense
                    logits = outputs.logits
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())

                nlls.append(neg_log_likelihood)

                if end_loc == seq_len:
                    break




    print("******Final Evaluation******")
    ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
    print(f'calculated perplexity: {ppl:.2f}')
    print("The selected scaling factors are:")
    print(t)



    val_log = {}
    val_log['score'] = np.mean(ppl_per_context_length)
    ppl_per_context_length_str = '\t'.join(f'{x:.2f}' for x in ppl_per_context_length)
    val_log['ppl_per_ctx_len'] = {'ppl_per_context_length': ppl_per_context_length_str}
    samples_df = []
    #print(tabulate([['Perplexities:'] + [f'{x:.2f}' for x in ppl_per_context_length]], headers=['ctx len:'] + [f'{x//1000}K' for x in context_lengths] , tablefmt='pretty'))
    return samples_df, val_log
