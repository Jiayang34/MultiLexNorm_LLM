#!/usr/bin/env python
# coding: utf-8

# # Load Model

# In[1]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# load model from local
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path on local WSL: local_path = "/home/jiayang_wsl/byte5_local_model/"
# path on LRZ
local_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/models/byte5_local_model/"
tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path).to(device)

print(f"Model loading successfully")


# # Load Dataset (MultiLexNorm2026)

# In[2]:


from datasets import load_dataset, load_from_disk
"""
load from Huggingface to WSL
my_token = "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"
pub_data = load_dataset("weerayut/multilexnorm2026-dev-pub", token=my_token)
"""
# load from LRZ disk
data_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/datasets/weerayut"
pub_data = load_from_disk(data_path)
test = pub_data["validation"]


# # Inference Implementation

# In[9]:


import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def run_evaluation(dataset, model, tokenizer, device, lang="en"):
    results = []
    y_true = []
    y_pred = []
    
    print(f"Language: {lang} | Samples amount: {len(dataset)}")
    print(f"------Inference Start!------")
    model.eval()
    
    for item in tqdm(dataset):
        # raw sentence
        raw_tokens = item['raw']
        # norm sentence
        norm_tokens = item['norm']
        
        # Prompt Engineering: mark word which need replacement with extra_id
        """
        e.g. "u r so funy"
        -> "<extra_id_124> u <extra_id_123> r so funy"
        -> "u <extra_id_124> r <extra_id_123> so funy"
        -> "u r <extra_id_124> so <extra_id_123> funy"
        -> "u r so <extra_id_124> funy <extra_id_123>"
        """
        prompts = []
        for i in range(len(raw_tokens)):
            prefix = " ".join(raw_tokens[:i])
            target = raw_tokens[i]
            suffix = " ".join(raw_tokens[i+1:])
            prompt = f"{prefix} <extra_id_124> {target} <extra_id_123> {suffix}".strip()
            prompts.append(prompt)
        
        # inference: one sentence -> a batch with N-sentences(N is num of words in this sentence)
        if prompts:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=32, 
                    num_beams=5, 
                    early_stopping=True
                )    
            # clean result sentence
            pred_tokens = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in outputs]
        else:
            pred_tokens = []

        y_true.extend(norm_tokens)
        y_pred.extend(pred_tokens)
        
        results.append({
            "raw": " ".join(raw_tokens),
            "norm": " ".join(norm_tokens),
            "pred": " ".join(pred_tokens)
        })
    
    print(f"------Evaluation Start------")
    acc = accuracy_score(y_true, y_pred)
    err = 1 - acc
    print(f"Accuracy (Word Level): {acc:.4f}")
    print(f"Error Rate (ERR): {err:.4f}")
    
    return results, err


# # Inference with ALL LANGUAGES

# In[ ]:


all_languages = sorted(list(set(test['lang'])))
print(f"Available Languages: {all_languages}")

print(f"--------------Evaluation Start--------------")
for lang in all_languages:
    print(f"--------------Processing Language: {lang}--------------")
    lang_test_set = test.filter(lambda x: x["lang"] == lang)
    res, err = run_evaluation(lang_test_set, model, tokenizer, device, lang=lang)

print(f"--------------Evaluation Finished--------------")


# In[ ]:




