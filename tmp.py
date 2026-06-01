# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def run_single_test():
    model_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/models/byte5_local_model/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- loading ---")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
    except Exception as e:
        print(f"Loading Failure{e}")
        return

    raw_tokens = "u r"
    prompts = []
    for i in range(len(raw_tokens)):
        prefix = " ".join(raw_tokens[:i])
        target = raw_tokens[i]
        suffix = " ".join(raw_tokens[i + 1:])
        prompt = f"{prefix} <extra_id_0>{target}<extra_id_1> {suffix}".strip()
        prompts.append(prompt)

    print(f"After Prompt: {prompt}")

    if prompts:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=32, num_beams=1, do_sample=False, repetition_penalty=1.0)
            # clean result sentence
            pred_tokens = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in outputs]
    else:
        pred_tokens = []

    clean_pred = pred_tokens.replace('<extra_id_0>', '').replace('<extra_id_1>', '').strip()

    print(f"Pred: {pred_tokens}")
    print(f"Clean: {clean_pred}")


if __name__ == "__main__":
    run_single_test()