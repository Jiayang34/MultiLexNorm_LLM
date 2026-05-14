from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import torch
from tqdm import tqdm
from src import evaluate_metrics, surgical_clean, save_results


def run_evaluation(dataset, model, tokenizer, device, lang="en", dataset_type="validation"):
    results = []
    y_true = []
    y_pred = []
    y_raw = []

    print(f"Language: {lang} | Samples amount: {len(dataset)}")
    model.eval()


    for count, item in enumerate(tqdm(dataset)):
        raw_tokens, norm_tokens = surgical_clean(item['raw'], item['norm'])

        # Prompt Engineering: mark word which needs replacement with extra_id
        """
        e.g. "u r so funy"
        -> "<extra_id_0> u <extra_id_1> r so funy"
        -> "u <extra_id_0> r <extra_id_1> so funy"
        -> "u r <extra_id_0> so <extra_id_1> funy"
        -> "u r so <extra_id_0> funy <extra_id_1>"
        """
        prompts = []
        for i in range(len(raw_tokens)):
            prefix = " ".join(raw_tokens[:i])
            target = raw_tokens[i]
            suffix = " ".join(raw_tokens[i+1:])
            prompt = f"{prefix} <extra_id_0>{target}<extra_id_1> {suffix}".strip()
            prompts.append(prompt)

        # inference: one sentence -> a batch with N-sentences (N is num of words in this sentence)
        if prompts:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=32, num_beams=1, do_sample=False,repetition_penalty=1.0)
            # clean result sentence
            pred_tokens = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in outputs]
        else:
            pred_tokens = []

        # data type: list[str]
        y_true.extend(norm_tokens)
        y_pred.extend(pred_tokens)
        y_raw.extend(raw_tokens)

        #List[Any]
        results.append({
            "lang": lang,                         # str
            "raw": " ".join(raw_tokens),          # List[str]
            "norm": " ".join(norm_tokens),        # List[str]
            "pred": " ".join(pred_tokens),        # List[str]
            "dataset_type": dataset_type          # str
        })

    # data type: dict['accuracy', 'lai', 'err_reduction']
    metrics = evaluate_metrics(y_raw, y_true, y_pred)
    print(f"Accuracy (Word Level): {metrics['accuracy']:.4f}")
    print(f"Leave-As-Is (LAI): {metrics['lai']:.4f}")
    print(f"Error Reduction Rate (ERR): {metrics['err']:.4f}")

    return results, metrics['err']


# load model from local
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path on LRZ
local_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/models/byte5_local_model/"
tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path).to(device)

print(f"Model loading successfully")

# load datasets from LRZ disk
dataset_type = "validation"
data_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/datasets/weerayut"
pub_data = load_from_disk(data_path)
val = pub_data[dataset_type]

all_languages = sorted(list(set(val['lang'])))
model_name = "UFAL"
print(f"Available Languages: {all_languages}")

print(f"--------------{model_name} Evaluation Start--------------")
for lang in all_languages:
    print(f"--------------Processing Language: {lang}--------------")
    lang_test_set = val.filter(lambda x: x["lang"] == lang)
    res, err = run_evaluation(lang_test_set, model, tokenizer, device, lang=lang)
    save_results(res, model_name)
    print(f"------------------Processing Finished------------------")

print(f"------------------Evaluation Finished-----------------")