import os
import json
import pandas as pd

def save_results(results, model_name, is_local=False):
    if is_local:
        file_path = "/home/jiayang_wsl/llm_course/MultiLexNorm_LLM/results/"
    else:
        file_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/results/"
    os.makedirs(file_path, exist_ok=True)
    # e.g. UFAL_validation_en
    base_filename = f"{model_name}_{results[0]['dataset_type']}_{results[0]['lang']}"
    jsonl_path = os.path.join(file_path, f"{base_filename}.jsonl")
    csv_path = os.path.join(file_path, f"{base_filename}.csv")

    # save as csv
    df = pd.DataFrame(results)
    df = df[['raw', 'norm', 'pred']]
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved to CSV: {csv_path}")

    # save as JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in results:
            filtered_entry = {
                "raw": entry["raw"],
                "norm": entry["norm"],
                "pred": entry["pred"]
            }
            f.write(json.dumps(filtered_entry, ensure_ascii=False) + '\n')
    print(f"Successfully saved to JSONL: {jsonl_path}")