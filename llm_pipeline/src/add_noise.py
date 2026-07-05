import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from src.config import DATASET_NAME, DEV_RATIO, LANGUAGE, SEED, SPLIT_NAME


DEFAULT_INPUT = Path("data/wiki/enwiki_tokens_2100.jsonl")
DEFAULT_OUTPUT = Path("data/wiki/enwiki_noise_2100.jsonl")
PHRASE_LENGTHS = (3, 2)


# Load train split records
def load_train_records(language):
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == language)

    records = []
    for row in dataset:
        # Token-level noise statistics require aligned raw/norm tokens.
        if len(row["raw"]) != len(row["norm"]):
            continue
        records.append(
            {
                "raw": [str(token) for token in row["raw"]],
                "norm": [str(token) for token in row["norm"]],
            }
        )

    random.Random(SEED).shuffle(records)
    dev_size = int(len(records) * DEV_RATIO)
    return records[dev_size:]


# Count noise probability and variant candidates probability
def build_noise_tables(records):
    total_counts = Counter()
    variant_counts = defaultdict(Counter)

    for record in records:
        for raw_token, norm_token in zip(record["raw"], record["norm"]):
            total_counts[norm_token] += 1
            if raw_token != norm_token:
                variant_counts[norm_token][raw_token] += 1

    # Keep only norm tokens that were observed with at least one noisy raw form.
    variant_counts = {
        norm_token: counts
        for norm_token, counts in variant_counts.items()
        if counts
    }
    return total_counts, variant_counts


# Sample one item proportionally to its observed count.
def weighted_sample(counter, rng):
    total = sum(counter.values())
    threshold = rng.uniform(0, total)
    cumulative = 0
    for item, count in counter.items():
        cumulative += count
        if cumulative >= threshold:
            return item
    return next(reversed(counter))


# Add noise by probablity
def maybe_add_noise(token, total_counts, variant_counts, noise_scale, rng):
    if token not in variant_counts:
        return token

    noisy_count = sum(variant_counts[token].values())
    noise_probability = noisy_count / total_counts[token]
    noise_probability = min(1.0, noise_probability * noise_scale)

    if rng.random() >= noise_probability:
        return token
    return weighted_sample(variant_counts[token], rng)



# Build a phrase from adjacent clean norm tokens.
def make_phrase(tokens, start, length):
    return " ".join(tokens[start:start + length])


# Try replacing a 2-gram or 3-gram clean phrase with one noisy raw token
# trying to -> tryna
def maybe_add_phrase_noise(tokens, start, total_counts, variant_counts, noise_scale, rng):
    for length in PHRASE_LENGTHS:
        if start + length > len(tokens):
            continue

        norm_phrase = make_phrase(tokens, start, length)
        raw_phrase = maybe_add_noise(
            norm_phrase,
            total_counts,
            variant_counts,
            noise_scale,
            rng,
        )
        if raw_phrase != norm_phrase:
            return raw_phrase, norm_phrase, length

    return None, None, 0


# Remove apostrophes
# don't -> dont
def maybe_remove_apostrophe(token, probability, rng):
    if "'" not in token and "’" not in token:
        return token
    if rng.random() >= probability:
        return token
    noisy_token = token.replace("'", "").replace("’", "")
    if not noisy_token or not any(char.isalnum() for char in noisy_token):
        return token
    return noisy_token


# Missing typos
# thought -> thoght
def maybe_delete_char(token, probability, rng):
    if len(token) <= 3 or not token.isalpha():
        return token
    if rng.random() >= probability:
        return token

    candidates = list(range(1, len(token) - 1))
    index = rng.choice(candidates)
    return token[:index] + token[index + 1:]


# Repeat typos
# with -> witth
def maybe_repeat_char(token, probability, rng):
    if len(token) <= 2 or not token.isalpha():
        return token
    if rng.random() >= probability:
        return token

    index = rng.randrange(len(token))
    return token[:index] + token[index] + token[index:]


# Swap neighbour letters typos
# opponents -> opponenst
def maybe_swap_chars(token, probability, rng):
    if len(token) <= 3 or not token.isalpha():
        return token
    if rng.random() >= probability:
        return token

    index = rng.randrange(1, len(token) - 1)
    chars = list(token)
    chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


# Lowercase
# London -> london
def maybe_change_case(token, probability, rng):
    if not token or token.lower() == token:
        return token
    if rng.random() >= probability:
        return token
    return token.lower()


# Apply simple token-internal synthetic noise rules.
def maybe_add_rule_noise(token, args, rng):
    noisy_token = maybe_remove_apostrophe(token, args.apostrophe_prob, rng)
    noisy_token = maybe_delete_char(noisy_token, args.delete_char_prob, rng)
    noisy_token = maybe_repeat_char(noisy_token, args.repeat_char_prob, rng)
    noisy_token = maybe_swap_chars(noisy_token, args.swap_char_prob, rng)
    noisy_token = maybe_change_case(noisy_token, args.case_prob, rng)
    return noisy_token


# Read clean wiki token JSONL, one token list per line.
def load_token_lists(path):
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                yield json.loads(line)


# Apply train-frequency phrase noise 
# Apply train-frequency token noise 
# Apply rule noise
def write_noisy_records(
    input_path,
    output_path,
    language,
    total_counts,
    variant_counts,
    args,
    seed,
):
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    replacements = Counter()

    with output_path.open("w", encoding="utf-8") as writer:
        for norm_tokens in load_token_lists(input_path):
            raw_tokens = []
            output_norm_tokens = []
            stats["sentences"] += 1

            index = 0
            while index < len(norm_tokens):
                # Apply train-frequency phrase noise 
                raw_phrase, norm_phrase, phrase_length = maybe_add_phrase_noise(
                    norm_tokens,
                    index,
                    total_counts,
                    variant_counts,
                    args.noise_scale,
                    rng,
                )

                if phrase_length:
                    raw_tokens.append(raw_phrase)
                    output_norm_tokens.append(norm_phrase)
                    stats["tokens"] += 1
                    stats["covered_tokens"] += 1
                    stats["changed_tokens"] += 1
                    stats["phrase_changed_spans"] += 1
                    stats["phrase_changed_input_tokens"] += phrase_length
                    replacements[(norm_phrase, raw_phrase)] += 1
                    index += phrase_length
                    continue

                # Apply train-frequency token noise 
                norm_token = norm_tokens[index]
                raw_token = maybe_add_noise(
                    norm_token,
                    total_counts,
                    variant_counts,
                    args.noise_scale,
                    rng,
                )
                # Apply rule noise
                if raw_token == norm_token:
                    raw_token = maybe_add_rule_noise(norm_token, args, rng)
                raw_tokens.append(raw_token)
                output_norm_tokens.append(norm_token)
                stats["tokens"] += 1

                if norm_token in total_counts:
                    stats["covered_tokens"] += 1
                if raw_token != norm_token:
                    stats["changed_tokens"] += 1
                    if norm_token not in variant_counts:
                        stats["rule_changed_tokens"] += 1
                    replacements[(norm_token, raw_token)] += 1
                index += 1

            writer.write(
                json.dumps(
                    {
                        "raw": raw_tokens,
                        "norm": output_norm_tokens,
                        "lang": language,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return stats, replacements


# Print statistic
'''
Train noise rate
Wiki coverage
Changed wiki tokens
Phrase changed spans
Rule changed tokens
Top replacements
'''
def print_stats(train_records, total_counts, variant_counts, stats, replacements):
    token_count = stats["tokens"]
    changed_count = stats["changed_tokens"]
    covered_count = stats["covered_tokens"]
    train_token_count = sum(total_counts.values())
    train_noisy_count = sum(sum(counts.values()) for counts in variant_counts.values())

    print(f"Train records: {len(train_records)}")
    print(f"Train tokens: {train_token_count}")
    print(f"Train noisy tokens: {train_noisy_count}")
    print(f"Train noise rate: {train_noisy_count / train_token_count:.4f}")
    print(f"Norm types: {len(total_counts)}")
    print(f"Noisy norm types: {len(variant_counts)}")
    print(f"Wiki sentences: {stats['sentences']}")
    print(f"Wiki tokens: {token_count}")
    print(f"Covered wiki tokens: {covered_count} ({covered_count / token_count:.4f})")
    print(f"Changed wiki tokens: {changed_count} ({changed_count / token_count:.4f})")
    print(f"Phrase changed spans: {stats['phrase_changed_spans']}")
    print(f"Phrase changed input tokens: {stats['phrase_changed_input_tokens']}")
    print(f"Rule changed tokens: {stats['rule_changed_tokens']}")

    if replacements:
        print("Top replacements:")
        for (norm_token, raw_token), count in replacements.most_common(20):
            print(f"  {norm_token} -> {raw_token}: {count}")


# Parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Add synthetic noise to clean wiki token JSONL."
    )
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--apostrophe-prob", type=float, default=0.468)
    parser.add_argument("--delete-char-prob", type=float, default=0.003)
    parser.add_argument("--repeat-char-prob", type=float, default=0.003)
    parser.add_argument("--swap-char-prob", type=float, default=0.003)
    parser.add_argument("--case-prob", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


# Run synthetic noise generation.
def main():
    args = parse_args()

    train_records = load_train_records(args.language)
    total_counts, variant_counts = build_noise_tables(train_records)
    stats, replacements = write_noisy_records(
        args.input,
        args.output,
        args.language,
        total_counts,
        variant_counts,
        args,
        args.seed,
    )

    print_stats(train_records, total_counts, variant_counts, stats, replacements)
    print(f"Wrote noisy wiki records to {args.output}")


if __name__ == "__main__":
    main()
