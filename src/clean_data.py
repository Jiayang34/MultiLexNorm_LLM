import re
"""
According to manual text detection, following noise should be ignored:
1. @user_name XXX
2. @user_name : XXX
3. some_operation_symbol (e.g. rt) @user_name : XXX
4. XXX @user_name
5. 4. XXX @user_name https://...
Therefore, we use @ and https as detection criteria.
"""
def is_noise(token):
    # two noise-detection patterns
    mention_pattern = r'^@\w+'
    url_pattern = r'^https?://\S+'
    return bool(re.match(mention_pattern, token) or re.match(url_pattern, token))

def is_ignorable_punctuation(token):
    # invalid characters at the head, e.g. rf = "reference to"
    return token.lower() in ['rt', ':', '..', '...', '…'] or re.match(r'^[^\w\s]+$', token)

def surgical_clean(raw_tokens, norm_tokens):
    n = len(raw_tokens)
    if n == 0:
        return [], []

    # Head Pruning (only check the first four words)
    start_idx = 0
    for i in range(min(n,4)):
        token = raw_tokens[start_idx]
        if is_noise(token) or is_ignorable_punctuation(token):
            start_idx += i+1
        if i > 3:
            break

    # Tail Pruning
    end_idx = n
    while end_idx > start_idx:
        token = raw_tokens[end_idx - 1]
        if is_noise(token) or is_ignorable_punctuation(token):
            end_idx -= 1
        else:
            break

    return raw_tokens[start_idx:end_idx], norm_tokens[start_idx:end_idx]