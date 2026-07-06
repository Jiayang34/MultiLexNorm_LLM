import argparse
import bz2
import html
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from tqdm import tqdm


DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_SENTENCES = 2100
DEFAULT_OUTPUT = Path("data/wiki/enwiki_tokens_2100.jsonl")
DEFAULT_INPUT_URLS = {
    "en": (
        "https://dumps.wikimedia.org/enwiki/latest/"
        "enwiki-latest-pages-articles.xml.bz2"
    ),
    "de": (
        "https://dumps.wikimedia.org/dewiki/latest/"
        "dewiki-latest-pages-articles.xml.bz2"
    ),
    "it": (
        "https://dumps.wikimedia.org/itwiki/latest/"
        "itwiki-latest-pages-articles.xml.bz2"
    ),
    "ko": (
        "https://dumps.wikimedia.org/kowiki/latest/"
        "kowiki-latest-pages-articles.xml.bz2"
    ),
    "nl": (
        "https://dumps.wikimedia.org/nlwiki/latest/"
        "nlwiki-latest-pages-articles.xml.bz2"
    ),
    "ja": (
        "https://dumps.wikimedia.org/jawiki/latest/"
        "jawiki-latest-pages-articles.xml.bz2"
    ),
}

MIN_LINE_CHARS = 32
MIN_SENTENCE_CHARS = 32
MAX_SENTENCE_CHARS = 160


# Tokenization list for tweeter like text
FALLBACK_TOKEN_PATTERN = re.compile(
    r"""
    https?://\S+|www\.\S+
    |[@#][^\W_]+
    |[:;=8xX][\-o\*']?[\)\]\(\[dDpP/:\}\{@\|\\]
    |[^\W\d_]+(?:['’][^\W\d_]+)?
    |\d+(?:[.,]\d+)*
    |\.{2,}
    |[!?.,;:()\[\]{}"“”‘’'`-]
    """,
    re.VERBOSE,
)


# Strip the XML namespace
def tag_name(tag):
    return tag.rsplit("}", 1)[-1]


# Open Wiki dump and return stream
def open_dump(url):
    request = Request(url, headers={"User-Agent": "MultiLexNorm_LLM/0.1"})
    response = urlopen(request)
    return response, bz2.BZ2File(response)


# Extract text from wiki dump stream
def iter_articles(stream):
    for event, elem in ElementTree.iterparse(stream, events=("end",)):
        if tag_name(elem.tag) != "page":
            continue

        namespace = ""
        redirect = False
        text = ""

        for child in elem:
            name = tag_name(child.tag)
            if name == "ns":
                namespace = child.text or ""
            elif name == "redirect":
                redirect = True
            elif name == "revision":
                for revision_child in child:
                    if tag_name(revision_child.tag) == "text":
                        text = revision_child.text or ""

        if namespace == "0" and text and not redirect:
            yield text

        # Important for real dumps: release parsed XML nodes immediately.
        elem.clear()


# Remove simple nested {{...}} wiki templates.
def remove_templates(text):
    old_text = None
    while old_text != text:
        old_text = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    return text


# Lightly convert raw wikitext into plain text lines
def clean_wiki(text):
    text = html.unescape(text)
    text = re.sub(r"(?is)<ref\b[^>/]*?>.*?</ref>", " ", text)
    text = re.sub(r"(?is)<ref\b[^>]*/>", " ", text)
    text = re.sub(r"(?is)<gallery\b.*?</gallery>", " ", text)
    text = remove_templates(text)

    # Keep the displayed text from wiki links: [[target|label]] -> label.
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Keep labels from external links and drop raw URLs.
    text = re.sub(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://[^\]]+\]", " ", text)
    text = re.sub(r"(?:https?|ftp)://\S+", " ", text)

    # Drop remaining HTML/table/list markup, but preserve newlines so line
    # filtering can still remove title-like fragments.
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"(?m)^\s*[|!].*$", " ", text)
    text = re.sub(r"(?m)^\s*[#*:;]+\s*", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text


# Normalize text then Tokenization
def tokenize_raw_tweet_text(text):
    return FALLBACK_TOKEN_PATTERN.findall(html.unescape(text.replace("&amp;", "&")))


# Build stanza pipline for sentence splitting
def build_stanza(language, model_dir):
    import stanza

    processors = "tokenize"
    if model_dir:
        stanza.download(language, processors=processors, model_dir=model_dir)
        return stanza.Pipeline(
            lang=language,
            processors=processors,
            model_dir=model_dir,
            tokenize_no_ssplit=False,
        )

    stanza.download(language, processors=processors)
    return stanza.Pipeline(
        lang=language,
        processors=processors,
        tokenize_no_ssplit=False,
    )


# Sentence filtering
def process_line(line, nlp, tokenize, seen, lower):
    line = line.strip()

    # Exclude short lines or end with :
    if len(line) <= MIN_LINE_CHARS or line.endswith(":"):
        return []

    # Stanza sentence splitting
    try:
        sentences = nlp(line).sentences
    except Exception:
        return []

    # Exclude short/long/repeated sentences, tokenization, lower lettering
    out = []
    for sentence in sentences:
        text = sentence.text.strip()
        if len(text) <= MIN_SENTENCE_CHARS or len(text) > MAX_SENTENCE_CHARS:
            continue
        if text in seen:
            continue
        seen.add(text)

        tokens = tokenize(text)
        if lower:
            tokens = [token.lower() for token in tokens]
        out.append(tokens)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream Wikipedia XML, clean text, sentence-split, tokenize, and write JSONL."
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--input-url", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES)
    parser.add_argument("--stanza-model-dir", default=None)
    parser.add_argument("--keep-case", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.input_url or DEFAULT_INPUT_URLS.get(args.language)
    if not url:
        raise ValueError(f"No default dump URL for language {args.language!r}")

    nlp = build_stanza(args.language, args.stanza_model_dir)
    seen = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    response, stream = open_dump(url)
    written = 0

    try:
        with args.output.open("w", encoding="utf-8") as writer:
            progress = tqdm(total=args.max_sentences, desc="Importing wiki")
            for article in iter_articles(stream):
                for line in clean_wiki(article).splitlines():
                    for tokens in process_line(
                        line,
                        nlp,
                        tokenize_raw_tweet_text,
                        seen,
                        lower=not args.keep_case,
                    ):
                        writer.write(json.dumps(tokens, ensure_ascii=False) + "\n")
                        written += 1
                        progress.update(1)
                        if written >= args.max_sentences:
                            progress.close()
                            print(f"Wrote {written} sentences to {args.output}")
                            return
            progress.close()
    finally:
        stream.close()
        response.close()

    print(f"Wrote {written} sentences to {args.output}")


if __name__ == "__main__":
    main()
