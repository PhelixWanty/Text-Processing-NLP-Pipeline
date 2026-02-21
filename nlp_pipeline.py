"""
nlp_pipeline.py
Implements an NLP mini-pipeline (Steps 1–5):
1) Sentence segmentation
2) Tokenization
3) POS classification (heuristic baseline OR manual labeling with caching)
4) Lemmatization (dictionary + tiny suffix rules; optional external dataset)
5) Stop-word removal

Run:
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv --pos manual

Optional external lemma dataset (English):
  Download lemmatization-en.txt and pass:
  python nlp_pipeline.py --input ... --output ... --lemma-dataset lemmatization-en.txt

Manual POS mode caches labels so you don’t need to relabel every run:
  --pos manual --manual-store manual_pos.json
"""

import re, json, argparse
from pathlib import Path

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")

DEFAULT_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","they","their",
    "we","our","you","your","i","me","my","he","she","him","her","them","from","can","could","may","might",
    "will","would","should","do","does","did","not","so","many","often","each","after","before","still"
}

FALLBACK_LEMMA_DICT = {
    "helps":"help","computers":"computer","apps":"app","results":"result","students":"student",
    "sentences":"sentence","words":"word","parts":"part","processing":"process","systems":"system",
    "documents":"document","questions":"question","starts":"start","assigns":"assign","reduces":"reduce",
    "affects":"affect","includes":"include","meanings":"meaning","running":"run"
}

VALID_POS = ["NOUN","VERB","ADJ","ADV","PRON","DET","ADP","CONJ","NUM","PUNCT","OTHER"]
COMMON_DET = {"a","an","the","this","that","these","those"}
COMMON_ADP = {"in","on","at","by","with","for","to","of","from","as"}
COMMON_CONJ = {"and","or","but"}
COMMON_PRON = {"i","me","my","you","your","he","him","her","she","it","its","we","our","they","them","their"}

def sentence_segment(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    return SENT_SPLIT_RE.split(text) if text else []

def tokenize(sentence: str):
    return TOKEN_RE.findall(sentence)

def load_lemma_dict(path: Path):
    """
    Loads a file where each line contains "word<lemma" or "word<space>lemma" or "word<TAB>lemma".
    The referenced dataset on GitHub is usually tab-separated.
    """
    lemma = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                w, l = line.split("\t", 1)
            elif " " in line:
                w, l = line.split(" ", 1)
            else:
                continue
            lemma[w.lower()] = l.lower()
    return lemma

def lemmatize_token(tok: str, lemma_dict: dict):
    key = tok.lower()
    if key in lemma_dict:
        return lemma_dict[key]
    if key in FALLBACK_LEMMA_DICT:
        return FALLBACK_LEMMA_DICT[key]
    if re.fullmatch(r"[A-Za-z]+", tok):
        if key.endswith("ies") and len(key) > 3:
            return key[:-3] + "y"
        if key.endswith("ing") and len(key) > 5:
            return key[:-3]
        if key.endswith("ed") and len(key) > 4:
            return key[:-2]
        if key.endswith("s") and len(key) > 3 and not key.endswith("ss"):
            return key[:-1]
    return key

def is_stopword(tok: str):
    return tok.lower() in DEFAULT_STOPWORDS

def heuristic_pos(tok: str) -> str:
    t = tok.lower()
    if re.fullmatch(r"[^\w\s]", tok):
        return "PUNCT"
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return "NUM"
    if t in COMMON_DET:
        return "DET"
    if t in COMMON_ADP:
        return "ADP"
    if t in COMMON_CONJ:
        return "CONJ"
    if t in COMMON_PRON:
        return "PRON"
    if t.endswith("ly"):
        return "ADV"
    if t.endswith(("ous","ful","able","ible","al","ive","ic","y")) and len(t) > 3:
        return "ADJ"
    if t.endswith(("ing","ed")):
        return "VERB"
    if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok):
        return "NOUN"
    return "OTHER"

def manual_pos_tag(tokens, storage_path: Path):
    stored = {}
    if storage_path.exists():
        try:
            stored = json.loads(storage_path.read_text(encoding="utf-8"))
        except Exception:
            stored = {}

    out=[]
    for tok in tokens:
        key = tok.lower()
        if key in stored:
            out.append((tok, stored[key], "manual(cache)"))
            continue
        if re.fullmatch(r"[^\w\s]", tok):
            stored[key] = "PUNCT"
            out.append((tok, "PUNCT", "manual(auto-punct)"))
            continue

        print(f"\nToken: {tok}")
        print("Choose POS:", ", ".join(VALID_POS))
        choice = input("POS> ").strip().upper()
        if choice not in VALID_POS:
            choice = "OTHER"
        stored[key] = choice
        out.append((tok, choice, "manual"))

    storage_path.write_text(json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

def run_pipeline(text: str, lemma_dict: dict, pos_mode: str, manual_store: Path):
    results=[]
    for si, sent in enumerate(sentence_segment(text), start=1):
        tokens = tokenize(sent)
        tagged = manual_pos_tag(tokens, manual_store) if pos_mode == "manual" else [
            (t, heuristic_pos(t), "heuristic") for t in tokens
        ]

        sent_rows=[]
        for tok, pos, pos_src in tagged:
            lemma = lemmatize_token(tok, lemma_dict)
            removed = is_stopword(tok) or pos == "PUNCT"
            sent_rows.append((si, sent, tok, pos, pos_src, lemma, removed))
        results.append(sent_rows)
    return results

def write_tsv(results, out_path: Path):
    lines=[]
    for sent_rows in results:
        sid, sent = sent_rows[0][0], sent_rows[0][1]
        lines.append(f"Sentence {sid}: {sent}")
        lines.append("token\tpos\tlemma\tremoved")
        for (_, _, tok, pos, _src, lemma, removed) in sent_rows:
            lines.append(f"{tok}\t{pos}\t{lemma}\t{removed}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pos", choices=["heuristic","manual"], default="heuristic")
    ap.add_argument("--manual-store", default="manual_pos.json")
    ap.add_argument("--lemma-dataset", default=None, help="Path to external lemma dataset (optional)")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    lemma_dict = load_lemma_dict(Path(args.lemma_dataset)) if args.lemma_dataset else {}
    results = run_pipeline(text, lemma_dict, args.pos, Path(args.manual_store))
    write_tsv(results, Path(args.output))

    # Print final kept lemmas per sentence (for screenshots)
    for sent_rows in results:
        sid = sent_rows[0][0]
        kept = [lemma for (_, _, _tok, pos, _src, lemma, removed) in sent_rows if not removed]
        print(f"Sentence {sid} kept lemmas:", " ".join(kept))

if __name__ == "__main__":
    main()
