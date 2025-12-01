# -*- coding: utf-8 -*-
import os
import re
import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0)
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import tiktoken


# ============================================================
#                     CONSTANTS
# ============================================================
enc = tiktoken.get_encoding("gpt2")

OUT_DIR = "data/finance_data"
MAX_LENGTH = 1024
VAL_RATIO = 0.1
TOKENIZER_NAME = "gpt2"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
#                     BASE REGEX (YOUR ORIGINAL)
# ============================================================

URL_RE = re.compile(
    r"""(?xi)
    (https?://[^\s]+)|
    (www\.[^\s]+)|
    ([A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)
    """
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BRACKETED_META_RE = re.compile(r"(\[[^\]]{1,200}\])|(\([^\)]{1,200}\))")

REPORTING_EDITING_RE = re.compile(
    r"(\(Reporting by [^\)]+?\))|(\(Reporting by [^\)]+?; Editing by [^\)]+?\))|"
    r"(\(Edited by [^\)]+?\))|(\(Compiled by [^\)]+?\))",
    flags=re.IGNORECASE,
)

BOILERPLATE_PHRASES = [
    r"the company did not immediately respond",
    r"the person declined to comment",
    r"no immediate comment was available",
    r"could not be reached for comment",
    r"for more information visit",
    r"follow us on",
    r"contact us at",
    r"further reporting by",
    r"source: [^\n]+",
    r"additional reporting by",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PHRASES), flags=re.IGNORECASE)

ALL_CAPS_DATELINE_RE = re.compile(r"^[A-Z0-9 ,'-]{2,}\s*-\s*", flags=re.MULTILINE)

TICKER_RE = re.compile(
    r"(\(|\[)?\b(NYSE|NASDAQ|LSE|TSX|HKEX|ASX|OTC|NYSEAMERICAN)\b[^\)\]]{0,40}(\)|\])?",
    flags=re.IGNORECASE
)
TICKER_SIMPLE_RE = re.compile(r"\b[A-Z]{2,5}\.(?:N|O|HE|L|K|F|T)\b")

REPEATED_PUNC_RE = re.compile(r"([!?.\-]{2,})")
MULTI_URLS_RE = re.compile(r"(https?://\S+[\s\r\n]*){2,}")

SIGNATURE_RE = re.compile(
    r"((^|\n)\s*(Regards|Best regards|Sincerely|Thanks|Thank you|Cheers|Yours)\b[^\n]{0,120})",
    flags=re.IGNORECASE,
)

TRAIL_SOURCE_TEXT_RE = re.compile(
    r"(?:^|\s)\bSource\s+text\s*:.*$", flags=re.IGNORECASE | re.DOTALL
)

HTML_ESC_RE = re.compile(r"&[a-z]+;")

MIN_SENTENCE_CHARS = 3


# ============================================================
#              NEW PATTERNS FROM SECOND SCRIPT
# ============================================================

LEAD_DATE_TIME_UPDATED = re.compile(
    r"""^\s*(
        (?:January|February|March|April|May|June|July|August|September|October|November|December)
        \s+\d{1,2},\s*\d{4}
        (?:\s*/\s*\d{1,2}:\d{2}\s*(?:AM|PM))?
        (?:\s*/\s*Updated[^A-Za-z0-9\n]+?\bago\b)?
    )\s+""",
    re.IGNORECASE | re.VERBOSE,
)

LEAD_REUTERS_DATELINE_2 = re.compile(
    r"""^\s*(
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}\s*)?
        \(Reuters\)\s*-\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

LEAD_MIN_READ = re.compile(
    r"^\s*Reuters\s+[A-Za-z]+\s+\d+\s+Min\s+Read\s*",
    re.IGNORECASE,
)

TRAIL_FURTHER_COVERAGE = re.compile(
    r"(?:^|\s)\bFurther\s+company\s+coverage\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)

TRAIL_URL_2 = re.compile(
    r"((\s|\n)+(?:https?://\S+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}\S*))\s*$",
    re.IGNORECASE | re.VERBOSE,
)

LEAD_FALLBACK_DATE = re.compile(
    r"""^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)
         \s+\d{1,2},\s*\d{4}\b[:,]?\s*""",
    re.IGNORECASE | re.VERBOSE,
)

ASTERISK_EMPH_RE = re.compile(r"\*{1,3}[^*\n]{1,200}?\*{1,3}")
LONE_ASTERISK_RE = re.compile(r"\s*\*\s*")

# Bloomberg dataset
PAT_BLOOMBERG_EMAIL = re.compile(r"""\b[A-Za-z0-9._%+-]+@bloomberg\.net\b""", re.IGNORECASE)

# ashraq dataset patterns
PAT_ASHRAQ_LINKS = re.compile(
    r"""\b(?:https?://\S+|bit\.ly/\S+|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}|[A-Za-z0-9.-]+\.com)\b""",
    re.IGNORECASE
)
PAT_ASHRAQ_TEL = re.compile(r"""(?:^|\s)\bTel:\s*[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_EMAIL = re.compile(r"""(?:^|\s)\bEmail:\s*[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_MORE_INFO = re.compile(r"""(?:^|\s)\bFor\s+more\s+information[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_INQUIRIES = re.compile(r"""(?:^|\s)\bFor\s+investor\s+and\s+media\s+inquiries[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_VIEW_SOURCE = re.compile(r"""(?:^|\s)\bView\s+source[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_SOURCE_EIKON = re.compile(r"""(?:^|\s)\bSource\s+text\s+for\s+Eikon[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_EMAIL_DIST = re.compile(r"""(?:^|\s)\bTo\s+be\s+included\s+in\s+the\s+company's\s+e-mail\s+distributions[^.]*\.""", re.IGNORECASE)
PAT_ASHRAQ_IN_MINUTES = re.compile(r"""\s+/\s+in\s+\d+\s+minutes\b""", re.IGNORECASE)
PAT_ASHRAQ_UPDATED_HOURS = re.compile(r"""\s+/\s+Updated\s+\d+\s+hours\s+ago\b""", re.IGNORECASE)

# ============================================================
#                 NEW EXTRA AGGRESSIVE PATTERNS
# ============================================================

# Remove [text] used as bold markers
BOLD_BRACKET_RE = re.compile(r"\[[^\]]+\]")

# Bloomberg contact info removal:
TRAIL_CONTACT_INFO = re.compile(
    r"To\s+contact\s+(?:the\s+)?(?:reporter|editor)[^\.]+(?:\.)?",
    re.IGNORECASE,
)

# Contributor lists:
CONTRIB_LIST_RE = re.compile(
    r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)?(?:,\s)?){2,}[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s+in\s+[A-Za-z]+(?:,\s[A-Z][a-z]+\s+in\s+[A-Za-z]+)*(?:;|\.)?",
    re.IGNORECASE,
)

# Headlines, timestamps, Min Read, BRIEF-, Updated X minutes ago
HEADLINE_TIME_RE = re.compile(
    r"^\s*(Updated\s+\d+\s+\w+\s+ago.*|"
    r"\d+\s+(Hours|Minutes)\s+Ago.*|"
    r"\d{1,2}[:\.]?\d{0,2}\s*(AM|PM)\s*/\s*Updated.*|"
    r"BRIEF-[^\n]+|"
    r"\d{1,2}\s+Min\s+Read.*)$",
    re.IGNORECASE | re.MULTILINE
)

# City dateline: NEW YORK -, LOS ANGELES -, PARIS -
CITY_DATELINE_RE = re.compile(
    r"^[A-Z][A-Z ]{2,30}\s*-\s*", re.MULTILINE
)


# ============================================================
#                   MODIFIED aggressive_clean()
# ============================================================

def _split_sentences(text: str) -> List[str]: pieces = re.split(r"(?<=[\.\?\!])\s+", text) return [p.strip() for p in pieces if p and len(p.strip()) >= MIN_SENTENCE_CHARS]

def aggressive_clean(raw: str) -> str:
    if not raw:
        return ""

    s = str(raw)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = ASTERISK_EMPH_RE.sub(" ", s)
    s = LONE_ASTERISK_RE.sub(" ", s)

    # Remove [bold] formatting
    s = BOLD_BRACKET_RE.sub(" ", s)

    # HTML removal
    s = HTML_TAG_RE.sub(" ", s)
    s = HTML_ESC_RE.sub(" ", s)

    # Emails
    s = EMAIL_RE.sub(" ", s)
    s = PAT_BLOOMBERG_EMAIL.sub(" ", s)

    # URLs
    s = MULTI_URLS_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = PAT_ASHRAQ_LINKS.sub(" ", s)

    # Remove headline/timestamp junk
    s = HEADLINE_TIME_RE.sub(" ", s)
    s = CITY_DATELINE_RE.sub(" ", s)

    # Lead removals
    for regex in [
        LEAD_DATE_TIME_UPDATED,
        LEAD_REUTERS_DATELINE_2,
        LEAD_MIN_READ,
        LEAD_FALLBACK_DATE,
    ]:
        s = regex.sub(" ", s)

    # Parentheticals
    s = REPORTING_EDITING_RE.sub(" ", s)

    # Remove reporter/editor contact lines (Bloomberg style)
    s = TRAIL_CONTACT_INFO.sub(" ", s)

    # Remove contributor name chains
    # s = CONTRIB_LIST_RE.sub(" ", s)

    # Ashraq patterns
    for regex in [
        PAT_ASHRAQ_TEL,
        PAT_ASHRAQ_EMAIL,
        PAT_ASHRAQ_MORE_INFO,
        PAT_ASHRAQ_INQUIRIES,
        PAT_ASHRAQ_VIEW_SOURCE,
        PAT_ASHRAQ_SOURCE_EIKON,
        PAT_ASHRAQ_EMAIL_DIST,
        PAT_ASHRAQ_IN_MINUTES,
        PAT_ASHRAQ_UPDATED_HOURS,
    ]:
        s = regex.sub(" ", s)

    # Boilerplate
    s = BOILERPLATE_RE.sub(" ", s)

    # Signatures
    s = SIGNATURE_RE.sub(" ", s)

    # Tickers
    s = TICKER_RE.sub(" ", s)
    s = TICKER_SIMPLE_RE.sub(" ", s)

    # Trailing junk
    s = TRAIL_SOURCE_TEXT_RE.sub(" ", s)
    s = TRAIL_FURTHER_COVERAGE.sub(" ", s)
    s = TRAIL_URL_2.sub(" ", s)

    # All caps dateline
    s = ALL_CAPS_DATELINE_RE.sub(" ", s)

    # Repeated punctuation
    s = REPEATED_PUNC_RE.sub(lambda m: m.group(1)[0], s)

    # Whitespace cleanup
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("(Reuters)", " ")
    s = s.strip()

    # Sentence filtering
    sentences = _split_sentences(s)
    clean_sents = []
    for sent in sentences:
        if len(sent) < MIN_SENTENCE_CHARS:
            continue
        alnum_ratio = sum(c.isalnum() for c in sent) / max(1, len(sent))
        if alnum_ratio < 0.2:
            continue
        if len(sent.split()) <= 1:
            continue
        if BOILERPLATE_RE.search(sent):
            continue
        clean_sents.append(sent)

    final = " ".join(clean_sents)
    final = re.sub(r"[^ -~]+", " ", final)
    final = re.sub(r"\s{2,}", " ", final).strip()
    return final


# ============================================================
#                    DATASETS LIST
# ============================================================

datasets_config = [
    {"name": "alvanlii/finance-textbooks", "cols_to_parse": ["book_text"], "split": "train"},
    {"name": "genloop/bloomberg_financial_news_120k", "cols_to_parse": ["Article"], "split": "train"},
    {"name": "ashraq/financial-news-articles", "cols_to_parse": ["text"], "split": "train"},
    {"name": "edaschau/financial_news", "cols_to_parse": ["article_text"], "split": "train", "sample_frac": 0.25},
]


# ============================================================
#                   LOAD + CLEAN
# ============================================================

all_datasets = []

for cfg in datasets_config:
    dataset_name = cfg["name"].replace("/", "_")
    txt_path = f"{OUT_DIR}/{dataset_name}.txt"

    # If already processed → load
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            arr = [line.strip() for line in f if line.strip()]
        all_datasets.append(Dataset.from_dict({"text": arr}))
        continue

    print(f"📦 Loading {cfg['name']}...")
    try:
        ds = load_dataset(cfg["name"], split=cfg["split"])
    except Exception as e:
        print("❌ Failed:", e)
        continue

    # Optional sampling
    if "sample_frac" in cfg:
        N = len(ds)
        k = int(N * cfg["sample_frac"])
        k = max(k, 1000)
        ds = ds.shuffle(seed=42).select(range(k))
        print(f"🔹 Sampled {k}/{N}")

    clean_list = []

    with open(txt_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"Cleaning {cfg['name']}"):
            parts = []
            for col in cfg["cols_to_parse"]:
                if col in row and row[col]:
                    cleaned = aggressive_clean(row[col])
                    if cleaned:
                        parts.append(cleaned)

            if not parts:
                continue

            combined = " ".join(parts)
            if len(combined) < 30:
                continue

            clean_list.append(combined)
            f.write(combined + "\n")

    all_datasets.append(Dataset.from_dict({"text": clean_list}))
    print(f"💾 Saved {txt_path} ({len(clean_list)} samples)")


# ============================================================
#                    MERGE + SPLIT
# ============================================================

merged = concatenate_datasets(all_datasets)
print("📊 Total samples:", len(merged))

split = merged.train_test_split(test_size=VAL_RATIO, seed=42)
train_data, val_data = split["train"], split["test"]


# ============================================================
#               TOKENIZE  →  train.bin / val.bin
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if not tokenizer.pad_token:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

def encode_split(ds, filename):
    out_file = f"{OUT_DIR}/{filename}"
    if os.path.exists(out_file):
        print("⚠️ Already exists, skipping:", out_file)
        return

    buf = []
    for ex in tqdm(ds, desc=f"Encoding {filename}"):
        text = ex["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, truncation=True, max_length=MAX_LENGTH)
        buf.extend(ids)

    arr = np.array(buf, dtype=np.uint16)
    arr.tofile(out_file)
    print(f"💾 Wrote {out_file} ({arr.nbytes} bytes)")


encode_split(train_data, "train.bin")
encode_split(val_data, "val.bin")

print("✅ Done.")
