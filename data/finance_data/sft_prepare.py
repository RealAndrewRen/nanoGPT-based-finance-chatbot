import os
import re
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

OUT_DIR = "data/finance_data"
MAX_LENGTH = 1024
VAL_RATIO = 0.1
TOKENIZER_NAME = "gpt2"

# ------------------------------------------------
# NOW INCLUDING 4 DATASETS
# ------------------------------------------------
datasets = [
    {
        "name": "Josephgflowers/Finance-Instruct-500k",
        "type": "sft",
        "split": "train"
    },
    {
        "name": "gbharti/finance-alpaca",
        "type": "sft",
        "split": "train"
    },
    {
        "name": "Aletheia-ng/personal_finance_v0.2",
        "type": "personal_finance",
        "split": "train"
    },
    {
        "name": "winddude/reddit_finance_43_250k",
        "type": "reddit",
        "split": "train"
    }
]

os.makedirs(OUT_DIR, exist_ok=True)
all_datasets = []

# ============================================================
# Cleaning regex
# ============================================================
URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.IGNORECASE
)

SLUR_PATTERN = re.compile(
    r"\b(fuck|shit|bitch|asshole|bastard|retard|nigger|fag)\b",
    re.IGNORECASE
)

def clean_joseph_text(text):
    # first run the general cleaning
    text = clean_text(text)
    
    # then remove LaTeX for Joseph dataset only
    text = re.sub(r"\$\$?.*?\$\$?", "", text)        # remove $...$ or $$...$$
    text = re.sub(r"\\[A-Za-z]+", "", text)          # remove LaTeX commands
    text = re.sub(r"[\[\]\{\}]", "", text)           # remove brackets and braces

    # collapse whitespace again in case cleaning added extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    # remove links and emails
    text = re.sub(URL_PATTERN, "", text)

    # remove slurs
    text = re.sub(SLUR_PATTERN, "[CLEANED]", text)

    # remove repeated punctuation
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Formatting functions
# ============================================================
def remove_human_assistant_prefixes(text: str):
    """
    Remove 'Human:' and 'Assistant:' labels from the Aletheia dataset.
    """
    if not text:
        return ""
    text = re.sub(r"\bHuman:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAssistant:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_sft(example):
    instr, resp = None, None

    if "instruction" in example and "output" in example:
        instr, resp = example["instruction"], example["output"]

    elif "user" in example and "assistant" in example:
        instr, resp = example["user"], example["assistant"]

    if instr and resp:
        instr = clean_text(instr)
        resp = clean_text(resp)
        return f"<user>\n{instr}\n</user>\n<assistant>\n{resp}\n</assistant>"
    return None


def format_personal_finance(example):
    """
    context = user; chosen = assistant
    Also strip Human:/Assistant:
    """
    if "context" in example and "chosen" in example:
        instr = remove_human_assistant_prefixes(clean_text(example["context"]))
        resp = remove_human_assistant_prefixes(clean_text(example["chosen"]))

        return f"<user>\n{instr}\n</user>\n<assistant>\n{resp}\n</assistant>"
    return None


def format_reddit(example):
    """
    selftext = user; body = assistant
    """
    if "selftext" in example and "body" in example:
        instr = clean_text(example["selftext"])
        resp = clean_text(example["body"])

        if instr and resp:
            return f"<user>\n{instr}\n</user>\n<assistant>\n{resp}\n</assistant>"
    return None


# ============================================================
# Save helper
# ============================================================
def save_texts_to_file(texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            t = str(t).strip()
            if t:
                f.write(t + "\n")


# ============================================================
# Dataset loading loop
# ============================================================
for ds_info in datasets:
    dataset_name = ds_info["name"].replace("/", "_")
    txt_path = os.path.join(OUT_DIR, f"{dataset_name}.txt")

    # load from cache if exists
    if os.path.exists(txt_path):
        print(f"✅ Skipping {ds_info['name']} (already saved)")
        with open(txt_path, "r", encoding="utf-8") as f:
            processed = [x.strip() for x in f if x.strip()]
        all_datasets.append(Dataset.from_dict({"text": processed}))
        continue

    print(f"📦 Loading {ds_info['name']}...")
    data = load_dataset(ds_info["name"], split=ds_info["split"])
    print(f"🔹 Using full dataset: {len(data)} rows")

    processed = []
    print(f"🪄 Formatting {ds_info['name']}...")

    for ex in tqdm(data, desc=f"Processing {ds_info['type']}"):
        # Joseph dataset: apply special cleaning
        if ds_info["name"] == "Josephgflowers/Finance-Instruct-500k":
            instr = clean_joseph_text(ex.get("user", ""))       # user column
            resp = clean_joseph_text(ex.get("assistant", ""))   # assistant column
            if instr and resp:
                out = f"<user>\n{instr}\n</user>\n<assistant>\n{resp}\n</assistant>"
            else:
                out = None

        # Other SFT datasets
        elif ds_info["type"] == "sft":
            out = format_sft(ex)

        elif ds_info["type"] == "personal_finance":
            out = format_personal_finance(ex)

        elif ds_info["type"] == "reddit":
            out = format_reddit(ex)

        else:
            out = None

        if out:
            processed.append(out)

    save_texts_to_file(processed, txt_path)
    print(f"💾 Saved {txt_path}")

    all_datasets.append(Dataset.from_dict({"text": processed}))



# ============================================================
# Merge all datasets
# ============================================================
print(f"🔗 Merging {len(all_datasets)} datasets...")
merged = concatenate_datasets(all_datasets)
print(f"📊 Total merged samples: {len(merged)}")

split = merged.train_test_split(test_size=VAL_RATIO, seed=42)
train_data, val_data = split["train"], split["test"]

print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

# ============================================================
# Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<user>", "</user>", "<assistant>", "</assistant>"]
}

tokenizer.add_special_tokens(SPECIAL_TOKENS)

user_token_id = tokenizer.convert_tokens_to_ids("<user>")
assistant_token_id = tokenizer.convert_tokens_to_ids("<assistant>")

print("USER TOKEN ID:", user_token_id)
print("ASSISTANT TOKEN ID:", assistant_token_id)

# ============================================================
# Encoding with assistant-only loss mask
# ============================================================
def encode_split(split_ds, tokenizer, filename, out_dir=OUT_DIR, max_length=MAX_LENGTH):
    token_path = os.path.join(out_dir, filename)
    mask_path = token_path.replace(".bin", "_mask.bin")

    all_tokens = []
    all_masks = []

    assistant_start_id = tokenizer.convert_tokens_to_ids("<assistant>")
    assistant_end_id = tokenizer.convert_tokens_to_ids("</assistant>")


    for ex in tqdm(split_ds, desc=f"Encoding {filename}"):

        text = str(ex["text"]).strip()
        if not text:
            continue

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        ids = encoded["input_ids"]
        mask = []
        in_assistant = False
        
        assistant_region = False
        for tok in ids:
            if tok == assistant_start_id:
            assistant_region = True

            mask.append(1 if assistant_region else 0)

            if tok == assistant_end_id:
                assistant_region = False



        all_tokens.extend(ids)
        all_masks.extend(mask)

    np.array(all_tokens, dtype=np.uint16).tofile(token_path)
    np.array(all_masks, dtype=np.uint8).tofile(mask_path)

    print(f"💾 Saved {token_path} & {mask_path}")


encode_split(train_data, tokenizer, "train.bin")
encode_split(val_data, tokenizer, "val.bin")
