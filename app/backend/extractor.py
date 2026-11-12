"""Shared training and inference conventions for the doping JSON model."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "doping-flan-t5"
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256
ABBREVIATIONS = ("i.e.", "e.g.", "et al.", "etc.", "fig.", "figs.", "eq.", "ca.", "vs.", "approx.", "ref.", "no.")
PROMPT_SUFFIX = "\n\nExtract doping information from this sentence.\n\n###\n"
KEYS = ("basemats", "dopants", "dopants2basemats")


def prompt_for(sentence):
    return sentence.strip() + PROMPT_SUFFIX


def normalize_prediction(value):
    """Require the original doping schema, including valid graph references."""
    if not isinstance(value, dict) or set(value) != set(KEYS):
        raise ValueError("Model output does not match the doping JSON schema")
    bases, dopants, links = (value[key] for key in KEYS)
    if not all(isinstance(item, dict) for item in (bases, dopants, links)):
        raise ValueError("Model output fields must be objects")
    if not all(isinstance(k, str) and isinstance(v, str) for d in (bases, dopants) for k, v in d.items()):
        raise ValueError("Entity names must be strings")
    for key, targets in links.items():
        if key not in dopants or not isinstance(targets, list) or not all(isinstance(t, str) and t in bases for t in targets):
            raise ValueError("Model output contains invalid entity links")
    return value


def decode_output(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # The T5 tokenizer inserts a space after added backslash tokens.
    return re.sub(r"\\\s+u([0-9a-fA-F]{4})", r"\\u\1", text)


def parse_prediction(text):
    try:
        return normalize_prediction(json.loads(text.strip()))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Model returned invalid structured JSON") from exc


def split_sentences(text):
    # Blank lines separate sections; single newlines usually wrap a sentence.
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        flat = re.sub(r"\s*\n\s*", " ", paragraph)
        merged = []
        for part in (part.strip() for part in re.split(r"(?<=[.!?])\s+", flat)):
            if not part:
                continue
            previous = merged[-1] if merged else ""
            abbreviation = any(
                previous.lower().endswith(word)
                and (len(previous) == len(word) or not previous[-len(word) - 1].isalpha())
                # Ca and No are also element symbols; uppercase uses merge only
                # when followed by a number, as in "No. 2" or "ca. 3 K".
                and (word not in ("ca.", "no.") or previous[-len(word):] == word or part[0].isdigit())
                for word in ABBREVIATIONS
            )
            if abbreviation:
                merged[-1] += " " + part
            else:
                merged.append(part)
        sentences.extend(merged)
    return sentences


def load_seq2seq_model(model_path, **kwargs):
    """Preserve FLAN-T5's separate input/output embeddings under Transformers 5."""
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
    # Transformers 5 forces T5 tying on config load, including FLAN checkpoints
    # whose original tie_word_embeddings=False is retained as output scaling=False.
    # Restore it after loading so encoder/decoder embeddings are loaded as shared,
    # while a later vocabulary resize preserves the separate language-model head.
    if model.config.model_type == "t5" and not model.config.scale_decoder_outputs:
        model.config.tie_word_embeddings = False
    return model


class Extractor:
    def __init__(self, model_path):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = load_seq2seq_model(str(model_path), local_files_only=True)
        self.model.eval()

    def extract(self, sentence):
        inputs = self.tokenizer(prompt_for(sentence), return_tensors="pt")
        if self.tokenizer.unk_token_id in inputs["input_ids"]:
            raise ValueError("A sentence contains characters unsupported by the model tokenizer")
        if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            raise ValueError(f"A sentence exceeds the {MAX_INPUT_TOKENS}-token model limit")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKENS, do_sample=False)
        return parse_prediction(decode_output(self.tokenizer, output[0]))
