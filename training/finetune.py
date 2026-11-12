"""Fine-tune FLAN-T5 on the repository's sentence-level doping JSON data."""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.extractor import ROOT, DEFAULT_MODEL, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, normalize_prediction, decode_output, load_seq2seq_model


def read_examples(path):
    examples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        target = record["completion"].strip()
        if target.endswith("END"):
            target = target[:-3].strip()
        try:
            target = normalize_prediction(json.loads(target))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid annotation on line {line_number}") from exc
        examples.append((record["prompt"], json.dumps(target, ensure_ascii=True, separators=(",", ":"))))
    if not examples:
        raise ValueError("Training data is empty")
    return examples


def save_checkpoint(model, tokenizer, output, metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build the complete checkpoint alongside its destination before publishing it.
    with tempfile.TemporaryDirectory(prefix=".nerre-training-", dir=output.parent) as tmp:
        checkpoint = Path(tmp) / "checkpoint"
        checkpoint.mkdir()
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        (checkpoint / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        checkpoint.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/labeled/training_json.jsonl")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new directory or remove the old checkpoint")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AddedToken, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    examples = read_examples(args.data)
    # T5's vocabulary lacks JSON braces and some scientific characters. Add
    # missing characters from the training corpus, preserving targets as ASCII
    # JSON escapes so added Unicode tokens cannot introduce spaces in values.
    characters = sorted(set(''.join(prompt + target for prompt, target in examples)))
    missing = [char for char in characters if tokenizer.unk_token_id in tokenizer(char, add_special_tokens=False)['input_ids']]
    tokenizer.add_tokens([AddedToken(char, normalized=False) for char in missing])
    model = load_seq2seq_model(args.base_model).to(device)
    model.resize_token_embeddings(len(tokenizer))
    encoded = []
    for prompt, target in examples:
        source = tokenizer(prompt)["input_ids"]
        label = tokenizer(target)["input_ids"]
        if len(source) > MAX_INPUT_TOKENS or len(label) > MAX_OUTPUT_TOKENS:
            raise ValueError("Training example exceeds the model token limit; no data was silently truncated")
        try:
            decoded = json.loads(decode_output(tokenizer, label))
        except json.JSONDecodeError as exc:
            raise ValueError("Training target does not round-trip through the tokenizer") from exc
        if decoded != json.loads(target):
            raise ValueError("Training target does not round-trip through the tokenizer")
        encoded.append((source, label))

    def collate(batch):
        inputs = tokenizer.pad({"input_ids": [x[0] for x in batch]}, return_tensors="pt")
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x[1]) for x in batch], batch_first=True, padding_value=-100)
        return {**inputs, "labels": labels}

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss {loss_sum / len(loader):.4f}", flush=True)
    save_checkpoint(model, tokenizer, args.output, {"base_model": args.base_model, "data": str(args.data), "examples": len(examples), "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed})
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
