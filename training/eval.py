"""Evaluate a fine-tuned checkpoint against the held-out doping annotations."""
import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.extractor import DEFAULT_MODEL, ROOT, Extractor, KEYS, normalize_prediction


def graph_sets(graph):
    """Compare unique names and named links, collapsing duplicate entity names.

    Counts and exact_graph_matches ignore entity multiplicity and arbitrary IDs.
    Entity strings are compared exactly, including their whitespace.
    """
    bases, dopants, links = (graph[key] for key in ('basemats', 'dopants', 'dopants2basemats'))
    return {
        'basemats': set(bases.values()),
        'dopants': set(dopants.values()),
        'links': {(dopants[key], bases[target]) for key, targets in links.items() for target in targets},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--data', default=str(ROOT / 'data/labeled/test.json'))
    parser.add_argument('--limit', type=int, default=None, help='Evaluate a reproducible sample of this many relevant sentences')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    docs = json.loads(open(args.data, encoding='utf-8').read())
    examples = [sentence for doc in docs for sentence in doc['doping_sentences'] if sentence['relevant']]
    if args.limit is not None:
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))
    if not examples:
        raise ValueError('No relevant evaluation sentences')
    gold_graphs = []
    for index, item in enumerate(examples, 1):
        try:
            graph = normalize_prediction({key: item.get(key) for key in KEYS})
        except ValueError as exc:
            raise ValueError(f"Invalid gold graph for sentence {index}: {exc}") from exc
        gold_graphs.append(graph_sets(graph))
    model = Extractor(args.model)
    counts = {key: {'tp': 0, 'predicted': 0, 'gold': 0} for key in ('basemats', 'dopants', 'links')}
    valid = exact = 0
    for item, gold in zip(examples, gold_graphs):
        try:
            predicted = graph_sets(model.extract(item['sentence_text']))
            valid += 1
            exact += predicted == gold
        except ValueError:
            # Count invalid predictions as empty for recall, and report them separately.
            predicted = {key: set() for key in counts}
        for key, count in counts.items():
            count['tp'] += len(gold[key] & predicted[key])
            count['predicted'] += len(predicted[key])
            count['gold'] += len(gold[key])
    metrics = {}
    for key, count in counts.items():
        precision = count['tp'] / count['predicted'] if count['predicted'] else 0
        recall = count['tp'] / count['gold'] if count['gold'] else 0
        metrics[key] = {**count, 'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}
    print(json.dumps({'sentences': len(examples), 'valid_json': valid, 'exact_graph_matches': exact, 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
