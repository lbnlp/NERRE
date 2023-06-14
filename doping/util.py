"""
Utility functions for LLM-NERRE doping experiments.
"""
import json


def read_jsonl(filename):
    """
    Read a jsonlines file into a list of dictionaries.

    Args:
        filename (str): Path to input file.

    Returns:
        ([dict]): List of dictionaries.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        samples = []
        for l in lines:
            samples.append(json.loads(l))
        return samples


def dump_jsonl(d, filename):
    """
    Dump a list of dictionaries to file as jsonlines.

    Args:
        d ([dict]): List of dictionaries.
        filename (str): Path to output file.

    Returns:
        None
    """
    with open(filename, "w") as f:
        for entry in d:
            f.write(json.dumps(entry) + "\n")
