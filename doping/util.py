import json

def read_jsonl(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        samples = []
        for l in lines:
            samples.append(json.loads(l))
        return samples


def dump_jsonl(d, filename):
    with open(filename, "w") as f:
        for entry in d:
            f.write(json.dumps(entry) + "\n")