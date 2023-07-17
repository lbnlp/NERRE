import json
import openai
openai.api_key = "<YOUR_API_KEY>"


def load_annotation(annotation_string, stop_sequence="\n\nEND\n\n"):
    """Convert anntation string to dictionary
    
    example: 
        " [{\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"H1-2xPtxLaNb2O7\", \"structure_or_phase\": [\"perovskite\"], \"description\": [\"layered\", \"intercalated Pt2+\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"RbLaNb2O7\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"Pt-deposited RbLaNb2O7\", \"formula\": \"\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"HLaNb2O7\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}]\n\nEND\n\n"

    output: 
        [{'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'H1-2xPtxLaNb2O7', 'structure_or_phase': ['perovskite'], 'description': ['layered', 'intercalated Pt2+']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'RbLaNb2O7', 'structure_or_phase': [''], 'description': ['']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': 'Pt-deposited RbLaNb2O7', 'formula': '', 'structure_or_phase': [''], 'description': ['']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'HLaNb2O7', 'structure_or_phase': [''], 'description': ['']}]

    """
    s = annotation_string.replace(stop_sequence, "").strip()
    annotation = json.loads(s)
    return annotation


def pretty_print_annotation(annotation_string):
    """Pretty prints annotation string as structured json object"""
    print(json.dumps(load_annotation(annotation_string), indent=4))


def extract_materials_data(prompt, model, start_sequence=""):
    prompt = prompt + start_sequence
    response = openai.Completion.create(
      model = model,
      prompt=prompt,
      temperature=0,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n\nEND\n\n"],
    )
    return response.choices[0].text


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