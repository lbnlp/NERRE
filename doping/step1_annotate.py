"""
This is a script for annotating data relevant to doping.
The doping data is annotated on a sentence-by-sentence basis.
It is a command line utility meant to be used interactively through the command line.
See the --help for this script for more help.
"""

import os
import json
import pprint
import argparse
import random
import warnings
import time
from subprocess import call
import datetime
import traceback
import sys

from chemdataextractor.doc import Paragraph
from monty.serialization import loadfn, dumpfn

from constants import DATADIR

MULTI_DELIMITER = "||"
YN_SUFFIX = "(y/n, or enter for yes): "
NULL_SUFFIX = "(enter text, or enter for 'null'): "
MULTI_SEPARATION = f"separated by '{MULTI_DELIMITER}'"
MULTI_AND_NULL = f"{MULTI_SEPARATION} {NULL_SUFFIX}"

ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
            'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
            'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
            'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
            'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
            'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
            'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
            'Lv', 'Ts', 'Og', 'Uue']


def wrap_input(
        prompt,
        null_possible=False,
        multi_and_null=False
):
    """
    Wrap the input for an entry from <input>. Note "prompt" here does
    not correspond to anything relevant to LLMs, it is just a command
    line utility.

    Args:
        prompt (str): The prompt to display to the user.
        null_possible (bool): Whether or not the user can enter "null" as an input.
        multi_and_null (bool): Whether or not the user can enter multiple inputs

    Returns:
        (str) The updated prompt
    """
    if null_possible:
        prompt = f"{prompt} {NULL_SUFFIX}"
    elif multi_and_null:
        prompt = f"{prompt} {MULTI_AND_NULL}"

    try:
        return input(prompt)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except BaseException as BE:
        print(f"Input not captured because of error {BE}")


def yn_input(prompt):
    """
    A wrapper for yes/no input for <input>.
    Note prompt here does not correspond to anything relevant to LLMs, it is just a command
    line utility.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        (bool) True if the user entered "y", False if the user entered "n".
    """
    satisfied = False
    output = None

    yn_map = {
        "": True,
        "n": False,
        "y": True
    }

    while not satisfied:
        yn = wrap_input(f"{prompt} {YN_SUFFIX}")
        if isinstance(yn, str):
            yn = yn.lower()
        if yn not in yn_map:
            print(f"Input {yn} not recognized as one of 'y', 'n', None.")
            continue
        else:
            satisfied = True
            output = yn_map[yn]
    return output


def colored(r, g, b, text):
    """
    Color text for ease of annotating.

    Args:
        r (int): Red value.
        g (int): Green value.
        b (int): Blue value.
        text (str): Text to color.

    Returns:
        (str) The colored text.
    """
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)


def wrap_input_multi(prompt):
    """
    Wrap multiple inputs.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        (list) The list of inputs.
    """
    wrapped = wrap_input(prompt, multi_and_null=True)
    return [item for item in wrapped.split(MULTI_DELIMITER) if item]


def preprocess_text(text):
    """
    Preprocess a string to be presented to the user.

    Args:
        text (str): The text to preprocess.

    Returns:
        (str, list) The preprocessed text, and the list of chemical entities.
    """
    for tok in ("<inf>", "</inf>", "<sup>", "</sup>", "<hi>", "</hi>", "<sub>", "</sub", "$$", "\hbox", "\emph", "\\bf"):
        text = text.replace(tok, "")

    text = text.replace("\n", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    p = Paragraph(text)
    sentences = [s.text for s in p.sentences]
    cems = [[c.text for c in s.cems] for s in p.sentences]
    return sentences, cems


def sentence_is_paradigm(sentence, cems):
    """
    Returns true if the sentence probably has to do with doping.

    Used as a basic relevance screener to avoid inferring on
    sentences which do not have to do with doping.

    Args:
        sentence (str): The sentence to check.
        cems (list): The chemical entities (as per CDE) in the sentence.

    Returns: (bool)

    """
    # Paradigm: has a directly doping related word
    if any([paradigm in sentence.lower() for paradigm in (" dop", "-dop", "n-type", "p-type", "codop")]):
        return True
    elif cems:

        # Paradigm: has a host:dopant type syntax
        if ":" in sentence:
            possible_subtoks = [f":{cem}" for cem in cems] + \
                               [f"{cem}:" for cem in cems] + \
                               [f"{cem} :" for cem in cems] + \
                               [f": {cem}" for cem in cems] + \
                               [f":{el}+" for el in ELEMENTS] + \
                               [f":{el}-" for el in ELEMENTS] + \
                               [f":{el} " for el in ELEMENTS]
            if any([pst.lower() in sentence.lower() for pst in possible_subtoks]):
                return True

    # Paradigm: has a solid-solution-like material
    # May occur even if the cem is not recognized due to bad CDE
    if any([pst in sentence.lower() for pst in ("-x", "+x", "-y", "+y", "-z", "+z", "−x", "−y", "−z")]):
        return True


def annotate_paradigm(s, cems, entry):
    """
    Annotate a single sentence.

    Args:
        s (str): The sentence to annotate.
        cems ([str]): A list of chemical entities from CDE
        entry (dict): The entry to annotate.

    Returns:
        (dict, bool) The updated entry, and whether or not the entry was accepted by user.
    """
    print("Sentence contains possible doping information.")
    s_pretty = s.replace("dop", colored(255, 0, 0, "dop")).replace(":", colored(255, 0, 0, ":"))

    print(f"\n\t{s_pretty}\n\n")

    relevant = yn_input("Sentence contains any doping info?")

    if relevant:

        basemats = wrap_input_multi("Enter all base materials: ")
        dopants = wrap_input_multi("Enter all dopants: ")
        results = wrap_input_multi("Enter all results  or solid solutions (1-x's, specific stoichiometries): ")
        basemats = {f"b{i}": bmat for i, bmat in enumerate(basemats)}
        dopants = {f"d{i}": dop for i, dop in enumerate(dopants)}
        results = {f"r{i}": res for i, res in enumerate(results)}

        # Link dopants to basemats
        dopants2basemats = {}

        if basemats:
            for didx, dname in dopants.items():
                links = wrap_input_multi(
                    f"What basemats among \n {pprint.pformat(basemats)} \nare linked to dopant {dname}, per their index?")
                links = [idx.strip() for idx in links]
                dopants2basemats[didx] = links

        # Enter any modifiers for this sentence
        modifiers = wrap_input_multi(f"Enter any modifiers for this doping mention (doping amounts, non-doping, self-doping, etc.)")

        entry = {
            "sentence_text": s,
            "sentence_cems": cems,
            "basemats": basemats,
            "dopants": dopants,
            "results": results,
            "dopants2basemats": dopants2basemats,
            "doping_modifiers": modifiers,
            "relevant": True,
        }
    else:
        entry["relevant"] = False

    pprint.pprint(entry)
    accepted = yn_input(colored(255, 255, 0, "Accept entry?"))

    return entry, accepted


def annotate_doping_basic(doc):
    """

    Do a basic annotation of an abstract (or document)/

    Doc must have the following fields:

    - 'abstract'
    - 'doi'
    - 'title'

    Args:
        doc (dict): The document to annotate. Must have abstract, doi, and title fields.

    Returns:
        extracted (dict): The extracted information, annotated on a sentence-by-sentence basis.

    """

    title = doc["title"]
    doi = doc["doi"]
    text = doc["abstract"]

    extracted = {
        "doi": doi,
        "text": text,
        "title": title,
        "doping_sentences": []
    }

    title_and_text = f"{title}. {text}" if title else text
    sentences, cems_per_sentence = preprocess_text(title_and_text)


    entries = []
    for i, s in enumerate(sentences):
        cems = cems_per_sentence[i]

        entry = {
            "sentence_text": s,
            "sentence_cems": cems,
            "basemats": {},
            "dopants": {},
            "results": {},
            "doping_modifiers": [],
            "dopants2basemats": {},
            "relevant": False
        }

        if sentence_is_paradigm(s, cems):
            entry, accepted = annotate_paradigm(s, cems, entry)

        else:
            print(colored(255, 0, 0, f"\tSentence \t\t\n'{s}'\n \tdoes not contain paradigm."))
            accepted = yn_input(colored(255, 255, 0, "Accept empty entry?"))

            if not accepted:
                entry, accepted = annotate_paradigm(s, cems, entry)

        if accepted:
            print(f"\t\t{colored(0, 255, 0, 'Entry accepted!')}")
            entries.append(entry)
        else:
            while not accepted:
                print("\nFinal entry:\n")
                pprint.pprint(entry)
                print("\n")

                accepted = yn_input(f"Accept extracted entry?")

                if accepted:
                    entries.append(entry)
                else:
                    edit_it = yn_input(f"Edit entry manually?")

                    if edit_it:
                        EDITOR = os.environ.get('EDITOR', 'vim')  # that easy!
                        tfname = ".tmpfile.json"

                        dumpfn(entry, tfname)
                        call([EDITOR, tfname])

                        try:
                            entry = loadfn(tfname)
                        except:
                            warnings.warn("Failure to read file (likely bad formatting!) Reverting...")
                            accepted = False

    n_sentences = len(entries)
    n_relevant = len([e for e in entries if e["relevant"]])
    n_irrelevant = n_sentences - n_relevant
    print(f"Extracted {n_relevant} sentence level doping graphs from {n_sentences} (ignored {n_irrelevant}).")

    extracted["doping_sentences"] = entries
    return extracted

if __name__ == "__main__":
    p = argparse.ArgumentParser(fromfile_prefix_chars='@')

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    p.add_argument(
        '--corpus_file',
        help='Specify the JSON file to read from. Must contain a "doi", "title", '
             'and "abstract" fields in each document. Default is only the 162 '
             'abstracts used in the publication ("raw_data.json"). In practice, '
             'more abstracts should be used.',
        default=os.path.join(DATADIR, "raw_data.json"),
        required=False
    )

    p.add_argument(
        '--output_file',
        help='Specify the output file to dump json annotations into.',
        default=os.path.join(DATADIR, f"all_annotations_{dt}.json"),
        required=False
    )

    p.add_argument(
        '--n',
        type=int,
        help="Specify the number of documents to annotate.",
        required=False
    )

    p.add_argument(
        "--randomize",
        type=bool,
        default=False,
        help="If True, gets a random aggregation of n samples from the dataset.",
        required=False
    )

    args = p.parse_args()
    # _file = args.file
    output_name = args.output_file
    source_file = args.corpus_file
    n_samples = args.n
    randomize = args.randomize

    print(f"Reading source file {source_file}")
    t0 = time.time()
    docs = loadfn(source_file)
    t1 = time.time()
    print(f"Loaded source file (took {t1 - t0} seconds).")

    if randomize:
        docs = random.choices(docs, k=n_samples)

    j = 0
    annotated = []
    for doc in docs:
        j += 1
        print(f"\n\nDoc {j} of {n_samples}: {doc['doi']}")
        print("----"*10)

        repeat = True
        do_exit = None
        while repeat:
            try:
                ed = annotate_doping_basic(doc)
                annotated.append(ed)
                repeat = False
            except BaseException as BE:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                warnings.warn("Exited by Exception!")

                print(traceback.format_exception(exc_type, exc_value, exc_traceback))

                do_exit = wrap_input("Exit (e), redo (r), or skip (s)? ")
                do_exit = do_exit.lower() if do_exit else do_exit

                if do_exit == "e":
                    repeat = False
                    break
                elif do_exit == "r":
                    repeat = True
                    continue
                elif do_exit == "s":
                    repeat = False
                    break

        if do_exit in (None, "r", "s"):
            continue

        if do_exit == "e":
            break

        if j == n_samples:
            print(f"Completed {n_samples} annotations.")
            break

    print(f"Writing {len(annotated)} annotations to file {output_name}!")
    with open(output_name, "w") as f:
        json.dump(annotated, f)
    print("File written, exiting.")