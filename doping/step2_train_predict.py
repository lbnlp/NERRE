"""
Script for:

- formatting LLM-NERRE JSONL training files in different schemas for doping
- training a GPT-3 model with fine-tuning
- using trained models to infer on new data (e.g., for performance evaluation).

Use --help with this script for more information on usage.
"""

import copy
import os
import time
import sys
import traceback
import pprint
import json

import argparse
import openai
from openai.error import RateLimitError
import tqdm
from monty.serialization import loadfn, dumpfn
import warnings
import datetime

from constants import DATADIR
from util import dump_jsonl
from step1_annotate import preprocess_text, sentence_is_paradigm


START_TOKEN = "\n###\n"
UNKNOWN_STR = "unknown"
STOP_TOKEN = "\nEND"
WHITESPACE = " "

def llm_completion_from_sentence_json(
        sentence_json,
        write_links=True,
        write_nonlinked_basemats=True,
        write_nonlinked_dopants=True,
        write_results=True,
        write_modifiers=True,
        stop_token=STOP_TOKEN,
        whitespace=WHITESPACE,
        fmt="eng"
):
    """
    Create an LLM completion (target) from a sentence json according to different schemas.
    Used for training.

    Args:
        sentence_json (dict): A dictionary for a sentence with "sentence_text" field and other keys
            relevant for doping (basemats, dopants, doping_modifierts, dopants2basemats, results.
        write_links (bool): Whether to write the links between dopants and basemats.
        write_nonlinked_basemats (bool): Whether to write "isolated" basemats.
        write_nonlinked_dopants (bool): whether to write "isolated" dopants
        write_results (bool): Whether to write the results.
        write_modifiers (bool): Whether to write the doping modifiers.
        stop_token (str): The stop token to use.
        whitespace (str): The whitespace to use.
        fmt (str): The format to create a completion in. Note this does not mean the schema, but only
            whether the completion should be written as english sentences or as stringified JSON. Should
            be either "json" or "eng"; to use EngExtra as in the publication, use write_results=True and
            write_modifiers=True.

    Returns:
        str: The GPT-3 completion to be used for training.

    """
    if fmt not in ("eng", "json"):
        raise ValueError(f"Value of fmt='{fmt}' not valid!")

    if fmt == "json":
        keys = ["basemats", "dopants", "dopants2basemats"]
        if write_results:
            keys.append("results")
        if write_modifiers:
            keys.append("doping_modifiers")
        subjson = {k: v for k, v in sentence_json.items() if k in keys}
        output = json.dumps(subjson, indent=1)
    else:
        output = ""
        basemats = sentence_json["basemats"]
        dopants = sentence_json["dopants"]
        modifiers = sentence_json["doping_modifiers"]
        links = sentence_json["dopants2basemats"]
        results = sentence_json["results"]

        basemats_left = copy.deepcopy(basemats)
        dopants_left = copy.deepcopy(dopants)

        if links and write_links:
            for dopant_id, d2b_links in links.items():
                dopant = dopants[dopant_id]
                for basemat_id in d2b_links:
                    basemat = basemats[basemat_id]
                    output += f"The host '{basemat}' was doped with '{dopant}'.\n"

                    if basemat_id in basemats_left:
                        basemats_left.pop(basemat_id)
                dopants_left.pop(dopant_id)

        if basemats_left and write_nonlinked_basemats:
            for basemat in basemats_left.values():
                output += f"The host '{basemat}' was doped.\n"

        if dopants_left and write_nonlinked_dopants:
            for dopant in dopants_left.values():
                output += f"'{dopant}' is a dopant.\n"

        if write_results:
            for result in results.values():
                output += f"'{result}' is a likely solid solution.\n"

        if modifiers and write_modifiers:
            modifier_str = ", ".join([f"'{m}'" for m in modifiers])
            output += f"Modifiers of the doping are: {modifier_str}.\n"

        if not output:
            output = "There is no doping information.\n"

    output = whitespace + output + stop_token
    return output


def decode_entities_from_llm_completion(text, fmt="eng"):
    """
    Obtain entities as a dictionary (to be converted to json) from a GPT-3 completion
    string. Used for decoding LLM string replies to structured doping data.

    Args:
        text (str): The LLM completion string.
        fmt (str): The format to decode from, either "eng" or "json". Extra entities
            are automatically decoded if present.

    Returns:
        (dict): The structured doping entities representing a graph (JSON document).
    """
    if fmt not in ("eng", "json"):
        raise ValueError(f"Value of fmt='{fmt}' not valid!")

    ents = {
        "basemats": {},
        "dopants": {},
        "results": {},
        "doping_modifiers": {},
        "dopants2basemats": {},
    }

    if not text:
        return ents

    if fmt == "json":
        try:
            ents = json.loads(text)
        except json.decoder.JSONDecodeError:
            warnings.warn(f"Could not json decode entry '{text}'")
        return ents

    # todo: implement doping modifiers and results
    text = text.strip()

    if "There is no doping information" in text:
        return ents

    lines = [l for l in text.split("\n") if l]

    dopant_counter = 0
    basemat_counter = 0

    results = []
    modifiers = []

    for l in lines:
        inverted_basemats = {v: k for k, v in ents["basemats"].items()}
        inverted_dopants = {v: k for k, v in ents["dopants"].items()}

        if l[-1] == ".":
            l = l[:-1]

        # print(l)
        basemat = None
        dopant = None
        result = None
        modifier_list = None

        if "The host" in l and "was doped with" in l:
            # has basemats and dopants linked
            try:
                left, right = l.split("was doped with")
            except ValueError:
                return ents

            right = [r.strip() for r in right.split("'") if r.strip()]
            left = [le.strip() for le in left.split("'") if le.strip() and "The host" not in le]

            if not right or not left:
                return ents

            if len(left) != 1:
                left = [" ".join(left)]
            elif len(right) != 1:
                right = [" ".join(right)]
                # raise BaseException(f"Left or right split on link was longer than 1!\nLeft was {left} and right was {right}")

            basemat = left[0]
            dopant = right[0]

        elif "The host" in l and "was doped" in l:
            left, _ = l.split("was doped")
            left = [le.strip() for le in left.split("'") if le.strip() and "The host" not in le]

            if len(left) != 1:
                # raise BaseException(f"Left split on basemat was longer than 1!\nLeft was {left}")
                left = [" ".join(left)]
            basemat = left[0]

        elif "is a dopant" in l:

            split = l.split("is a dopant")
            left = "".join(split[:-1])
            left = [le.strip() for le in left.split("'") if le.strip()]

            if len(left) != 1:
                # raise BaseException(f"Left split on dopant was longer than 1!\nLeft was {left}")
                left = [" ".join(left)]
            dopant = left[0]

        elif "is a likely solid solution" in l:
            left, _ = l.split("is a likely solid solution")
            left = [le.strip() for le in left.split("'") if le.strip()]
            if len(left) != 1:
                left = " ".join(left)
            else:
                left = left[0]
            result = left

        elif "Modifiers of the doping are" in l:
            if l[-1] == ".":
                l = l[:-1]
            _, right = l.split("Modifiers of the doping are:")
            right = [ri.strip() for ri in right.split("'") if
                     len(ri.strip()) > 1]
            modifier_list = right

        else:
            warnings.warn(f"Line {l} gave no parsable data!")
            continue

        if basemat:
            if basemat in inverted_basemats:
                bid = inverted_basemats[basemat]
            else:
                bid = f"b{basemat_counter}"
                basemat_counter += 1

            ents["basemats"][bid] = basemat

        if dopant:
            if dopant in inverted_dopants:
                did = inverted_dopants[dopant]
            else:
                did = f"d{dopant_counter}"
                dopant_counter += 1

            ents["dopants"][did] = dopant

        if basemat and dopant:
            if did in ents["dopants2basemats"]:
                ents["dopants2basemats"][did].append(bid)
            else:
                ents["dopants2basemats"][did] = [bid]

        if dopant and not basemat:
            if did not in ents["dopants2basemats"]:
                ents["dopants2basemats"] = []

        if result:
            results.append(result)
        if modifier_list:
            modifiers += modifier_list

    ents["doping_modifiers"] = {f"m{i}": m for i, m in enumerate(modifiers)}
    ents["results"] = {f"r{i}": r for i, r in enumerate(results)}
    # print(f"Text:\n{text}\n\nResulted in {pprint.pformat(ents)}\n")
    return ents


def llm_prompt_from_sentence_json(
        sentence_json,
        include_relevance_hint=False,
        include_question=True,
        start_token=START_TOKEN,
):
    """
    Create an LLM prompt from a sentence's json representation.

    Args:
        sentence_json (dict): The JSON dict representation of the sentence.
        include_relevance_hint (bool): Whether to include a hint about the relevance of the sentence.
            Not used in publication, and in practice, does not actually affect performance.
        include_question (bool): Whether to include a question about the sentence (i.e., an instruction).
        start_token (str): The start token to use.

    Returns:
        str: The prompt for the LLM.
    """
    text = sentence_json["sentence_text"]
    relevant = sentence_json["relevant"]

    if relevant:
        relevance_hint = "This text probably has information about doping."
    else:
        relevance_hint = "This text probably does not have information about doping."

    if include_relevance_hint:
        text = f"{text}\n\n{relevance_hint}"

    if include_question:
        text = f"{text}\n\nExtract doping information from this sentence."

    return f"{text}\n{start_token}"


def create_jsonl(
        abstracts_raw_data,
        output_filename,
        include_irrelevant=False,
        dry_run=False,
        prompt_kwargs={},
        completion_kwargs={},
        fmt="eng"
):
    """
    Create a JSONL file from a list of abstracts (annotated or LLM-completed).
    Used for training of the LLM.

    Dry run means it will the prompts and completions to the console and not write them to file.


    Args:
        abstracts_raw_data ([dict]): List of documents to create the JSONL from.
        output_filename (str): The filename to write the JSONL to.
        include_irrelevant (bool): Whether to include irrelevant sentences.
        dry_run (bool): Whether to do a dry run (i.e., not write to file).
        prompt_kwargs (dict): Keyword arguments to pass to llm_prompt_from_sentence_json.
        completion_kwargs (dict): Keyword arguments to pass to llm_completion_from_sentence_json.

    Returns:
        None

    """
    completions = []
    prompts = []

    for i, abstract_extracted in enumerate(abstracts_raw_data):
        for s in abstract_extracted["doping_sentences"]:

            if not s["relevant"] and not include_irrelevant:
                if dry_run:
                    print("SKIPPED FOR RELEVANCE", s["sentence_text"])
                continue

            prompt = llm_prompt_from_sentence_json(s, **prompt_kwargs)
            completion = llm_completion_from_sentence_json(s, fmt=fmt, **completion_kwargs)

            if dry_run:
                print(abstract_extracted["doi"])
                pprint.pprint(s)
                print("n")
                print(f"PROMPT:\n{prompt}\n")
                print(f"COMPLETION:\n{completion}\n")
                print("-"*30 + "\n\n")

            prompts.append(prompt)
            completions.append(completion)

    if dry_run:
        print("File not written as dry_run=True")
    else:
        with open(output_filename, "w") as f:
            for i, c in enumerate(completions):
                sample = {
                    "prompt": prompts[i],
                    "completion": completions[i]
                }

                j = json.dumps(sample)
                f.write(j)
                f.write("\n")

        print(f"file written to {output_filename} with {len(completions)} sentence samples.")


def create_sentences_json_for_inference(entry):
    """
    Prepare an entry for prediction with an LLM.
    Entry must have abstract, doi, and title fields.

    Args:
        entry (dict): The entry to prepare.

    Returns:
        (dict): The updated, preprocessed entry.
    """
    title = entry["title"]
    doi = entry["doi"]
    text = entry["text"]
    title_and_text = f"{title}. {text}" if title else text
    sentences, cems_per_sentence = preprocess_text(title_and_text)

    entry = {
        "doi": doi,
        "title": title,
        "text": text,
        "doping_sentences": [{"sentence_text": s, "sentence_cems": cems_per_sentence[i]} for i, s in enumerate(sentences)]
    }
    return entry


# Major core functions
def gpt3_finetune(
        data_training,
        training_filename,
        fmt="eng",
        write_extras=False,
        n_epochs=7
):
    """
    Fine tune a doping model using data from the annotation script.

    MUST adhere to the annotation script formatting for the json.

    Args:
        data_training (list): The training data, in the annotation script heirarchical format.
        training_filename (str): the name of the file you want to save
            the jsonl training tuples to. E.g., "my_GPT3_training_file_version1.jsonl".
        fmt (str): Either "json" or "eng". Note to use ExtraEng use "eng" with write_extras=True.
        write_extras (bool): Whether to write extras' information (results, modifiers) to the
            training file.
        n_epochs (int): The number of epochs to use for training.

    Returns:
        data_training_dois ([str]): The list of dois included here for training.
        training_filename (str): The name of the training file output as jsonl.
    """
    print("loading training set")
    data_training_dois = [d["doi"] for d in data_training]
    print("training set loaded.")

    create_jsonl(
        data_training,
        output_filename=training_filename,
        include_irrelevant=False,
        dry_run=False,
        prompt_kwargs=dict(
            include_relevance_hint=False,
            include_question=True
        ),
        completion_kwargs=dict(
            write_links=True,
            write_nonlinked_dopants=True,
            write_nonlinked_basemats=True,
            write_results=write_extras,
            write_modifiers=write_extras
        ),
        fmt=fmt
    )
    print(f"JSONL written to {training_filename}.")

    os.system(f"openai api fine_tunes.create -t '{training_filename}' -m 'davinci' --n_epochs={n_epochs}")

    print(f"Model fine-tuning is in progress. Raw training JSONL data stored at {training_filename}.")
    return data_training_dois, training_filename


def gpt3_infer(
        data_inference,
        model,
        output_filename=None,
        save_every_n=100,
        halt_on_error=False,
):
    """
    Infer gpt3 entries from raw data (e.g., from a dump of a mongodb query).

    Args:
        data_inference ([dict]): List of documents for inference. MUST have
            the following fields: "text", "title", "doi".
        model (str): The OpenAI GPT3 model name to use.
        output_filename (str): The filename to write the final outputs to. If not
            specified, will automatically name the file according to datetime.
        save_every_n (int): How often to write a backup file for the inferred data.
            Data will automatically be saved every time a rate limit error
            occurs.
        halt_on_error (bool): Whether to halt the inference on an exception
            which is NOT a RateLimitError. If False, will not halt; if true,
            will halt.

    Returns:
        None
    """
    print(f"Loaded {len(data_inference)} samples for inference.")
    print(f"Using {model} for prediction")

    gpt3_predictions = []
    jsonl_data = []
    for d in tqdm.tqdm(data_inference, desc="Texts processed"):
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        dois_skipped = []
        entry_json = create_sentences_json_for_inference(d)

        sentences_json = entry_json["doping_sentences"]
        for s_json in sentences_json:
            text = s_json["sentence_text"]
            cems = s_json["sentence_cems"]

            if sentence_is_paradigm(text, cems):
                s_json["relevant"] = True
                prompt = llm_prompt_from_sentence_json(
                    s_json,
                    include_relevance_hint=False,
                    include_question=True
                )

                has_response = False
                while not has_response:
                    try:
                        response = openai.Completion.create(
                            model=model,
                            prompt=prompt,
                            max_tokens=512,
                            n=1,
                            # top_p=1,
                            temperature=0,
                            stop=[STOP_TOKEN],
                            logprobs=5
                        ).choices[0]
                        has_response = True
                    except RateLimitError:
                        warnings.warn("Ran into rate limit error, sleeping for 60 seconds and dumping midstream...")
                        dumpfn(gpt3_predictions, os.path.join(DATADIR, f"midstream_ratelimit_{dt}.json"))
                        time.sleep(60)
                        print("Resuming...")
                        continue
                    except BaseException as BE:
                        if halt_on_error:
                            raise BE
                        else:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            warnings.warn(f"Ran into external error: {BE}")
                            traceback.print_exception(exc_type, exc_value,
                                                      exc_traceback,
                                                      limit=2,
                                                      file=sys.stdout)
                            print("Resuming...")
                            break

                # Record predictions, or put None if Error not halted on
                s_json["llm_completion"] = response.text if has_response else None
                s_json["gpt3_logprobs_numbers"] = response.logprobs.token_logprobs if has_response else None
                s_json["gpt3_logprobs_tokens"] = response.logprobs.tokens if has_response else None

            else:
                prompt = None
                s_json["relevant"] = False
                s_json["llm_completion"] = None
                s_json["gpt3_logprobs_numbers"] = None
                s_json["gpt3_logprobs_tokens"] = None

            if prompt:
                jsonl_data.append({
                    "prompt": prompt,
                    "completion": s_json["llm_completion"],
                })

        gpt3_predictions.append(entry_json)
        if len(gpt3_predictions) % save_every_n == 0:
            print(f"Saving {len(gpt3_predictions)} docs midstream")
            dumpfn(gpt3_predictions, os.path.join(DATADIR, f"midstream_{dt}.json"))

    dumpfn(gpt3_predictions, output_filename)
    jsonl_filename = output_filename.replace(".json", ".jsonl")
    dump_jsonl(jsonl_data, jsonl_filename)
    print(f"Dumped {len(gpt3_predictions)} total to {output_filename} (and raw jsonl to {jsonl_filename}).")


def gpt3_decode(inferred_filename, output_filename, fmt="eng"):
    """
    Decode and coalesce GPT-3 completions to structured graphs.

    Simply adds an "entity_graph_raw" key to each sample using the
    "doping_sentences" as input.

    Args:
        inferred_filename (str): The filename holding the GPT-3 inferences, generated
            by gpt3_infer.
        output_filename (str): The filename to write structured graphs to.
        fmt (str): The format to use (eng or json).
    """
    inferred_samples = loadfn(inferred_filename)

    for abstract_json in tqdm.tqdm(inferred_samples):
        for sentence_json in abstract_json["doping_sentences"]:
            ents = decode_entities_from_llm_completion(sentence_json["llm_completion"], fmt=fmt)
            sentence_json["entity_graph_raw"] = ents

    n_decoded = len(inferred_samples)
    dumpfn(inferred_samples, output_filename)

    print(f"Decoded {n_decoded} samples to file {output_filename}")
    return output_filename


if __name__ == "__main__":

    p = argparse.ArgumentParser(fromfile_prefix_chars='@')

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    p.add_argument(
        'op_type',
        help='Specify either "train" or "predict".',
        choices=["train", "predict"]
    )

    p.add_argument(
        "--openai_api_key",
        help="Your OpenAI API key. If not specified, will look for an environment variable OPENAI_API_KEY.",
    )

    p.add_argument(
        '--schema_type',
        help='The type of NERRE schema; choose between eng, engextra, and json. Default is eng. Only used if op_type is train.',
        default="eng",
        choices=["eng", "engextra", "json"]
    )

    p.add_argument(
        '--training_json',
        default=os.path.join(DATADIR, "train.json"),
        help='If training, specify the name of the training JSON file. Should NOT be a JSONL, as an OpenAI-compatible JSONL will be automatically created.',
    )

    p.add_argument(
        '--training_jsonl_output',
        help='If training, specify the name for the OpenAI-compatible JSONL to be written. Default is automatically written to a timestamped file in the data directory.'
    )

    p.add_argument(
        '--training_n_epochs',
        help="The number of epochs to train for; this arg is passed to openai cli. For more resolution in training, cancel the training operation and call the openai API directly."
    )

    p.add_argument(
        '--inference_model_name',
        help="Name of the trained model to use if op_type is 'predict'",
    )

    p.add_argument(
        '--inference_json',
        default=os.path.join(DATADIR, "test.json"),
        help='If predicting, specify the name of the raw inferred JSON file. This file will contain the raw sentence strings returned by GPT. Should NOT be a JSONL, as JSONL will automatically be saved as well. Default is the test set used in the publication, but this can be used to predict on many thousands of samples.'
    )

    p.add_argument(
        '--inference_json_raw_output',
        help="If predicting, specify the name for the JSONL file you would like to save the raw predictions to. Default is automatically written to a timestamped file in the data directory."
    )

    p.add_argument(
        '--inference_json_final_output',
        help="If predicting, specify the name for the decoded (non raw, structured) entries to be saved to. Default is automatically written to a timestamped file in the data directory."
    )

    p.add_argument(
        '--inference_halt_on_error',
        default=True,
        choices=[True, False],
        help="If predicting, specify whether to halt on an error. Default is True.",
    )

    p.add_argument(
        '--inference_save_every_n',
        default=100,
        help="If predicting, specify how often to save the raw predictions to the JSONL file in case of a midstream interruption. Default is 100.",
    )

    args = p.parse_args()

    op_type = args.op_type
    api_key = args.openai_api_key
    schema_type = args.schema_type
    training_json = args.training_json
    training_jsonl_output = args.training_jsonl_output
    training_n_epochs = args.training_n_epochs
    inference_model_name = args.inference_model_name
    inference_json = args.inference_json
    inference_json_raw_output = args.inference_json_raw_output
    inference_json_final_output = args.inference_json_final_output
    inference_halt_on_error = args.inference_halt_on_error
    inference_save_every_n = args.inference_save_every_n

    print(f"Doing '{op_type}' operation with schema type '{schema_type}'.")

    print(f"Using training json of {training_json}, saving formatted output file to {training_jsonl_output}.")

    if not training_jsonl_output:
        training_jsonl_output = os.path.join(DATADIR, f"training_{schema_type}_{dt}.jsonl")
        print(f"Training JSONL file will be saved to {training_jsonl_output}")

    if not inference_json_raw_output:
        inference_json_raw_output = os.path.join(DATADIR, f"inference_raw_{schema_type}_{dt}.json")
        print(f"Inference JSONL file will be saved to {inference_json_raw_output}")

    if not inference_json_final_output:
        inference_json_final_output = os.path.join(DATADIR, f"inference_decoded_{schema_type}_{dt}.json")

    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


    st = schema_type.lower()
    if st == "eng":
        fmt = "eng"
        write_extras = False
    elif st == "engextra":
        fmt = "eng"
        write_extras = True
    elif st == "json":
        fmt = "json"
        write_extras = False
    else:
        raise ValueError(
            f"Unknown schema type: {st}. Choose from 'json', 'eng', or 'engextra'.")

    if op_type == "train":
        data_training = loadfn(training_json)
        gpt3_finetune(
            data_training=data_training,
            training_filename=training_jsonl_output,
            fmt=fmt,
            write_extras=write_extras,
            n_epochs=training_n_epochs

        )
    elif op_type == "predict":
        data_infer = loadfn(inference_json)
        data_infer = [{k: d[k] for k in ("title", "text", "doi")} for d in data_infer]

        if not inference_model_name:
            raise ValueError("No inference_model_name specified!")

        gpt3_infer(
            data_inference=data_infer,
            output_filename=inference_json_raw_output,
            model=inference_model_name,
            save_every_n=inference_save_every_n,
            halt_on_error=inference_halt_on_error
        )

        gpt3_decode(
            inferred_filename=inference_json_raw_output,
            output_filename=inference_json_final_output,
            fmt=fmt
        )
    else:
        raise ValueError(f"Op type {op_type} unknown; choose from train or predict.")
