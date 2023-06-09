from utils import load_annotation

import json
import os
import numpy as np
import jellyfish
import copy
import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
import pprint
import random 


def read_jsonl(filename, num_samples=None):
    with open(filename, "r") as f:
        lines = f.readlines()
        samples = []
        for l in lines:
            samples.append(json.loads(l))
        if num_samples is not None:
            return random.sample(samples, num_samples)
        return samples

def ent_str_to_words(ent):
    stripped =  [e.strip() for e in ent.split(" ")]
    return [e for e in stripped if e]


def ent_json_to_word_basis_sets(ent_json, return_empty=False):
    """
    Where ent_json is multiple entries in a list

    Return all entities and links in a set-based word basis
    """
    # Must account for these in a weird way because the entries are not ordered :(
    to_account = {e: set() for e in ENTS_FROZEN + ENTS_LINKS_FROZEN}

    if return_empty:
        return to_account

    for entry in ent_json:
        formula_words = []
        for etype, ent_strs in entry.items():
            if isinstance(ent_strs, str):
                for w in ent_str_to_words(ent_strs):
                    to_account[etype].add(w)
                    formula_words.append(w)
            elif isinstance(ent_strs, list):
                for ent_str in ent_strs:
                    for w in ent_str_to_words(ent_str):
                        to_account[etype].add(w)
            else:
                raise ValueError(f"Ent strings was a weird type: {type(ent_strs)}, {ent_strs}")

        # Add links
        if formula_words:
            for e in ENTS_FROZEN_NOFORMULA:
                ent_strs = entry[e]
                words = []
                if isinstance(ent_strs, str):
                    words = ent_str_to_words(ent_strs)
                elif isinstance(ent_strs, list):
                    for ent_str in ent_strs:
                        words += ent_str_to_words(ent_str)
                else:
                    raise ValueError(f"Ent strings was a weird type: {type(ent_strs)}, {ent_strs}")


                if words:
                    for w in words:
                        for fw in formula_words:
                            to_account[f"{ROOT}{LINK_DELIMITER}{e}"].add(f"{fw}{LINK_DELIMITER}{w}")
    return to_account


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="general_results")
    # must be general or mof
    parser.add_argument("--task", type=str, default="general", choices=["general", "mof"], help="Which schema is being used")
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    TASK = args.task


    all_results = []
    all_winkler_similarities = []
    all_exact_match_accuracy = []

    if TASK == "mof":
        ENTS_FROZEN =  ['name_of_mof', 'mof_formula', 'mof_description', 'guest_species', 'applications']
        ENTS_FROZEN_NOFORMULA = [e for e in ENTS_FROZEN if e != "name_of_mof"]
    elif TASK == "general":
        ENTS_FROZEN = ["acronym", "applications", "name", "formula", "structure_or_phase", "description"]
        ENTS_FROZEN_NOFORMULA = [e for e in ENTS_FROZEN if e != "formula"]
    # ENTS_FROZEN_NOFORMULA = [e for e in ENTS_FROZEN if e != "mof_formula"]
    LINK_DELIMITER = "|||"
    if TASK == "mof":
        ROOT = "name_of_mof"
    elif TASK == "general":
        ROOT = "formula"
    ENTS_LINKS_FROZEN = [f"{ROOT}{LINK_DELIMITER}{e}" for e in ENTS_FROZEN_NOFORMULA]


    for fn in os.listdir(RESULTS_DIR):
        run = read_jsonl(os.path.join(RESULTS_DIR, fn))
        exact_matches = 0
        unparsable = 0
        total = 0
        jaro_winkler_similarities = []


        ent_scores_test = {e: [] for e in ENTS_FROZEN}
        ent_scores_gold = {e: [] for e in ENTS_FROZEN}


        subdict = {"test_correct_triplets": 0, "test_retrieved_triplets": 0, "gold_retrieved_triplets": 0}
        links_scores = {el: copy.deepcopy(subdict) for el in ENTS_LINKS_FROZEN}


        for sample in tqdm.tqdm(run):
            gold_string = sample["completion"].replace("\n\nEND\n\n", "").strip()
            test_string = sample["gpt3_completion"].replace("\n\nEND\n\n", "").replace('\\', '').strip()

            # print(f"Gold string is {gold_string}")
            # print(f"Test string is {test_string}")

            gold_json = json.loads(gold_string)
            prompt = sample["prompt"].replace("\n\n###\n\n", "").strip()
            n_prompt_words = len([w for w in prompt.split(" ") if w])

            total += 1
            if gold_string == test_string:
                exact_matches += 1
            test_json = {}
            try:
                test_json = sample["gpt3_completion"]
                if isinstance(test_json, str):
                    try :
                        test_json = json.loads(test_json)
                    except json.decoder.JSONDecodeError as jse:
                        test_json = []
                for d in test_json:
                    for key in ENTS_FROZEN:
                        if key not in d:
                            if key in ["formula", "name", "acronym"]:
                                d[key] = ""
                            else:
                                d[key] = [""]
            except json.decoder.JSONDecodeError as jse:
                unparsable += 1

            jws = jellyfish.jaro_winkler_similarity(gold_string, test_string, long_tolerance=True)
            jaro_winkler_similarities.append(jws)

            gold_accounting = ent_json_to_word_basis_sets(gold_json)

            if test_json:
                test_accounting = ent_json_to_word_basis_sets(test_json)
            else:
                test_accounting = ent_json_to_word_basis_sets({}, return_empty=True)

            for etype in ENTS_FROZEN:
                ent_accounting_copy = copy.deepcopy(test_accounting[etype])
                n_unlabelled_words = copy.deepcopy(n_prompt_words)
                for ew in gold_accounting[etype]:

                    # Account for true positives
                    if ew in test_accounting[etype]:
                        ent_scores_test[etype].append(1)
                        ent_scores_gold[etype].append(1)
                        ent_accounting_copy.remove(ew)
                        n_unlabelled_words -= 1
                    # account for false negatives
                    else:
                        ent_scores_test[etype].append(0)
                        ent_scores_gold[etype].append(1)
                        n_unlabelled_words-= 1

                # Among the remaining test accounting words, only false positives
                # should remain in the set
                for ew in ent_accounting_copy:
                    ent_scores_test[etype].append(1)
                    ent_scores_gold[etype].append(0)
                    n_unlabelled_words -= 1

                # the only labels remaining are true negatives
                ent_scores_test[etype] += [0] * n_unlabelled_words
                ent_scores_gold[etype] += [0] * n_unlabelled_words


            for elinktype in ENTS_LINKS_FROZEN:
                gold_triples = gold_accounting[elinktype]
                test_triples = test_accounting[elinktype]

                n_correct_triples = len([e for e in test_triples if e in gold_triples])
                links_scores[elinktype]["test_correct_triplets"] += n_correct_triples
                links_scores[elinktype]["test_retrieved_triplets"] += len(test_triples)
                links_scores[elinktype]["gold_retrieved_triplets"] += len(gold_triples)

        results = {"ents": {}, "links": {}}
        for etype in ENTS_FROZEN:
            gold_arr = ent_scores_gold[etype]
            test_arr = ent_scores_test[etype]

            subdict = {"recall": 0, "precision": 0, "f1": 0}
            subdict["recall"] = recall_score(gold_arr, test_arr)
            subdict["precision"] = precision_score(gold_arr, test_arr)
            subdict["f1"] = f1_score(gold_arr, test_arr)
            results["ents"][etype] = subdict


        for elinktype in ENTS_LINKS_FROZEN:
            subdict = {"precision": 0, "recall": 0, "f1": 0}
            n_correct = links_scores[elinktype]["test_correct_triplets"]
            n_retrieved = links_scores[elinktype]["test_retrieved_triplets"]
            n_gold_retrieved = links_scores[elinktype]["gold_retrieved_triplets"]
            subdict["precision"] = n_correct/n_retrieved
            subdict["recall"] = n_correct/n_gold_retrieved
            subdict["f1"] = 2 * (subdict["precision"] * subdict["recall"])/(subdict["precision"] + subdict["recall"])
            results["links"][elinktype] = subdict

        all_exact_match_accuracy.append(exact_matches/total)
        all_winkler_similarities.append(np.mean(jaro_winkler_similarities))
        all_results.append(results)


    print("All Exact match accuracy average:", np.mean(all_exact_match_accuracy))
    print("Jaro-Winkler avg similarity:", np.mean(all_winkler_similarities))


    outer_keys = ("links", "ents")
    inner_keys = ("recall", "precision", "f1")

    r_dict_avg = copy.deepcopy(all_results[0])
    for k, v in r_dict_avg.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                r_dict_avg[k][k2][k3] = None


    for ok in outer_keys:
        if ok == "links":
            mid_keys = ENTS_LINKS_FROZEN
        else:
            mid_keys = ENTS_FROZEN
        for mk in mid_keys:
            for ik in inner_keys:
                arr2avg = []
                for rd in all_results:
                    arr2avg.append(rd[ok][mk][ik])
                r_dict_avg[ok][mk][ik] = np.mean(arr2avg)

        pprint.pprint(r_dict_avg)