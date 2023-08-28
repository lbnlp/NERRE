import json
import os
import numpy as np
import jellyfish
import copy
import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
import pprint

from utils import read_jsonl


def check_equivalence_of_entries(gold_entry, test_entry):
    ## Entries are a list of dictionaries
    ## We first order each list, then each dictionary, then compare strings


    ### order list by formula key
    gold_entry = sorted(gold_entry, key=lambda x: x.get('formula', ''))
    test_entry = sorted(test_entry, key=lambda x: x.get('formula', ''))

    ### order each dictionary by keys
    gold_entry = [dict(sorted(d.items())) for d in gold_entry]
    test_entry = [dict(sorted(d.items())) for d in test_entry]

    ### compare strings
    return str(gold_entry) == str(test_entry)



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
        root_accounting = {root: set() for root in ROOT}
        for etype in ENTS_FROZEN:
            ent_strs = entry[etype]
            if isinstance(ent_strs, str):
                for w in ent_str_to_words(ent_strs):
                    to_account[etype].add(w)
                if etype in ROOT and ent_strs:
                    # Formulae/roots must be counted as single words
                    root_accounting[etype].add(ent_strs)
                    # root_accounting[etype] = root_accounting[etype].union(set(ent_str_to_words(ent_strs)))
            elif isinstance(ent_strs, list):
                for ent_str in ent_strs:
                    for w in ent_str_to_words(ent_str):
                        to_account[etype].add(w)
            else:
                raise ValueError(f"Ent strings was a weird type: {type(ent_strs)}, {ent_strs}")

        # Add links
        for root, accounting in root_accounting.items():
            if accounting:
                for e in ENTS_FROZEN_NOROOT:
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
                        for f in accounting:
                            for w in words:
                                # avoid self-links
                                if f != w:
                                    to_account[f"{root}{LINK_DELIMITER}{e}"].add(f"{f}{LINK_DELIMITER}{w}")
    return to_account


if __name__ == "__main__":
    printmode=False
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
    all_unparsable = []

    if TASK == "mof":
        ENTS_FROZEN = ['name_of_mof', 'mof_formula', 'mof_description', 'guest_species', 'applications']
    elif TASK == "general":
        ENTS_FROZEN = ["acronym", "applications", "name", "formula", "structure_or_phase", "description"]
        # ENTS_FROZEN = ["applications", "name", "formula", "structure_or_phase", "description"]
    LINK_DELIMITER = "|||"
    if TASK == "mof":
        ROOT = ("name_of_mof",)
    elif TASK == "general":
        # ROOT = "formula"
        ROOT = ("formula",)
    else:
        raise ValueError(f"There is no task '{TASK}'")
        
    ENTS_FROZEN_NOROOT = [e for e in ENTS_FROZEN if e not in ROOT]
    ENTS_LINKS_FROZEN = [f"{root}{LINK_DELIMITER}{e}" for e in ENTS_FROZEN_NOROOT for root in ROOT]


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

            if printmode:
                print(f"Gold string is {gold_string}")
                print(f"Test string is {test_string}")

                print(gold_string)

            gold_json = json.loads(gold_string)
            prompt = sample["prompt"].replace("\n\n###\n\n", "").strip()
            n_prompt_words = len([w for w in prompt.split(" ") if w])

            total += 1
            # if gold_string == test_string:
            #     exact_matches += 1
            test_json = {}
            try:
                test_json = sample["gpt3_completion"]
                if isinstance(test_json, str):
                    try:
                        test_json = json.loads(test_json)
                    except json.decoder.JSONDecodeError as jse:
                        test_json = []
                for d in test_json:
                    for key in ENTS_FROZEN:
                        if key not in d:
                            if key in ["formula", "name", "acronym", "mof_formula", "name_of_mof"]:
                                d[key] = ""
                            else:
                                d[key] = [""]

                    # remove extra keys as they are "parsable" but invalid
                    extra_keys = []
                    for key in d:
                        if key not in ENTS_FROZEN:
                            extra_keys.append(key)
                    for key in extra_keys:
                        d.pop(key)

            except json.decoder.JSONDecodeError as jse:
                unparsable += 1

            if check_equivalence_of_entries(gold_json, test_json):
                exact_matches += 1

            jws = jellyfish.jaro_winkler_similarity(gold_string, test_string, long_tolerance=True)
            jaro_winkler_similarities.append(jws)

            gold_accounting = ent_json_to_word_basis_sets(gold_json)

            if test_json:
                test_accounting = ent_json_to_word_basis_sets(test_json)
            else:
                test_accounting = ent_json_to_word_basis_sets({}, return_empty=True)

            if printmode:
                print("Gold entry is", pprint.pformat(gold_json))
                print("Test accounting is", pprint.pformat(test_accounting))
                print("Gold accounting is", pprint.pformat(gold_accounting))

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
                        n_unlabelled_words -= 1

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
            subdict = {}#"precision": 0, "recall": 0, "f1": 0}
            n_correct = links_scores[elinktype]["test_correct_triplets"]
            n_retrieved = links_scores[elinktype]["test_retrieved_triplets"]
            n_gold_retrieved = links_scores[elinktype]["gold_retrieved_triplets"]

            if printmode:
                print(elinktype)
            # subdict["precision"] = n_correct/n_retrieved if n_retrieved > 0 else 0

            # # n_gold_retrieved can be zero in some folds
            # subdict["recall"] = n_correct/n_gold_retrieved if n_gold_retrieved > 0 else 1
            # try:
            #     subdict["f1"] = 2 * (subdict["precision"] * subdict["recall"])/(subdict["precision"] + subdict["recall"])
            # except ZeroDivisionError:
            #     subdict["f1"] = 0
            # results["links"][elinktype] = subdict
            try:
                subdict["precision"] = n_correct/n_retrieved
                subdict["recall"] = n_correct/n_gold_retrieved
            except ZeroDivisionError: # if n_retrieved or n_gold_retrieved is zero, do not append this fold
                results["links"][elinktype] = {}#subdict # {}
                continue
            if n_correct==0: # equivalent to subdict["precision"]==0 & subdict["recall"]==0
                subdict["f1"] = 0 # not actually defined but at least this is strict
            else:
                subdict["f1"] = 2 * (subdict["precision"] * subdict["recall"])/(subdict["precision"] + subdict["recall"])
            results["links"][elinktype] = subdict

        all_exact_match_accuracy.append(exact_matches/total)
        all_winkler_similarities.append(np.mean(jaro_winkler_similarities))
        all_unparsable.append(unparsable/total)
        all_results.append(results)


    print("All Exact match accuracy average:", np.mean(all_exact_match_accuracy))
    print("Jaro-Winkler avg similarity:", np.mean(all_winkler_similarities))
    print("Parsable percentage", 1-np.mean(all_unparsable))


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
        for mk in mid_keys: # elink
            for ik in inner_keys: # recall/precision/f1
                arr2avg = []
                foldidx = -1
                for rd in all_results:# fold 
                    foldidx+=1
                    if rd[ok][mk]=={}: # pass for this fold
                        print("skipped", ok,mk,ik,"for fold",foldidx)
                        continue
                    else:
                        arr2avg.append(rd[ok][mk][ik])
                r_dict_avg[ok][mk][ik] = np.mean(arr2avg) #average over folds

    pprint.pprint(r_dict_avg)