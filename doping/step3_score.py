import os

import pandas as pd
from monty.serialization import loadfn
import pprint
import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from constants import DATADIR

EVALUATE_MODIFIERS_AND_RESULTS = False

def evaluate(gold, test, loud=False):
    if EVALUATE_MODIFIERS_AND_RESULTS:
        ent_categories = ["basemats", "dopants", "results", "doping_modifiers"]
    else:
        ent_categories = ["basemats", "dopants"]

    scores = {
        k: {k2: 0 for k2 in ["tp", "tn", "fp", "fn"]} for k in ent_categories
    }

    scores["dopants2basemats"] = {"n_correct": 0, "test_retrieved": 0, "gold_retrieved": 0}

    for i, val_entry in enumerate(gold):
        for j, s in enumerate(val_entry["doping_sentences"]):
            if not s["relevant"]:
                break

            test_entry_tot = test[i]["doping_sentences"][j]["entity_graph_raw"]
            test_entry = {k: test_entry_tot[k] for k in ent_categories + ["dopants2basemats"]}
            gold_entry = {k: s[k] for k in ent_categories + ["dopants2basemats"]}

            if test_entry["dopants2basemats"] == []:
                test_entry["dopants2basemats"] = {}

            sentence_text = s["sentence_text"]
            if loud:
                print(s["sentence_text"])
                pprint.pprint(gold_entry)
                pprint.pprint(test_entry)

            for ent_type in ent_categories:

                # correcting a relic of a previous annotation scheme
                if ent_type == "doping_modifiers":
                    gold_ents_words = " ".join(gold_entry[ent_type]).split(" ")
                else:
                    gold_ents_words = " ".join(list(gold_entry[ent_type].values())).split(" ")

                if loud:
                    print(ent_type, test_entry)
                test_ents_words = " ".join(list(test_entry[ent_type].values())).split(" ")

                gold_ents_words = [w for w in gold_ents_words if w]
                test_ents_words = [w for w in test_ents_words if w]

                # print(f"GOLD: {gold_ents_words}")
                # print(f"TEST: {test_ents_words}")

                TP = 0
                TN = 0
                FP = 0
                FN = 0
                for w in gold_ents_words:
                    if w in test_ents_words:
                        TP += 1
                    else:
                        if loud:
                            print(f"FALSE NEGATIVE: {w}")
                        FN += 1

                for w in test_ents_words:
                    if w not in gold_ents_words:
                        if loud:
                            print(f"FALSE POSITIVE: {w}")
                        FP += 1

                TN = len(sentence_text.split(" ")) - TP - FN - FP

                scores[ent_type]["tp"] += TP
                scores[ent_type]["tn"] += TN
                scores[ent_type]["fp"] += FP
                scores[ent_type]["fn"] += FN


            gold_entry["triplets"] = []
            test_entry["triplets"] = []

            # assemble triplets
            for rel_entry in (gold_entry, test_entry):
                for did, bids in rel_entry["dopants2basemats"].items():
                    for bid in bids:
                        bmat_words = rel_entry["basemats"][bid]
                        dop_words = rel_entry["dopants"][did]
                        for bmat_word in bmat_words.split(" "):
                            for dop_word in dop_words.split(" "):
                                if bmat_word and dop_word:
                                    rel_entry["triplets"].append(f"{bmat_word} {dop_word}")

            gold_triplets = gold_entry["triplets"]
            test_triplets = test_entry["triplets"]

            n_correct_triplets = 0
            for triplet in gold_triplets:
                if triplet in test_triplets:
                    n_correct_triplets += 1

            scores["dopants2basemats"]["n_correct"] += n_correct_triplets
            scores["dopants2basemats"]["test_retrieved"] += len(test_triplets)
            scores["dopants2basemats"]["gold_retrieved"] += len(gold_triplets)

            # Precision = Number of correct triples/Number of triples retrieved
            # Recall = Number of correct triples/Number of correct triples that exist in Gold set.
            # F-Measure = Harmonic mean of Precision and Recall.
            if loud:
                print("-"*50 + "\n")

    if loud:
        pprint.pprint(scores)

    scores_computed = {k: {} for k in ent_categories}

    for k in ent_categories:
        tp = scores[k]["tp"]
        tn = scores[k]["tn"]
        fp = scores[k]["fp"]
        fn = scores[k]["fn"]

        if tp + fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = tp / (tp + 0.5 * (fp + fn))

        print(f"{k}: prec={prec}, recall={recall}, f1={f1}")
        scores_computed[k]["precision"] = prec
        scores_computed[k]["recall"] = recall
        scores_computed[k]["f1"] = f1


    triplet_scores = scores["dopants2basemats"]
    if triplet_scores["test_retrieved"] == 0:
        triplet_prec = 0
    else:
        triplet_prec = triplet_scores["n_correct"]/triplet_scores["test_retrieved"]
    triplet_recall = triplet_scores["n_correct"]/triplet_scores["gold_retrieved"]

    if triplet_recall == 0 or triplet_prec == 0:
        triplet_f1 = 0
    else:
        triplet_f1 = (2 * triplet_prec * triplet_recall)/(triplet_prec + triplet_recall)
    print(f"triplets: prec={triplet_prec}, recall={triplet_recall}, f1={triplet_f1}")
    scores_computed["link triplets"] = {"precision": triplet_prec, "recall": triplet_recall, "f1": triplet_f1}

    return scores_computed, ent_categories


if __name__ == "__main__":

    p = argparse.ArgumentParser(fromfile_prefix_chars='@')

    p.add_argument(
        '--test_file',
        help="The test file with correct answers. ",
        default=os.path.join(DATADIR, "test.json")
    )

    p.add_argument(
        "--pred_file",
        help="The file predicted by LLM-NERRE. Default is using the data from publication using the ENG schema.",
        default=os.path.join(DATADIR, "inference_decoded_eng.json"),
        required=False
    )

    p.add_argument(
        "--plot",
        action='store_true',
        help="If flag is present, show a simple bar chart of results.",
        required=False
    )

    p.add_argument(
        "--loud",
        action='store_true',
        help="If true, show a summary of each evaluated sentence w/ FP and FNs.",
        required=False
    )

    args = p.parse_args()
    gold = loadfn(args.test_file)
    test = loadfn(args.pred_file)
    plot = args.plot
    loud = args.loud

    print(f"Scoring outputs using \n\ttest file: {args.test_file}\n\tpred file: {args.pred_file}")

    scores_computed, ent_categories = evaluate(gold, test, loud=loud)
    # FOR PLOTTING ONLY

    ents_rows = []
    for entc in ent_categories:
        ents_rows += [entc] * 3

    df = pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1"] * (len(ent_categories) + 1),
            "entity": ents_rows + ["link triplets"] * 3,
         }
    )

    scores_df = []
    for i, r in df.iterrows():
        scores_df.append(scores_computed[r["entity"]][r["metric"]])
    df["score"] = scores_df

    print(df)

    if plot:
        ax = sns.barplot(x="entity", y="score", hue="metric", data=df)
        for container in ax.containers:
            ax.bar_label(container)
        plt.show()













