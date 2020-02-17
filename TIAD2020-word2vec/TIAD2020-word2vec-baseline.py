#!/usr/bin/env python3
"""
Authors: Besim Kabashi; Michael Ruppert
"""

import os
from random import shuffle
from unicodedata import normalize
import json

from gensim.models import Word2Vec


def norm(w): return normalize('NFKD', w)

"""
The program needs a folder named rdf with https://tiad2020.unizar.es/data/TranslationSetsApertiumRDF.zip unpacked
The file trans_OC_CA.tsv must be renamed to trans_OC-CA.tsv
"""

files = ["rdf/" + e for e in os.listdir("rdf/")]


"""
This part reads in the .tsv files and generates the list of graph transitions.
Additionally it creates dictionaries with the available target and source words in a language to enable the dictionary creation
"""

trainlist = []
wordlists = dict()
for file in files:
    print(file)
    lang = file.split("_")[-1].strip(".tsv").split("-")[0]
    lang2 = file.split("_")[-1].strip(".tsv").split("-")[1]
    if lang not in wordlists:
        wordlists[lang] = set()
    l1d = wordlists[lang]
    if lang2 not in wordlists:
        wordlists[lang2] = set()
    l2d = wordlists[lang2]
    for i, line in enumerate(open(file)):
        line = norm(line)
        if i:
            spltd = line.strip().replace("\"", "").split("\t")
            pos = spltd[-1].split("#")[-1]
            w1 = spltd[0]
            w2 = spltd[-2]
            l1d.add((w1, pos))
            l2d.add((w2, pos))
            trainlist.append([lang + "_" + pos + "_" + w1, lang2 + "_" + pos + "_" + w2])

"""
This part shuffles the trainlist and then trains a Word2Vec model with gensim.
"""

shuffle(trainlist)


"""
This list determines which new language pairs are generated.
"""

pairs = [("EN", "PT"), ("PT", "EN"), ("PT", "FR"), ("FR", "PT"), ("EN", "FR"), ("FR", "EN")]


"""
This Loop loops over the translation to generate
"""

for SIZE in [50, 100, 300, 30, 80, 150, 600]: #Change Params here
    model = Word2Vec(trainlist, iter=700, size=SIZE, min_count=2, window=3, workers=16).wv
    # model.save_word2vec_format("transl_vectors.vec")
    for cv1, cv2 in [(0.6, 0.8), (0.7, 0.9), (0.3, 0.6), (0.2, 0.4), (0.2, 0.7), (0.3, 0.9)]: #Change Limit Combinations here
        for SOURCE_LANG, TARGET_LANG in pairs:
            """
            The generation of results now loops over all words in the source language
            and then retrieves the top nearest words in the w2v model
            and filters out the words of the target words with the same pos tag

            the top word is added then the similarity is greater then .6
            all next words with similarity greater then .8 are then added
            """
            results = []
            for word, pos in wordlists[SOURCE_LANG]:
                if SOURCE_LANG + "_" + pos + "_" + word not in model:
                    continue
                res = [(i, j) for i, j in model.most_similar(SOURCE_LANG + "_" + pos + "_" + word, topn=1000) if
                       i.startswith(TARGET_LANG + "_" + pos)][:4]
                for cnt, rs in enumerate(res):
                    re = rs[0].split("_")[-1]
                    if not cnt:
                        if rs[1] ** 2 > cv1:  # modified cutvalues
                            results.append("\t".join((word.replace(" ", "_"), re, pos, "{:0.2f}".format(rs[1] ** 2))))
                    else:
                        if rs[1] ** 2 > cv2:
                            results.append("\t".join((word.replace(" ", "_"), re, pos, "{:0.2f}".format(rs[1] ** 2))))

            """
            The results generated are then written to tsv files in the target format
            """
            with open("base_trans_" + SOURCE_LANG.lower() + "-" + TARGET_LANG.lower() + ".tsv", "w") as f:
                lwa = "\t".join(["source written representation",
                                 "target written representation",
                                 "part of speech",
                                 "confidence score"
                                 ]) + "\n"
                f.write("\n".join(results))
            files = [f for f in os.listdir() if f.startswith("TIAD2019") and f.endswith(".csv")]
            targetfiles = [f for f in os.listdir() if f.startswith("base_trans") and f.endswith(".tsv")]
            # LIMIT = 0

            """
            Script tests all evaluation data in the directory and compares the entries with the inferred translations in the same folder

            """

            for LIMIT in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                Pc, Rc, Fc = [], [], []
                for file in files:
                    pair = file.split("_")[-1].split(".")[0]
                    tfilename = "base_trans_" + pair + ".tsv"
                    if tfilename in targetfiles:
                        with open(file) as eval, open(tfilename) as computed:
                            # print("Eval of ",file, tfilename)
                            evaldata = set([tuple([v for v in line.strip().split(",") if len(v)]) for line in eval])
                            evaldata = {e for e in evaldata if len(e)==3}
                            evalline = set([tuple(line.strip().split("\t")[:3]) for line in computed if
                                            float(line.strip().split("\t")[3]) > LIMIT])
                            
                            evalw = {w for w, _, _ in
                                     evaldata}  # Filter out the trranslations for which the source entry is not present in the golden standard
                            evalline = {(w1, w2, w3) for w1, w2, w3 in evalline if
                                        w1 in evalw}  # Filter out the trranslations for which the source entry is not present in the golden standard
                            evaldata = {(w1, w2) for w1, w2, _ in
                                        evaldata}  # Simplification, cause Golden Data have other Names for POS-Tags
                            evalline = {(w1, w2) for w1, w2, _ in evalline}
                            # Prints percentage of tuples found in computed data
                            P = len(evaldata & evalline) / len(evalline)
                            R = len(evaldata & evalline) / len(evaldata)
                            print("PrecisionS", P *100, "%")
                            print("RecallS", R*100, "%")
                            print("F-Measure", 100*(2*P*R)/(P+R),"%")
                            Pc.append(P)
                            Rc.append(R)
                            Fc.append((2 * P * R) / (P + R))
                # PARAMS
                print("=" * 20)
                print("PARAMS:", SIZE, cv1, cv2, LIMIT)
                print("Precision:", 100 * sum(Pc) / len(Pc))
                print("Recall:", 100 * sum(Rc) / len(Rc))
                print("F-Measure:", 100 * sum(Fc) / len(Fc))
                resultsave.append(
                    (100 * sum(Fc) / len(Fc), 100 * sum(Rc) / len(Rc), 100 * sum(Pc) / len(Pc), SIZE, cv1, cv2, LIMIT))
print(resultsave)


"""
The results generated are then written to files in the target format
"""
with open("results.json","w") as f:
    json.dump(resultsave,f)


print("\n".join(["F-Measure {}, Precision {}, Recall {}, Size {}, CV1 {}, CV2 {}, LIMIT {}".format(i1,i3,i2,i4,i5,i6,i7) for i1, i2,i3,i4,i5,i6,i7 in sorted(resultsave)[::-1]]))

