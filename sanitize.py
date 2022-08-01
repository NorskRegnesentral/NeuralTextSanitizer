# Test the following pipeline:
# Text, Target, posList -> blacklist
# blacklist + scores [Here I use random number as placeholders] -> decision for each entity
# Here decision 1 = Keep, decision 0 = Mask [Assumption used in generating constraints]

import torch
import torch.nn as nn
import time
import json
import numpy as np
from OptimizationAlgorithm.linearProgramming import linearOpt # Should include this first, otherwise will raise Segmentation Fault
from WebQueryModel.wq_model import MyModel5
from WebQueryModel.wq_emulate import simGoogle
from transformers import BertTokenizerFast, BertForMaskedLM
from MLMModel.mlmbert import mlmbert
from ERModel.bert_model import NERModel
from transformers import RobertaTokenizerFast
from ERModel.detect import detect_pii
from MaskClassifier.mask_classifier import MaskClassifier


if __name__ == "__main__":

    # Test for Step2 + Step3
    # with open("SampleData/sample.json","r") as f:
    #     tmp = json.load(f)
    # sequence = tmp["sequence"]
    # target = tmp["target"]
    # posList = tmp["posList"]
    # posList = [tuple(p) for p in posList] # Ensure posList List of tuple -> otherwise unhashable
    # sequence = "Bodewin Claus Eduard Keitel (German pronunciation: [ˈkaɪ̯tl̩]; 1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office."
    # target = "bodewin keitel"
    # posList = [(0, 27), (29, 35), (51, 61), (63, 67), (70, 74), (82, 88), (89, 96), (104, 116), (131, 135), (143, 164)]  # annotation text_span

    # Test Data for the whole pipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    ner_model = NERModel(model="roberta-base", num_labels=17)
    ner_model.load_state_dict(torch.load("SampleData/3ft_roberta_model.pt", map_location=device))

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')
    bertModelForMLM = BertForMaskedLM.from_pretrained('bert-large-cased')

    bertModel = nn.Sequential(bertModelForMLM.bert.embeddings, bertModelForMLM.bert.encoder)
    myModel = MyModel5(device, drop=True, concatenation=1).to(device)
    myModel.load_state_dict(torch.load('SampleData/(u,v)_annotator_based.pt', map_location=device))

    with open("SampleData/sample2.json","r") as f:
        tmp = json.load(f)
        final_decisions = []

    for document in tmp:
        sequence = document["text"]
        target = document["target"]
        print("Sanitization for target:", target,'\n')

        # Detect entities
        try:
            # If posList have manual annotations
            posList = document["annotations"]
            posList = [tuple(p) for p in posList]

            tagList = [i[1] for i in posList]
            posList = [tuple(i[0]) for i in posList]

            spans = {pos: tag for pos, tag in zip(posList, tagList)}

        except KeyError:
            # Privacy enhanced ER to detect all PII
            posList = detect_pii(document, ner_model, roberta_tokenizer) # List of {target: [Spans detected with labels]}
            posList = list(posList.values())[0] # get only the values

            tagList = [i[1] for i in posList]
            posList = [i[0] for i in posList] # text spans

            # posList = sorted(posList, key=lambda x: x[0])
            posList = list(set(posList))
            posList = sorted(posList, key=lambda x: x[0])

            spans = {pos: tag for pos, tag in zip(posList, tagList)}

        # Step 1: get blacklist
        start = time.time()
        # print("Loading Model for Blacklist 1")
        mlmbert_model = mlmbert(device, tokenizer, bertModelForMLM, thres = -4, N=1)
        # print(time.time() - start, "s")

        start = time.time()
        print("Getting Blacklist 1 [Language Model]")
        blacklist1, semantic_loss = mlmbert_model.get_blacklist_and_semantic_loss(sequence, posList)
        blacklist1 = [[(t,1) for t in pair] for pair in blacklist1]
        # print(blacklist1)
        # print(time.time() - start, "s")

        start = time.time()
        # print("Loading Model for Blacklist 2")
        # bertModel = BertModel.from_pretrained("bert-large-cased")
        # tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        # myModel = MyModel5(device, drop=True, pooling="max").to(device)
        # myModel.load_state_dict(torch.load('SampleData/model5max.pt', map_location=device))

        # print(time.time() - start, "s")

        start = time.time()
        print("Getting Blacklist 2 [Web Query Based Models]")
        sim = simGoogle(tokenizer, bertModel, myModel, device)
        sim.initialize(sequence, posList, target)
        res = sim.generateBlackList(2)
        blacklist2 = [[(t, 1) for t in pair] for pair in res]  # entity t can not choose option 1 [KEEP]
        sim.clear()
        # print(blacklist2)
        # print(time.time() - start, "s")

        start = time.time()
        # print("Loading Model for Blacklist 3")
        classifier = MaskClassifier.load("SampleData/mask_classifier.dill", device)
        # print(time.time() - start, "s")

        start = time.time()
        print("Getting Blacklist 3 [Mask Classifier]")
        res = classifier.predict_from_labelled_text(sequence, spans, device=device)
        blacklist3 = [[(i,1)] for i,pos in enumerate(posList) if res[pos]>0.5]
        # print(blacklist3)
        # print(time.time() - start, "s")

        blacklist = blacklist1 + blacklist2 + blacklist3

        # Step 2: get final decision for each entity
        start = time.time()
        print("Getting final decisions\n")
        n = len(semantic_loss)
        m = 2
        scores = np.array([[sem_loss,0] for sem_loss in semantic_loss]) # should input np.array
        optSolver = linearOpt(n, m, blacklist, scores)
        res = optSolver.solve()
        # print(res)
        # print(time.time()-start,"s")

        # Save final decisions
        decisions = {
            "opt_decision": [posList[i] for i,d in enumerate(res) if d==1],        # 1 for KEEP
            "ner_decision": posList,                                               # All the entities detected by NER
            "b1": list(set(posList[t[0]] for pair in blacklist1 for t in pair)),   # blacklist1 LM
            "b2": list(set(posList[t[0]] for pair in blacklist2 for t in pair)),   # blacklist2 Web Query Based Models
            "b3": list(set(posList[t[0]] for pair in blacklist3 for t in pair))    # blacklist3 2.3 mask_classifier
        }
        # sort the list
        for k,v in decisions.items():
            decisions[k] = [tuple(pair) for pair in v]
            decisions[k].sort()
        final_decisions.append({target:decisions})

    out_file = open("final_decision.json", "w")
    json.dump(final_decisions, out_file, ensure_ascii=False)

        # Some tests

        # Print Blacklist
        # print("Blacklist1")
        # for pair in blacklist1:
        #     print(" AND ".join(sequence[posList[p[0]][0]:posList[p[0]][1]] for p in pair))
        # print("Blacklist2")
        # for pair in blacklist2:
        #     print(" AND ".join(sequence[posList[p[0]][0]:posList[p[0]][1]] for p in pair))

        # Print Masking Decisions
        # entities = [sequence[pos[0]:pos[1]] for pos in posList]
        # print("Final Decision")
        # for entity, decision, sem_loss in zip(entities, res, semantic_loss):
        #     if decision == 1:
        #         print(entity, '\t', 'Keep', '\t', 0)
        #     else:
        #         print(entity, '\t', 'Mask', '\t', sem_loss)
