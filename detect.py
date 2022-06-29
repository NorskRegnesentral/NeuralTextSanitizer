# from typing_extensions import TypedDict
# import torch.nn.functional as F
# from typing import List,Any
# from transformers import BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizer, RobertaTokenizerFast, RobertaForTokenClassification, LongformerTokenizer, LongformerTokenizerFast
# from tokenizers import Encoding
# import itertools
# from torch import nn
# from dataclasses import dataclass
# from torch.utils.data import Dataset
# from transformers import PreTrainedTokenizerFast
# import json
# import torch

# from transformers import AdamW
# from transformers import get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report
# import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import collections
# import random
# import argparse
from data_handling import *
from torch.utils.data.dataloader import DataLoader
import string


def detect_pii(data, model, tokenizer):
    # data.keys() = 'text', 'target'

    # label_set.labels_to_id, ids_to_label (17 categories, BI)
    label_set = LabelSet(labels=['MISC', 'QUANTITY', 'CODE', 'ORG', 'PERSON', 'DEM', 'DATETIME', 'LOC'])

    # data_.label_set
    data_ = Dataset(data=data, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=512)
    dataloader = DataLoader(data_, collate_fn=TrainingBatch, batch_size=8, shuffle=True)

    model.eval()

    nums = {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 2, 9: 1, 10: 2, 11: 1, 12: 2, 13: 1, 14: 2, 15: 1, 16: 2} 

    predictions, offsets = [], [] # offsets [(target person, pos in original text) for each entity]
    for X in dataloader:
        with torch.no_grad():
            y_pred = model(X) # X of type TrainingBatch. We use X.input_ids and X.attention_masks
            y_pred = y_pred.permute(0,2,1)              # [N,512,17] -> [N,17,512]
            pred = y_pred.argmax(dim=1).cpu().numpy()   # [N,17,512] -> [N,512]
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred]) # [N,512], a category for each token


    out = []
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i])-count)] # Remove the padding, each i is a different batch
            predictions[i] = predictions[i][:len(offsets[i])] # Remove the padding, [CLS] ... [SEP] [PAD] [PAD]...

    l1 = [item for sublist in predictions for item in sublist] # Unravel predictions if it has multiple batches
    l3 = [data_.label_set.ids_to_label[i] for i in l1]         # Get Categories for each tokens
    l1 = [nums[i] for i in l1]                                 # B -> 1 I -> 2
    l2 = [item for sublist in offsets for item in sublist]     # Unravel subsets if it has multiple batches

    it = enumerate(l1+[0])
    sv = 0

    try:
        while True:
            if sv==1: # If an entity followed by another entity, fi, fv marks the beginning of an entity
                fi,fv = si,sv
            else:
                while True:
                    fi,fv = next(it)
                    if fv:
                        break
            while True: # Whenever it finds an 1, it tries to find the boundary for this entity (stops at 0 or 1)
                si,sv = next(it)
                if sv == 0 or sv == 1:
                    break
            out.append((l2[fi][0],l2[fi][1],l2[si-1][2], l3[fi][2:])) # target's name, (start, end) in original text, NER tag

    except StopIteration:
        pass
    
    ##Light filtering due to roberta-tokenizer issue which results in partial predictions
    text = data['text']

    alpha  = list(string.ascii_lowercase)
    Alpha = list(string.ascii_uppercase)
    beta = alpha+Alpha
    punct = list(string.punctuation)

    for i in out.copy():
        offsets = (i[1],i[2])
        if offsets[0] == offsets[1]: # No such entity in original text
            out.remove(i)
            break
        elif offsets[0]>offsets[1]:  # No such entity in original text
            out.remove(i)
            break

        if offsets[0]!= 0:
            if text[offsets[0]-1] in beta: # if the left side of the entity also has english character -> remove such entity
                out.remove(i)
                break
        if offsets[1] != len(text):
            if text[offsets[1]-1] not in punct:
                if text[offsets[1]] in beta: # if the right side of the entity also has english character -> remove such entity
                    out.remove(i)
                    break

    d = {}
    for i in out: # save the updated out { person's name: [(start,end), TAG] }
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append(((i[1],i[2]), i[3]))
        else:
            d[i[0]].append(((i[1],i[2]), i[3]))
    return d
