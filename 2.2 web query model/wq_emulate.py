import intervaltree
import numpy as np
import math
from transformers import BertTokenizerFast, BertModel
from itertools import combinations

import gc
import torch
from itertools import combinations

from wq_model import MyModel5

# Test using BertForMaskedLM
# import torch.nn as nn
# from transformers import BertForMaskedLM

def get_tokens_by_span(bert_token_spans, text_spans):
    """Given two lists of spans (one with the spans of the BERT tokens, and one with
    the text spans to weight), returns a dictionary where each text span is associated
    with the indices of the BERT tokens it corresponds to."""

    # We create an interval tree to facilitate the mapping
    text_spans_tree = intervaltree.IntervalTree()

    for start, end in text_spans:
        text_spans_tree[start:end] = True

    # We create the actual mapping between spans and tokens index in the token list
    tokens_by_span = {span: [] for span in text_spans}
    for token_idx, (start, end) in enumerate(bert_token_spans):
        for span_start, span_end, _ in text_spans_tree[start:end]:
            tokens_by_span[(span_start, span_end)].append(token_idx)

    # And control that everything is correct
    for span_start, span_end in text_spans:
        if len(tokens_by_span[(span_start, span_end)]) == 0:
            print("Warning: span (%i,%i) without any token" % (span_start, span_end))
    return tokens_by_span

def token_pos_to_subtexts(entity_tokens, token_list_orig,
                          pad_id=0, cls_token_id=101, sep_token_id=102,
                          max_seq_length=512):
    """
    Input:
    A list of token position in the original token list
    The original tokenlist
    Output:
    pos: [(subtext1,pos1),...]
    subtexts: [subtext_input1,...]

    Naive Method
    """
    assert max_seq_length % 2 == 0, "Please input an even max_seq_length"
    nb_tokens = len(token_list_orig)
    pos = []
    subtexts = []
    for i, token_pos in enumerate(entity_tokens):
        l, r = token_pos[0] - 1, token_pos[1] - 1  # Remove [CLS] token
        # If the entity's already longer than max_seq_length-2 tokens -> keep the first max_seq_length-2
        if (r - l) > max_seq_length - 2:
            print(
                f"entity longer than {max_seq_length - 2} tokens, keeping first {max_seq_length - 2}. token pos {token_pos}")
            pos.append((i, (1, 1 + max_seq_length - 2)))
            input_ids = [cls_token_id] + token_list_orig[l:l + max_seq_length - 2] + [sep_token_id]
            subtexts.append(input_ids.copy())
            continue

        l1 = int(max_seq_length / 2) - 1 - math.floor((r - l) / 2)
        l2 = int(max_seq_length / 2) - 1 - math.ceil((r - l) / 2)

        # assert l1+l2+r-l == max_seq_length-2
        input_ids = [cls_token_id] + [pad_id] * (l1 - l) + \
                    token_list_orig[max(0, l - l1):min(r + l2, nb_tokens)] + \
                    [pad_id] * ((r + l2) - nb_tokens) + [sep_token_id]
        assert len(input_ids) == max_seq_length, len(input_ids)
        pos.append((i, (1 + l1 + 1, 1 + l1 + 1 + (r - l))))
        subtexts.append(input_ids.copy())
    return pos, subtexts


def getSafeEntities(entityIdList, dangerQueries):
    def existsDanger(solution, entityPairs):
        sol = set(solution)
        for ep in entityPairs:
            if set(ep).issubset(sol):
                return True
        return False

    if len(dangerQueries) == 0:
        return entityIdList
    else:
        entityPairs = dangerQueries
        n = len(entityPairs[0])  # n-combination
        if n == 1:
            dangerQueries = [d[0] for d in dangerQueries]  # Flatten, [(2,)] -> [2]
            return [e for e in entityIdList if e not in dangerQueries]
        else:
            # If we don't have many dangerQueries, then we can use adjacency matrix
            res = entityIdList.copy()
            N = len(entityIdList)
            A = np.zeros((N, N))
            id2entity = {i: e for i, e in enumerate(entityIdList)}
            entity2id = {e: i for i, e in id2entity.items()}
            for ep in entityPairs:
                for comb in combinations(ep, 2):
                    id1 = entity2id[comb[0]]
                    id2 = entity2id[comb[1]]
                    A[id1][id2] += 1
                    A[id2][id1] += 1
            while existsDanger(res, entityPairs):
                # Remove nodes with highest degree
                removeId = np.argmax(np.sum(A, axis=0))
                res.remove(id2entity[removeId])  # remove
                A[removeId, :] *= 0
                A[:, removeId] *= 0
            return res

class simGoogle:
    def __init__(self, tokenizer, bertModel, myModel, device, bs=128):
        """
        For each article:
        1. First call initialize(sequence,posList), sequence is the article,
           posList is the text span for each entity in the article
        2. Then can try different combinations by calling the `judge` function
        3. Call `clear` function to the embedding for this article when finishes
        """
        self.device = device
        self.tokenizer = tokenizer
        self.bertModel = bertModel.to(self.device)
        self.myModel = myModel.to(self.device)
        self.myModel.eval()
        self.pos2embedding = {}
        self.targetEmb = None
        self.posList = None
        self.bs = bs  # To save GPU memory

    def initialize(self, sequence, posList, target):
        """
        get the embedding for each entity whose text span is the posList
        :param sequence: The original article
        :param posList: The list for each text span in the article, type list of tuples
        :param target: The person we want to protect, type string
        To get pos2embedding: (i,j): embedding
        == To Optimize ==
        """
        posList = [tuple(p) for p in posList] # in case the posList is [(0,10),(15,28),..]

        # Target embedding
        inputs = self.tokenizer.encode(target, truncation=True, return_tensors='pt')
        with torch.no_grad():
            self.targetEmb = self.bertModel(inputs.to(self.device))[0][0, :].cpu()  # move to cpu!

        #### Should KEEP consistant with the data acquisition method #####
        tokenized_res = self.tokenizer(sequence, return_offsets_mapping=True)
        token_list_orig = tokenized_res['input_ids']
        bert_token_spans = tokenized_res['offset_mapping']  # Each token -> text span
        tokens_by_span = get_tokens_by_span(bert_token_spans, posList)  # text span -> token list

        entity_tokens = [(min(tokens_by_span.get(text_span)),
                          max(tokens_by_span.get(text_span)) + 1)
                         for text_span in posList]  # [(i1,j1),...(in,jn)]

        # Get the embedding as a batch
        pos, subtexts = token_pos_to_subtexts(entity_tokens, token_list_orig,
                                              pad_id=0, max_seq_length=512)
        subtexts = torch.tensor(subtexts).to(self.device)
        outputs = []
        # print(subtexts.shape)
        with torch.no_grad():
            i = 0
            while i < subtexts.shape[0]:
                outputs.append(self.bertModel(subtexts[i:i + self.bs])[0])
                i += self.bs
            # outputs = bertModel(subtexts)[0]
        ##################################################################
        outputs = torch.concat(outputs, dim=0)
        # print(outputs.shape)

        # Get embedding for the entity
        for p, t in zip(posList, pos):
            self.pos2embedding[tuple(p)] = outputs[t[0], t[1][0]:t[1][1]].cpu()  # move to cpu!

        self.posList = posList

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    def judge(self, query):
        """
        :param query: [(i1,j1),(i2,j2)] The position of various entities in the original text.
        :return: The decision whether the query is dangerous or not.
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."
        embedding = [self.pos2embedding[tuple(pos)] for pos in query]
        self.myModel.eval()
        with torch.no_grad():
            outputs = self.myModel(embedding, self.targetEmb) # 需要修改这里
            pred = np.argmax(outputs.cpu().detach().numpy())
        return pred

    def generateBlackList(self, N=2):
        """
        Generate a list of entities which are dangerous, considering 1 to N-combinations:
        :param posList: The entity's textspan in the original document
        :return: A blacklist, e.g. [[i],[j,k]] = entity i should MASK, entity j and k can't appear together
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."
        res = []
        for n in range(1, N + 1):
            idx = [i for i in range(len(self.posList))]
            for idPair in combinations(idx, n):
                pair = [tuple(self.posList[i]) for i in idPair]  # list of tuples
                if self.judge(pair) == 1:
                    res.append(idPair)
        return res

    def generateBlackListHeuristic(self, N=2):
        """
        Use heuristics ways to generate balcklist
        [[i],[j]] i,j are the elements in the list(start-set(safeEntities))
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."

        safeEntities = list(range(len(self.posList)))  # [0,...,nb_entities]
        # print(safeEntities)
        start = set(safeEntities)
        for n in range(1, N + 1):
            dangerQueries = []
            for idPair in combinations(safeEntities, n):
                pair = [tuple(self.posList[i]) for i in idPair]
                if self.judge(pair) == 1:
                    dangerQueries.append(idPair)
            safeEntities = getSafeEntities(safeEntities, dangerQueries)
            # if n == 2 and len(dangerQueries)>0:
            #     print(dangerQueries)
            # print('\t',dangerQueries)
            # print(safeEntities)
        res = list(start - set(safeEntities))
        return [[r] for r in res]

    def generateBlackListHeuristic2(self, N=2):
        """
        Use heuristics ways to generate balcklist
        [[i],[j]] i,j are the elements in the list(start-set(safeEntities))
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."

        safeEntities = list(range(len(self.posList)))  # [0,...,nb_entities]
        # print(safeEntities)
        # start = set(safeEntities)
        res = []
        for n in range(1, N + 1):
            dangerQueries = []
            for idPair in combinations(safeEntities, n):
                pair = [tuple(self.posList[i]) for i in idPair]
                if self.judge(pair) == 1:
                    dangerQueries.append(idPair)
                    res.append(idPair)
            safeEntities = getSafeEntities(safeEntities, dangerQueries)
            # if n == 2 and len(dangerQueries)>0:
            #     print(dangerQueries)
            # print('\t',dangerQueries)
            # print(safeEntities)
        # res = list(start-set(safeEntities))
        return res

    def clear(self):
        self.pos2embedding = {}
        self.targetEmb = None
        self.posList = None
        gc.collect()

if __name__ == "__main__":
    import json

    # with open("../SampleData/sample.json", "r") as f:
    #     tmp = json.load(f)
    # sequence = tmp["sequence"]
    # target = tmp["target"]
    # posList = tmp["posList"]

    sequence = "Bodewin Claus Eduard Keitel (German pronunciation: [ˈkaɪ̯tl̩]; 1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office."
    target = "bodewin keitel"
    posList = [(0, 27), (29, 35), (51, 61), (63, 67), (70, 74), (82, 88), (89, 96), (104, 116), (131, 135), (143, 164)]  # annotation text_span

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')
    bertModel = BertModel.from_pretrained('bert-large-cased')
    # bertModelForMLM = BertForMaskedLM.from_pretrained('bert-large-cased')
    # bertModel = nn.Sequential(bertModelForMLM.bert.embeddings, bertModelForMLM.bert.encoder)
    myModel = MyModel5(device, drop=True, concatenation=1).to(device)
    myModel.load_state_dict(torch.load('../SampleData/(u,v)_annotator_based.pt', map_location=device))
    sim = simGoogle(tokenizer, bertModel, myModel, device, bs=32)

    sim.initialize(sequence, posList, target)

    print(sequence)
    print(posList)
    print(target)