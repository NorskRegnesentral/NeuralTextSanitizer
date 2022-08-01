import gc         # garbage collector
import torch
import numpy as np
from itertools import combinations

# In case of error, first install pytorch and tensorflow then install transformers
# pip install transformers
from transformers import BertTokenizer, BertModel
from models import MyModel5

def getRepresentation(tokenizer, sequence, start, end):
    """
    Single Sentence Situation
    - sequence doesn't contain '[CLS]' tokens
    - The entity in the original text is sequence[start:end]
    """
    inputs1 = '[CLS]' + sequence[0:start]
    inputs2 = sequence[start:end]
    start = len(tokenizer.tokenize(inputs1))
    end = start + len(tokenizer.tokenize(inputs2))
    return start,end


def getSubtexts(tokenizer, sequence, posList):
    """
    sequence: The original document
    posList: [(start1,end1),(start2,end2),..], the position for the entity in the orginal document
    intervals: [(l1,r1),(l2,r2),...], the interval for i-th entity in the token list

    entityId: the position in the original posList, from 0 to len(posList)-1
    idx: sort entities by its start point in the intervals, posList[idx[0]][0] <= posList[idx[1]][0]
    """
    intervals = []
    for i, (s, e) in enumerate(posList):
        l, r = getRepresentation(tokenizer, sequence, s, e)
        intervals.append((l - 1, r - 1))  # don't consider [CLS] token
    idx = np.argsort([item[0] for item in intervals])  # sort intervals by its starting time

    start = 0
    lastId = -1  # To Note the start entityId of previous intervals in the token list.
    numInterval = 0
    textSplitInterval = []  # Intervals in the original text (strings)
    splitInterval = []  # Intervals in final tokenList

    # (Num of subtext, startPos in this subtext, endPos in this subtext)
    entity2Pos = [(-1, -1, -1) for _ in range(len(idx))]

    for i in idx:
        # print(start,start+510,intervals[i],splitInterval)

        if intervals[i][1] > start + 510:
            numInterval += 1

            if lastId == -1:
                textSplitInterval.append((0, posList[i][0]))
            else:
                textSplitInterval.append((posList[lastId][0], posList[i][0]))

            if intervals[i][0] < start + 510:
                splitInterval.append([start, intervals[i][0]])
            else:
                # intervals[i][0] >= start+510
                splitInterval.append([start, start + 510])

            start = intervals[i][0]
            lastId = i

        # +1, recover to situation having [CLS]
        entity2Pos[i] = (numInterval, intervals[i][0] - start + 1, intervals[i][1] - start + 1)

    splitInterval.append([start, start + 510])
    if lastId != -1:
        textSplitInterval.append((posList[lastId][0], posList[i][1] + 100))  # to enclude the final words
    else:
        assert start == 0
        textSplitInterval.append((0, posList[i][1] + 1000))  # to include the final words and some more texts

    return entity2Pos, textSplitInterval

class simGoogle:
    def __init__(self, tokenizer, bertModel, myModel, device):
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

    def initialize(self, sequence, posList, target):
        """
        get the embedding for each entity whose text span is the posList
        :param sequence: The original article
        :param posList: The list for each text span in the article, type list of tuples
        :param target: The person we want to protect, type string

        == To Optimize ==
        """
        # Target embedding
        inputs = self.tokenizer.encode(target, truncation=True, return_tensors='pt')
        self.targetEmb = self.bertModel(inputs.to(self.device))[0][0, :]

        entity2Pos, textSplitInterval = getSubtexts(self.tokenizer, sequence, posList)
        # Get embedding for each token
        res = []
        for l, r in textSplitInterval:
            with torch.no_grad():
                inputs = self.tokenizer.encode(sequence[l:r], truncation=True, return_tensors='pt')
                res.append(self.bertModel(inputs.to(self.device))[0])

        # Get embedding for the entity
        for i, pos in enumerate(posList):
            subtextId, l, r = entity2Pos[i]
            embedding = res[subtextId][0, l:r]
            self.pos2embedding[tuple(pos)] = embedding

        self.posList = posList

    def judge(self, query):
        """
        :param query: [(i1,j1),(i2,j2)] The position of various entities in the original text.
        :return: The decision whether the query is dangerous or not.
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."
        embedding = [self.pos2embedding[tuple(pos)] for pos in query]
        self.myModel.eval()
        with torch.no_grad():
            outputs = self.myModel(embedding, self.targetEmb)
            pred = np.argmax(outputs.cpu().detach().numpy())
        return pred

    def generateBlackList(self, N = 2):
        """
        Generate a list of entities which are dangerous, considering 1 to N-combinations:
        :param posList: The entity's textspan in the original document
        :return: A blacklist, e.g. [[i],[j,k]] = entity i should MASK, entity j and k can't appear together
        """
        assert self.targetEmb != None and len(self.pos2embedding) > 0, "Please first initialize."
        res = []
        for n in range(1,N+1):
            idx = [i for i in range(len(self.posList))]
            for idPair in combinations(idx,n):
                pair = [tuple(self.posList[i]) for i in idPair] # list of tuples
                if self.judge(pair) == 1:
                    res.append(idPair)
        return res

    def clear(self):
        self.pos2embedding = {}
        self.targetEmb = None
        self.posList = None
        gc.collect()


if __name__ == "__main__":
    # sequence, target, posList
    """
    import json
    with open("SampleData/sample.json","r") as f:
        tmp = json.load(f)
    sequence = tmp["sequence"]
    target = tmp["target"]
    posList = tmp["posList"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bertModel = BertModel.from_pretrained("bert-large-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    myModel = MyModel5(device, drop=True, pooling="max").to(device)
    myModel.load_state_dict(torch.load('SampleData/model5max.pt',map_location=device))

    sim = simGoogle(tokenizer, bertModel, myModel, device)
    sim.initialize(sequence,posList,target)
    res = sim.generateBlackList(2)
    print(res)
    """
    # echr_test
    import json
    with open("WrapUp/SampleData/echr_test.json", "r") as f:
        data = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bertModel = BertModel.from_pretrained("bert-large-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    myModel = MyModel5(device, drop=True, pooling="max").to(device)
    myModel.load_state_dict(torch.load('SampleData/model5max.pt', map_location=device))
    sim = simGoogle(tokenizer, bertModel, myModel, device)

    from tqdm import tqdm
    res = {}
    n = len(data)
    for i in tqdm(range(n)):
        doc = data[i]
        # Get all the text_span given by the annotators
        posList = [(entity['start_offset'],entity['end_offset']) for k in doc['annotations'].keys()
                   for entity in doc['annotations'][k]['entity_mentions']]
        sim.initialize(doc["text"], posList, doc["task"][63:])
        print("initialize done")
        blacklist = sim.generateBlackList(1) # 2
        print("blacklist done")
        # All the items in the res is dangerous -> note its text spans
        dangerous_entities = set()
        for pair in blacklist:
            for p in pair:
                dangerous_entities.add(p)
        res[doc["doc_id"]] = [posList[p] for p in dangerous_entities]
        sim.clear()
    with open("outputs/echr_test_res.json","w") as f:
        json.dump(res,f)