from typing import List,Any
import itertools
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch

## Adapted from https://www.lighttag.io/blog/sequence-labeling-with-transformers/example
## See above if my comments are not explanatory enough for the procedure.

IntList = List[int]         # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BI")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l


@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    offsets:IntList

class Dataset(Dataset):
    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch # Default: move tokens_per_batch per window
        self.tokenizer = tokenizer
    
        self.texts = []
        ids = [] # To save target person

        # for example in data:
            # print(data["text"])
        self.texts.append(data["text"])
        ids.append(data['target'])

        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=True, return_offsets_mapping=True)

        ## This is used to keep track of the offsets of the tokens, and used to calculate the offsets on the entity level at evaluation time.
        offset_mapping = [] # [(target person,token's position in the original text),] for each token, len = len(tokens)
        for x,y in zip(ids, tokenized_batch.offset_mapping):
            l = []
            for tpl in y:
                l.append((x, tpl[0], tpl[1]))
            offset_mapping.append(l)

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, mapping  in zip(tokenized_batch.encodings, offset_mapping):
            length = len(mapping)  # How long is this sequence
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)
                # How much padding do we need ? We might break entities into different windows
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attention masks where we added padding
                        ),
                        offsets=(mapping[start:end]
                            + [-1] * padding_to_add # initial items are tuples, last few items are -1.
                        ),

                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:
        return self.training_examples[idx]

class TrainingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.offsets:List
        input_ids: IntListList = []
        masks: IntListList = []
        offsets: List = []

        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            offsets.append(ex.offsets)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.offsets = offsets

        # self.input_ids = self.input_ids.to('cuda')
        # self.attention_masks = self.attention_masks.to('cuda')
        # self.labels = self.labels.to('cuda')
