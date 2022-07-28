from __future__ import annotations
import math
from collections import deque
import numpy as np
import intervaltree
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForMaskedLM
# import json, re, sys, abc, argparse, math
# from typing import Any, Dict, List, Tuple
# from dataclasses import dataclass
# import tqdm
# import pandas

class mlmbert:
    def __init__(self, device, tokenizer, model, max_segment_size = 100, thres = -20, N = 3):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(device)
        self.max_segment_size = max_segment_size

        # Used in determing which entities are dangerous
        self.thres = thres
        self.N = N

    def get_model_predictions(self, input_ids, attention_mask):
        """Given tokenized input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalized) prediction scores for each token.

        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""

        nb_tokens = len(input_ids)

        input_ids = torch.tensor(input_ids)[None, :].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None, :].to(self.device)

        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens / self.max_segment_size)

            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_size * (i + 1) for i in range(nb_segments - 1)]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)
            # input_ids_splits = torch.tensor_split(input_ids[0], nb_segments)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)

            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            # attention_mask_splits = torch.tensor_split(attention_mask[0], nb_segments)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)

        # Run the model on the tokenized inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # And get the resulting prediction scores
        scores = outputs.logits

        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        return scores

    def get_tokens_by_span(self, bert_token_spans, text_spans):
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

    def get_logproba(self, probs_actual, text_spans, tokens_by_span):
        """
        :param probs_actual: (L,) The proba for each token
        :param text_spans:
        :param token_by_span:
        """
        res = []
        for text_span in text_spans:
            ## If the span does not include any actual token, skip
            ## Normally will not happen
            if not tokens_by_span[text_span]:
                continue
            # Added 1e-60 to avoid error
            res.append(sum([np.log10(probs_actual[token_idx]+1e-60) for token_idx in tokens_by_span[text_span]]))
        return res

    def tree_search(self, logproba):
        """
        Given logproba for choosing each entity
        Get the blacklist (which entity or entities should not appear together)
        Constraints consists of at most self.N entities
        """
        res = []
        n = len(logproba)
        q = deque([[i] for i in range(n)])
        while q:
            nodes = q.popleft()
            if sum(logproba[n] for n in nodes) <= self.thres:
                # Low Probability = Too specific -> dangerous
                res.append(nodes.copy())
            else:
                # Consider its combinations
                for i in range(n):
                    # == could use bit compression to optimize performance ==
                    if i not in nodes and len(nodes) < self.N:
                        q.append(nodes+[i])
        return res

    def get_blacklist_and_semantic_loss(self, text, text_spans):
        """
        Input: text,text_spans (entity position in the annotation)
        Output: blacklist, semantic_loss for each entity
        """
        blacklist = []
        semantic_loss = []

        tokenized_res = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = tokenized_res["input_ids"]
        input_ids_copy = np.array(input_ids)
        bert_token_spans = tokenized_res['offset_mapping']
        tokens_by_span = self.get_tokens_by_span(bert_token_spans, text_spans)

        attention_mask = tokenized_res["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokenizer.mask_token_id

        logits = self.get_model_predictions(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1) # (L, Number of tokens in the dict)

        # Get prob for the input tokens
        probs_actual = probs[torch.arange(len(input_ids_copy)), input_ids_copy]  # (L,)
        probs_actual = probs_actual.detach().cpu().numpy()

        logproba = self.get_logproba(probs_actual, text_spans, tokens_by_span)
        semantic_loss = [-logp for logp in logproba]
        blacklist = self.tree_search(logproba)
        # for entity_span,logp in zip(text_spans,logproba):
        #     print(text[entity_span[0]:entity_span[1]],'\t',logp)
        return blacklist, semantic_loss

if __name__ == "__main__":
    text = "Bodewin Claus Eduard Keitel (German pronunciation: [ˈkaɪ̯tl̩]; 1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office."
    text_spans = [(0, 27), (29, 35), (51, 61), (63, 67), (70, 74), (82, 88), (89, 96), (104, 116), (131, 135), (143, 164)]  # annotation text_span

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    mlmbert_model = mlmbert(device,tokenizer,model)
    blacklist, semantic_loss = mlmbert_model.get_blacklist_and_semantic_loss(text,text_spans)

    text_entities = [text[p[0]:p[1]] for p in text_spans]

    for entity,sem_loss in zip(text_entities,semantic_loss):
        print(entity,'\t',sem_loss)
    print("-------------")
    print("Blacklist:")
    print(blacklist)
    for pair in blacklist:
        print(" AND ".join(text_entities[p]for p in pair))