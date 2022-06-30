import tqdm

import numpy as np
from typing import Dict, Tuple, Union, List
import sys, uuid, re, json, collections, pickle, os, wget
import transformers
import torch
import torch.nn as nn 
import dill 
# import datasets
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_FILE = "mask_classifier.dill"

class MaskClassifier(nn.Module):
    
    def __init__(self, model):
        
        super(MaskClassifier,self).__init__() 
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model)
                    
    def forward(self, input_ids=None, attention_mask=None):
        
        (logits,) = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        probs = torch.sigmoid(logits[:,1])
        return probs
                 
            
    def fine_tune(self, train_data, dev_data=None, num_epochs=3, batch_size=1, device="cuda", lr=5e-5):
    
        self = self.to(device)
        if batch_size > 1:
            train_data.padding = True
            if dev_data is not None:
                dev_data.padding = True
                
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                   num_workers=4, shuffle=True)
             
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        lr_scheduler = transformers.get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, 
                                                  num_training_steps=num_epochs * len(train_data))

        if dev_data is not None:
            # accuracy = datasets.load_metric("accuracy")
            # f1 = datasets.load_metric("f1")
            dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                                     num_workers=4, shuffle=False)

        loss_fct = nn.BCELoss()
        losses = []
        accuracies = []
        
        for epoch in range(num_epochs):
            progress_bar = tqdm.auto.tqdm(range(len(train_data)))
            self.train()
            for _, inputs, labels in train_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                probs = self(**inputs).double()
                labels = labels.to(device)
                predictions = (probs > 0.5)
                
                loss = loss_fct(probs, labels)
                losses.append(loss.item())
                accuracies += (predictions == labels).to("cpu")
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix(avg_loss=np.mean(losses), avg_acc=np.mean(accuracies))
                progress_bar.update(batch_size)
            progress_bar.close()
            
            print("Finished training for epoch %i"%epoch)
 
            if dev_data is not None:
                progress_bar_eval = tqdm.auto.tqdm(range(len(dev_data)))
                self.eval()
                for _, inputs, labels in dev_loader:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        probs = self(**inputs)
                        predictions = (probs > 0.5)
                        labels = labels.to(device)
                        # accuracy.add_batch(predictions=predictions, references=labels)
                        # f1.add_batch(predictions=predictions, references=labels)
                        
                    progress_bar_eval.update(batch_size)
                progress_bar_eval.close()
                    
            # print("Accuracy on dev set:", accuracy.compute())
            # print("F1 score on dev set:", f1.compute())
            
            
    def predict(self, data, batch_size=10, device="cuda"):
        
        if batch_size == 1:
            data.padding = False
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=4, 
                                             shuffle=False) 
        print(device)
        self = self.to(device)
        self.eval()
        
        results = {}
        progress_bar = tqdm.auto.tqdm(range(len(data)))
        with torch.no_grad():
            for ent_ids, inputs, _ in loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                probs = self(**inputs).double().to("cpu").detach().numpy()
                for ent_id, prob in zip(ent_ids, probs):
                    results[ent_id] = prob
                    
                progress_bar.update(batch_size)
                
        return results
    
    
    def predict_from_labelled_text(self, text:str, spans:Dict[Tuple[int,int],str], tokenizer="roberta-base", device="cuda"):
        
        if not hasattr(self, "tokenizer"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base") # tokenizer
            
        data = MaskingDataset(self.tokenizer)
        data.add_annotated_text(text, spans)
        
        results = self.predict(data, device=device)
        results_by_span = {}
        for entity in data.entities:
            for start,end in entity["mentions"]:
                results_by_span[(start,end)] = results[entity["entity_id"]]
                
        return results_by_span
    
             
    def save(self, filename):
        print("Saving to", filename)
        torch.save(self, filename, pickle_module=dill)
        
    @classmethod
    def load(cls, filename, device="cuda"):
        model = torch.load(filename, pickle_module=dill, map_location=device)
        model.eval()
        return model
    
               

class MaskingDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, max_length = 512):
        if type(tokenizer)==str:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.entities = []
        self.texts = {}
        self.three_dots_id = self.tokenizer.convert_tokens_to_ids("...") 
        self.max_length = max_length 
        self.padding = True
        
    def read_json(self, json_file: str, cutoff=None):
        
        self.nb_ambiguous = 0
        
        with open(json_file, encoding="utf-8") as fd:
            dico = json.load(fd)
            progress_docs = tqdm.auto.tqdm(range(len(dico)))
            for annotated_doc in dico:
                
                text = annotated_doc["text"]
                tokenized_text = self.tokenizer(text, return_offsets_mapping=True)
                self.texts[annotated_doc["doc_id"]] = text
                     
                for entity in self._extract_entities(annotated_doc):
                    entity["token_indices"] = list(tokenized_text["input_ids"])
                    entity["offsets"] = list(tokenized_text["offset_mapping"])
                    self._add_category_sentence(entity) 
                    self._compute_entity_positions(entity)   
                    self._shorten_inputs(entity)
                    
                    self.entities.append(entity)
                    
                progress_docs.update(1)  
                if cutoff is not None and progress_docs.n >= cutoff:
                    break    
                        
        print("Discarding %i ambiguous entities"%self.nb_ambiguous)
        print("Total number of documents:", len(self.texts))
        print("Total number of entities:", len(self.entities))
            
        return self
    
    
    def add_entity(self, text: str, entity_strings:List[str], category: str):
        doc_id = str(uuid.uuid4())
        self.texts[doc_id] = text
        
        mentions = []
        for entity_string in entity_strings:
            for match in re.finditer(re.escape(entity_string), text):
                mentions.append((match.start(0), match.end(0)))
        
        entity = {"doc_id": doc_id, "mentions": mentions, "category": category}
        
        tokenized_text = self.tokenizer(text, return_offsets_mapping=True)
            
        entity["token_indices"] = list(tokenized_text["input_ids"])
        entity["offsets"] = list(tokenized_text["offset_mapping"])
           
        self._add_category_sentence(entity) 
        self._compute_entity_positions(entity) 
        self._shorten_inputs(entity)

        self.entities.append(entity)
          
        return self
    
    def add_annotated_text(self, text:str, spans:Dict[Tuple[int,int],str]):
        doc_id = str(uuid.uuid4())
        self.texts[doc_id] = text
        tokenized_text = self.tokenizer(text, return_offsets_mapping=True)
        
        mentions_by_string = {}
        for start, end in spans:
            ent_string = text[start:end]
            mentions_by_string[ent_string] = mentions_by_string.get(ent_string, []) + [(start,end)]

        for entity_string, mentions in mentions_by_string.items():
            
            category_counts = {}
            for (start,end), cat in spans.items():
                if (start,end) in mentions:
                    category_counts[cat] = category_counts.get(cat,0) +1    
            most_common_cat = max(category_counts, key=lambda x: category_counts[x])
            
            entity = {"doc_id":doc_id, "entity_id":str(uuid.uuid4()), "mentions":mentions,
                          "category":most_common_cat, "string":entity_string, 
                          "token_indices":list(tokenized_text["input_ids"]),
                          "offsets":list(tokenized_text["offset_mapping"])}

            self._add_category_sentence(entity) 
            self._compute_entity_positions(entity) 
            self._shorten_inputs(entity)
             
            self.entities.append(entity)
        
        return self        
        
        
    
    def _extract_entities(self, annotated_doc):
        mentions_by_ent = {}
        masks_by_ent = {}
        cat_by_ent = {}
        
        for annotator in annotated_doc["annotations"]:
            for mention in annotated_doc["annotations"][annotator]["entity_mentions"]:
                span = (mention["start_offset"], mention["end_offset"])
                entity_id = mention["entity_id"]
                decision = "MASK" if mention["identifier_type"] in {"QUASI", "DIRECT"} else "NO_MASK"
                category = mention["entity_type"]
                if entity_id in mentions_by_ent:
                    mentions_by_ent[entity_id].append(span)
                    masks_by_ent[entity_id][decision] = masks_by_ent[entity_id].get(decision, 0) + 1
                    cat_by_ent[entity_id][category] = cat_by_ent[entity_id].get(category, 0) + 1
                else:
                    mentions_by_ent[entity_id] = [span]
                    masks_by_ent[entity_id] = {decision: 1}
                    cat_by_ent[entity_id] = {category:1}

        for entity_id, mentions in mentions_by_ent.items():
            most_common_cat = max(cat_by_ent[entity_id], 
                                  key=lambda x:cat_by_ent[entity_id][x])
            most_common_decision = max(masks_by_ent[entity_id], 
                                       key=lambda x:masks_by_ent[entity_id][x])
            mention_strings_count = {}
            for start, end in mentions:
                mention_str = self.texts[annotated_doc["doc_id"]][start:end]
                mention_strings_count[mention_str] = mention_strings_count.get(mention_str, 0) + 1
            most_common_string = max(mention_strings_count.keys(), key=lambda x: mention_strings_count[x])
            
            if most_common_decision == "MASK" and masks_by_ent[entity_id].get("NO_MASK", 0) > 1:
                self.nb_ambiguous += 1
                continue
            elif most_common_decision=="NO_MASK" and "MASK" in masks_by_ent[entity_id]:
                self.nb_ambiguous += 1
                continue   
                             
            entity_dic = {"doc_id":annotated_doc["doc_id"], "entity_id":entity_id, 
                          "mentions":mentions, "mask_decision":most_common_decision, 
                          "category":most_common_cat, "string":most_common_string}
                    
            yield entity_dic
        
    
    def _compute_entity_positions(self, entity):
            
        mentions_token_indices = []
        for i, (start_token, end_token) in enumerate(entity["offsets"]):
            if start_token==end_token:
                continue    
            all_mentions = entity["mentions"] + [entity["extra_mention"]]
            for start_span, end_span in all_mentions:
                if start_token >=start_span and end_token <=end_span:
                    mentions_token_indices.append(i)
                elif start_token <= start_span and end_token > start_span:
                    mentions_token_indices.append(i)
                elif start_token < end_span and end_token >= end_span:
                    mentions_token_indices.append(i)
                    
        entity["entity_positions"] = list(np.unique(mentions_token_indices))
     
                
    def _add_category_sentence(self, entity):
    
        last_sentence = "%s is a %s."%(entity["string"], entity["category"])
        last_sentence_tokens = self.tokenizer(last_sentence, return_offsets_mapping=True)

        orig_text_length = len(self.texts[entity["doc_id"]])  
        
        entity["token_indices"] = (entity["token_indices"][:-1] + [self.tokenizer.sep_token_id] 
                                   + last_sentence_tokens["input_ids"][1:])
        entity["offsets"] += [(orig_text_length + start, orig_text_length + end) 
                                     for start, end in last_sentence_tokens["offset_mapping"][1:]]
        entity["extra_mention"] = (orig_text_length, orig_text_length + len(last_sentence))
        
        
    def _shorten_inputs(self, entity):
        
        while len(entity["token_indices"]) > self.max_length:
            
            tokens = np.array(entity["token_indices"])
            offsets = entity["offsets"]  
            
            distances_to_entity_tokens = np.abs(np.arange(len(tokens))[:,np.newaxis] - entity["entity_positions"])
            min_distance_to_entity = np.min(distances_to_entity_tokens, axis=1)
            
            nb_tokens_to_cut = len(tokens) - self.max_length + 2
            token_indices_sorted_by_distance = min_distance_to_entity.argsort()
            tokens_to_cut = [i for i in token_indices_sorted_by_distance 
                             if tokens[i]!=self.three_dots_id][-nb_tokens_to_cut:]
            tokens[tokens_to_cut] = self.three_dots_id
            
            new_tokens = []
            new_offsets = []
            span_start = None
            for i, token in enumerate(tokens):
                if token==self.three_dots_id:
                    if span_start is None:
                        span_start, _ = offsets[i]
                        new_tokens.append(token)
                    if i==(len(tokens)-1) or tokens[i+1]!=self.three_dots_id:
                        _, span_end = offsets[i]
                        new_offsets.append((span_start, span_end))
                else:
                    new_tokens.append(token)
                    new_offsets.append(offsets[i])
                    span_start = None
        
            entity["token_indices"] = np.array(new_tokens)
            entity["offsets"] = new_offsets
                    
            # The entity positions need to be recomputed      
            self._compute_entity_positions(entity)
      
                              
    def __len__(self):
        return len(self.entities)
    
    def __getitem__(self, idx):
        
        entity = self.entities[idx]

        feats = {"input_ids":torch.IntTensor(entity["token_indices"])}    
        
        feats["attention_mask"] = torch.ones(feats["input_ids"].shape, dtype=int)

        if self.padding and len(feats["input_ids"]) < self.max_length:
            nb_to_pad = self.max_length - len(feats["input_ids"])
            padded_seq = torch.ones(nb_to_pad, dtype=int) * self.tokenizer.pad_token_id
            feats["input_ids"] = torch.concat((feats["input_ids"], padded_seq))
            feats["attention_mask"] = torch.concat((feats["attention_mask"], torch.zeros(nb_to_pad, dtype=int)))

        label = float(entity["mask_decision"]=="MASK") if "mask_decision" in entity else np.nan

        return entity["entity_id"], feats, label  
    
    
    def save(self, filename):
        tokenizer = self.tokenizer
        setattr(self, "tokenizer", self.tokenizer.name_or_path)
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)
        self.tokenizer = tokenizer    
        return self
    
        
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as fd:
            data = pickle.load(fd)
            data.tokenizer = transformers.AutoTokenizer.from_pretrained(data.tokenizer)
            return data


def train_script():
    args = {"lr":1e-5, "num_epochs":3, "batch_size":10, "model":"allenai/longformer-base-4096", "doc_size":512}
    for argv in sys.argv[1:]:
        key, val = argv.rstrip(",").strip().split("=")
        if key=="lr":
            args[key] = float(val)
        elif key=="model":
            args[key] = val
        elif key in {"num_epochs", "batch_size", "doc_size"}:
            args[key] = int(val)
        else:
            raise RuntimeError("Unrecognized argument:", argv)
    print("Parameters:", args)
    
    
    train_data_pkl = "train-%i.pkl"%args["doc_size"]
    dev_data_pkl= "dev-%i.pkl"%args["doc_size"]
    if os.path.exists(train_data_pkl):
        train_data = MaskingDataset.load(train_data_pkl)
        dev_data = MaskingDataset.load(dev_data_pkl)
    else: 
       train_data = MaskingDataset("roberta-base").read_json("../text-anonymisation-benchmark/echr_train.json")
       train_data.save(train_data_pkl)
       dev_data = MaskingDataset("roberta-base").read_json("../text-anonymisation-benchmark/echr_dev.json")
       dev_data.save(dev_data_pkl)
    
    classifier = MaskClassifier(model=args["model"])
    classifier.fine_tune(train_data, dev_data, batch_size=args["batch_size"], 
                         lr=args["lr"], num_epochs=args["num_epochs"])
    
    desc = json.dumps(args).replace("\"", "").replace(": ", "=").replace(", ", "_")
    desc = desc.replace("allenai/longformer-base-4096", "longformer").replace("roberta-base", "roberta")
    classifier.save("fine_tuned_mask_classifier-%s.dill"%desc)

    
if __name__ == "__main__":
    # if not os.path.exists(MODEL_FILE):
    #     print("Downloading fine-tuned model...")
    #     wget.download("https://home.nr.no/~plison/data/" + MODEL_FILE)

    text = "This is Pierre Lison, living in Oslo. More precisely at Kalbakken, on the east side of Oslo, in Norway."
    spans = {(8,20):"PERSON", (32,36):"LOC", (56,65):"LOC", (87,91):"LOC", (96,102):"LOC"}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = MaskClassifier.load("mask_classifier.dill",device)
    results = classifier.predict_from_labelled_text(text, spans, device = device)
    print(results)