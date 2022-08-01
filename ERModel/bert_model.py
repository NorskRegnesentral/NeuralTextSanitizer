from transformers import RobertaModel
import torch.nn as nn


class NERModel(nn.Module):
    def __init__(self, model, num_labels):
        super().__init__()
        self._bert = RobertaModel.from_pretrained(model)

        for param in self._bert.parameters():
           param.requires_grad = True

        self.classifier = nn.Linear(768, num_labels)

    # [2,512] [[i1,...,i512],[j1,...j512]]; [[j1,...,j512],[i1,...i512]]
    def forward(self, batch):
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)