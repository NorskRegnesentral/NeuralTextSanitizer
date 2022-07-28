import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LSTM, Tanh, CrossEntropyLoss, Conv1d
import torch.nn.functional as F

class MyModel5(nn.Module):
    def __init__(self, device, drop=False, pooling="lstm"):
        super(MyModel5, self).__init__()
        self.lstm = LSTM(1024, 100, bidirectional=False)  # without ner tag
        self.tanh = Tanh()
        self.method = pooling
        if pooling == "lstm":
            self.pooling = LSTM(100, 100, bidirectional=False, batch_first=True)
        elif pooling == "max":
            self.pooling = lambda x: torch.max(x, dim=1)[0]
        self.fc1 = Linear(2 * 100, 64)
        self.fc2 = Linear(64, 2)
        self.drop = drop
        self.dropout = Dropout(p=0.2)  # 0.5
        self.device = device

    def forward(self, entities, target):
        hidden = []
        for entity in entities:
            out, (hn, _) = self.lstm(entity.to(self.device))  # entity (L,1024), out (L,100), hn (1,100)
            hidden.append(hn.reshape(1, -1))
        hidden = torch.concat(hidden, dim=0).unsqueeze(0)  # (1,len(entities), 100)

        if self.method == "lstm":
            _, (x, _) = self.pooling(hidden)  # (1,100)
            x = x.reshape(1, -1)
        elif self.method == "max":
            x = self.pooling(hidden)  # (1,100)

        outTarget, (hnTarget, _) = self.lstm(target.to(self.device))
        hnTarget = hnTarget.reshape(1, -1)  # (1,100)

        x = torch.concat([x, hnTarget], dim=1)  # 1,2*100

        # 2 FC layers
        if not self.drop:
            x = self.tanh(self.fc1(x))
            x = self.fc2(x)
            return x
        else:
            x = self.tanh(self.dropout(self.fc1(self.dropout(x))))
            x = self.fc2(x)
            return x

    def __str__(self):
        return f"Q5 Pooling Method {self.method}"