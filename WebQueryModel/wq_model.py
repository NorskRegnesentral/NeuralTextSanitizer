import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LSTM, Tanh

class MyModel5(nn.Module):
    def __init__(self, device, drop=False, concatenation=0):
        super(MyModel5, self).__init__()
        self.lstm = LSTM(1024, 100, bidirectional=False)  # without ner tag
        self.tanh = Tanh()

        # concatenation method
        # 0 (u*v) : 1,N
        # 1 (u,v) : 1,2N
        # 2 (|u-v|, u*v): 1,2N
        # 3 (u, v, |u-v|): 1,3N
        # 4 (u, v, u*v): 1,3N
        # 5 (u, v, |u-v|, u*v): 1,4N
        # 6 (|u-v|) : 1,N
        N = 100
        if concatenation == 0:
            self.concat = lambda x, y: x * y
            self.fc1 = Linear(N, 64)
            self.method = "(u*v)"
        elif concatenation == 1:
            self.concat = lambda x, y: torch.cat([x, y], dim=1)
            self.fc1 = Linear(2 * N, 64)
            self.method = "(u,v)"
        elif concatenation == 2:
            self.concat = lambda x, y: torch.cat([torch.abs(x - y), x * y], dim=1)
            self.fc1 = Linear(2 * N, 64)
            self.method = "(|u-v|,u*v)"
        elif concatenation == 3:
            self.concat = lambda x, y: torch.cat([x, y, torch.abs(x - y)], dim=1)
            self.fc1 = Linear(3 * N, 64)
            self.method = "(u,v,|u-v|)"
        elif concatenation == 4:
            self.concat = lambda x, y: torch.cat([x, y, x * y], dim=1)
            self.fc1 = Linear(3 * N, 64)
            self.method = "(u,v,u*v)"
        elif concatenation == 5:
            self.concat = lambda x, y: torch.cat([x, y, x * y, torch.abs(x - y)], dim=1)
            self.fc1 = Linear(4 * N, 64)
            self.method = "(u,v,|u-v|,u*v)"
        elif concatenation == 6:
            self.concat = lambda x, y: torch.abs(x - y)
            self.fc1 = Linear(N, 64)
            self.method = "(|u-v|)"

        self.pooling = lambda x: torch.max(x, dim=1)[0]
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

        x = self.pooling(hidden)  # (1,100)

        outTarget, (hnTarget, _) = self.lstm(target.to(self.device))
        y = hnTarget.reshape(1, -1)  # (1,100)

        # x = torch.concat([x,hnTarget],dim=1) # 1,2*100
        x = self.concat(x, y)

        # 2 FC layers
        if not self.drop:
            x = self.tanh(self.fc1(x))
            x = self.fc2(x)
            return x
        else:
            x = self.tanh(self.dropout(self.fc1(self.dropout(x))))
            x = self.fc2(x)
            return x

    def predict(self):
        pass

    def __str__(self):
        return f"M2 Concat={self.method}"