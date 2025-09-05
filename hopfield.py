import torch
import torch.nn as nn
import torch.nn.functional as F


class Hopfield:
    def __init__(self, beta=25, iters=5):
        self.beta = beta
        self.keys = None
        self.values = None
        self.iters = iters
        self.device = None

    def learn(self, key, value):
        if self.keys is None:
            self.keys = key.unsqueeze(0)
            self.values = value.unsqueeze(0)
        else:
            self.keys = torch.cat((self.keys, key.unsqueeze(0)), dim=0).to(self.device)
            self.values = torch.cat((self.values, value.unsqueeze(0)), dim=0).to(self.device)

    def infer(self, query, cat_query=True):
        orig_query = query
        query = query.unsqueeze(0)
        for _ in range(self.iters):
            sim = (query @ self.keys.t()).squeeze(0)
            query = (F.softmax(self.beta * sim, dim=0) @ self.keys).unsqueeze(0)
        sim = (query @ self.keys.t()).squeeze(0)
        if cat_query:
            return torch.cat([orig_query, (F.softmax(self.beta * sim, dim=0) @ self.values)], dim=-1)
        else:
            return F.softmax(self.beta * sim, dim=0) @ self.values
