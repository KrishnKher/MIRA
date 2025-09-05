import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import FlavaMultimodalConfig, FlavaMultimodalModel


class SimSelModel(nn.Module):
    def __init__(self, kq_dim, v_dim):
        super(SimSelModel, self).__init__()
        mm_model_config = FlavaMultimodalConfig(hidden_size=kq_dim, num_hidden_layers=2, intermediate_size=768*3, num_attention_heads=12)
        self.mm_model = FlavaMultimodalModel(mm_model_config)
        self.sim_pred = nn.Sequential(
            nn.GELU(),
            nn.Linear(768, 1)
        )

        self.key_generator = nn.Sequential(
            nn.Linear(v_dim, kq_dim),
            nn.GELU(),
            nn.Linear(kq_dim, kq_dim),
            nn.GELU(),
            nn.Linear(kq_dim, kq_dim)
        )

        self.value_proj = nn.Identity() # nn.Linear(v_dim, v_dim)
        # self.value_proj.weight.data = torch.eye(v_dim)
        # self.value_proj.bias.data = torch.zeros(v_dim)

    def forward(self, input_embeds, values):
        keys = self.key_generator(values)
        kq = torch.cat([input_embeds, keys], dim=1)
        kq = self.mm_model(kq)[0][:, -keys.size(1):, :]
        sim = self.sim_pred(kq)
        # print(keys.shape, sim.shape, values.shape)
        # exit()
        value_combination = torch.einsum('bd,bdo->bo', sim.squeeze(2), self.value_proj(values))
        return (value_combination * 0.1).softmax(dim=1)

