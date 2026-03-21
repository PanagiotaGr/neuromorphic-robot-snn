import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def rate_encode(x, num_steps):
    x_rep = x.unsqueeze(0).repeat(num_steps, 1, 1)
    return torch.bernoulli(x_rep)


class ANNController(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=96, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SNNController(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=96, output_dim=3, beta=0.92):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, spike_input):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec, mem2_rec = [], []
        for t in range(spike_input.size(0)):
            cur1 = self.fc1(spike_input[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec), torch.stack(mem2_rec)


class ANNPolicy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        logits = self.model(x)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        sparse_proxy = float(np.mean(np.abs(scores)))
        return action, scores, sparse_proxy


class SNNPolicy:
    def __init__(self, model, num_steps, device):
        self.model = model
        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def act(self, sensor_values):
        self.model.eval()
        x = torch.tensor([sensor_values], dtype=torch.float32, device=self.device)
        spk_in = rate_encode(x, self.num_steps).to(self.device)
        spk_out, _ = self.model(spk_in)
        logits = spk_out.sum(dim=0)
        action = int(logits.argmax(dim=1).item())
        scores = logits.squeeze(0).detach().cpu().numpy()
        spike_count = float(spk_out.sum().item())
        return action, scores, spike_count
