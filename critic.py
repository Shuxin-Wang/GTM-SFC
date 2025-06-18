import torch
import torch.nn as nn
import torch.nn.functional as F
from module import StateNetwork
import config


class StateNetworkCritic(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, hidden_dim=512):
        super().__init__()
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.state_linear = nn.Linear(vnf_state_dim, 1)
        self.l1 = nn.Linear(vnf_state_dim + config.MAX_SFC_LENGTH, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, mask=None):
        state = self.state_network(state, mask)  # batch_size * (node_num + max_sfc_length + 2) * vnf_state_dim
        # state attention pooling
        score = self.state_linear(state)
        weight = torch.softmax(score, dim=1)
        state = (state * weight).sum(dim=1)  # batch_size * vnf_state_dim
        action = action.squeeze(1)  # batch_size * max_sfc_length
        x = torch.cat((state, action), dim=1)
        x = self.l1(x)
        q = self.l2(F.relu(x))
        return q


class LSTMCritic(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.node_linear = nn.Linear(1, vnf_state_dim)
        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        _, sfc_state, source_dest_node_pair = zip(*state)
        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = torch.tensor(sfc_state[0]).unsqueeze(0)
            source_dest_node_pair = torch.tensor(source_dest_node_pair[0]).unsqueeze(0)
        source_dest_node_pair = self.node_linear(source_dest_node_pair)
        sfc = torch.cat((sfc_state, source_dest_node_pair), dim=1)
        embedded = self.embedding(sfc)
        _, (h, _) = self.encoder(embedded)
        value = self.fc_out(h[-1])
        return value  # batch_size * 1
