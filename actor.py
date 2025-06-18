import torch
import torch.nn as nn
import torch.nn.functional as F
from module import StateNetwork
import config

# todo: mask delivery
class StateNetworkActor(nn.Module):
    def __init__(self, num_nodes, net_state_dim, vnf_state_dim, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, state, mask=None):
        x = self.state_network(state, mask)
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(F.relu(x))
        x = self.l3(F.relu(x))
        logits = self.layer_norm(x)
        logits = logits.view(logits.size(0), config.MAX_SFC_LENGTH, self.num_nodes)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class Seq2SeqActor(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim, num_layers, num_nodes):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim

        self.node_linear = nn.Linear(1, vnf_state_dim)  # batch_size * 2 * vnf_state_dim
        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)   # input: batch_size * (max_sfc_length + 2) * vnf_state_dim
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, num_nodes)  # output: batch_size * (max_sfc_length + 2) * num_nodes

    def forward(self, state):
        _, sfc_state, source_dest_node_pair = zip(*state)
        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
            source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)
        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2, 1)
        source_dest_node_pair = self.node_linear(source_dest_node_pair)  # batch_size * 2 * vnf_state_dim
        sfc = torch.cat((sfc_state, source_dest_node_pair),
                        dim=1)  # batch_size * (max_sfc_length + 2) * vnf_state_dim
        embedded = self.embedding(sfc)  # batch_size * max_sfc_length * vnf_state_dim
        encoder_outputs, (h, c) = self.encoder(embedded)  # output, hidden state, cell state
        decoder_outputs, _ = self.decoder(encoder_outputs, (h, c))
        logits = self.fc_out(decoder_outputs)
        probs = F.softmax(logits, dim=-1)
        return logits, probs