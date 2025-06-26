import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from module import StateNetwork, GCNConvNet, Attention, Encoder
import config


class StateNetworkCritic(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, num_nodes, hidden_dim=512):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.state_linear = nn.Linear(vnf_state_dim, 1)
        self.action_linear = nn.Linear(num_nodes, 1)
        self.l1 = nn.Linear(vnf_state_dim + config.MAX_SFC_LENGTH, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, mask=None):
        # action: batch_size * max_vnf_length * num_nodes
        state = self.state_network(state, mask)  # batch_size * (node_num + max_sfc_length + 2) * vnf_state_dim

        # state attention pooling
        state_score = self.state_linear(state)
        state_weight = torch.softmax(state_score, dim=1)
        state = (state * state_weight).sum(dim=1)  # batch_size * vnf_state_dim

        # action attention
        action_score = self.action_linear(action)
        action_weight = torch.softmax(action_score, dim=-1)
        action = (action * action_weight).sum(dim=-1)

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
        sfc = torch.cat((sfc_state, source_dest_node_pair), dim=1)  # batch_size * (max_sfc_length + 2) * vnf_state_dim
        embedded = self.embedding(sfc)
        _, (h, _) = self.encoder(embedded)
        value = self.fc_out(h[-1])
        return value  # batch_size * 1

class DecoderCritic(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, embedding_dim=64):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.node_linear = nn.Linear(1, vnf_state_dim)  # source_dest_node_pair, batch_size * 2 * vnf_state_dim
        self.encoder = Encoder(vnf_state_dim, embedding_dim=embedding_dim)  # sfc_state
        self.gcn = GCNConvNet(net_state_dim, embedding_dim)  # net_state
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )  # net_state
        self.emb = nn.Linear(embedding_dim, 1)  # placement
        self.att = Attention(embedding_dim)  # placement
        self.gru = nn.GRU(embedding_dim, embedding_dim)  # decoder

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair = zip(*state)
        net_state_list = list(net_state)
        batch_net_state = Batch.from_data_list(net_state_list)

        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
            source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)

        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2, 1)
        source_dest_node_pair = self.node_linear(source_dest_node_pair)

        net_state = self.gcn(batch_net_state)
        encoder_input = torch.cat((sfc_state, source_dest_node_pair), dim=1)    # batch_size * (max_sfc_length + 2) * vnf_state_dim
        encoder_output, encoder_hidden_state = self.encoder(encoder_input)  # (max_sfc_length + 2) * batch_size * embedding_dim, 1 * batch_size * embedding_dim

        hidden_state = encoder_hidden_state

        placement_logits_list = []

        query = hidden_state[-1].unsqueeze(1)   # batch_size * 1 * embedding_dim
        seq_len = encoder_output.size(0)

        for t in range(config.MAX_SFC_LENGTH):
            repeated_query = query.expand(-1, seq_len, -1)  # batch_size * seq_len * embedding_dim
            context, attn = self.att(repeated_query, encoder_output)    # context: batch_size * 1 * embedding_dim
            gru_input = context.permute(1, 0, 2)    # 1 * batch_size * embedding_dim

            output, hidden_state = self.gru(gru_input, hidden_state)    # 1 * batch_size * embedding_dim

            decoder_output = output.squeeze(0)  # batch_size * embedding_dim

            scores = torch.matmul(net_state, decoder_output.t())    # num_nodes * batch_size
            scores = scores.permute(1, 0)   # batch_size * num_nodes
            placement_logits_list.append(scores)

            query = output.permute(1, 0, 2) # batch_size * 1 * embedding_dim

        all_logits = torch.stack(placement_logits_list, dim=1)  # batch_size * max_sfc_length * num_nodes
        value = torch.mean(all_logits, dim=-1, keepdim=True)

        return value
