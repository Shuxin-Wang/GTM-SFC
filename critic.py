import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from module import StateNetwork, GCNConvNet, Attention, Encoder, TransformerEncoder,GAT
import config

class StateNetworkCritic(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, num_nodes, hidden_dim=256):
        super().__init__()
        self.num_nodes = num_nodes
        self.vnf_state_dim = vnf_state_dim
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.l1 = nn.Linear(vnf_state_dim, 64)
        self.l2 = nn.Linear(vnf_state_dim, 64)
        self.lin = nn.Linear(vnf_state_dim, 64)
        self.attn = nn.Linear(num_nodes, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.l_out = nn.Linear(64, 1)

    def forward(self, state, mask=None):
        state_attention = self.state_network(state, mask)
        net_tokens = state_attention[:, :self.num_nodes, :]  # batch_size * num_nodes * vnf_state_dim
        sfc_tokens = state_attention[:, self.num_nodes:, :]  # batch_size * max_sfc_length * vnf_state_dim
        net_tokens = self.lin(net_tokens)  # batch_size * num_nodes * hidden_dim
        sfc_tokens = self.lin(sfc_tokens)  # batch_size * max_sfc_length * hidden_dim

        # attn_output, _ = self.cross_attn(query=sfc_tokens, key=net_tokens, value=net_tokens)    # batch_size * sfc_length * hidden_dim
        # attn_output = attn_output.mean(dim=1)
        # q = self.l_out(attn_output)

        logits = torch.matmul(sfc_tokens, net_tokens.transpose(1, 2))  # batch_size * max_sfc_length * num_nodes
        attn_weights = torch.softmax(self.attn(logits), dim=1)  # batch_size * max_sfc_length * 1
        weighted_sum = (logits * attn_weights).sum(dim=1)  # batch_size * num_nodes
        q = self.fc(weighted_sum)   # batch_size * 1
        return q

class StateNetworkCriticAction(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.fc = nn.Linear(num_nodes, 1)
        self.fc2 = nn.Linear(2 * num_nodes, 1)

    def forward(self, state, action, mask=None):
        net_state, sfc_state, source_dest_node_pair = zip(*state)
        # action: batch_size * max_vnf_length
        state = self.state_network(state, mask)  # batch_size * (node_num + max_sfc_length) * vnf_state_dim
        net_embed = state[:, :self.num_nodes, :] # batch_size * num_nodes * vnf_state_dim
        sfc_embed = state[:, self.num_nodes:, :] # batch_size * max_sfc_length * vnf_state_dim

        state = torch.matmul(sfc_embed, net_embed.transpose(1, 2)) # batch_size * max_sfc_length * num_nodes

        if action.dim() == 2:
            action = action.unsqueeze(2).to(dtype=torch.float32)
            logits = torch.matmul(state.transpose(1, 2), action).squeeze(2)    # batch_size * num_nodes
            q = self.fc(logits)
        else:
            x = torch.cat((state, action), dim=-1)
            logits = self.fc2(x)
            q = logits.sum(dim=1)

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
        value = self.fc_out(h[-1])  # h[-1]: batch_size * 1
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
