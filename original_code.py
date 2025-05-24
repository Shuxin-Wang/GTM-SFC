# class Critic(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=512):
#         super().__init__()
#         self.l1 = nn.Linear(input_dim, hidden_dim)
#         self.l2 = nn.Linear(hidden_dim, hidden_dim)
#         self.l3 = nn.Linear(hidden_dim, output_dim)
#     def forward(self, state, action):
#
#         net_state, sfc_state = zip(*state)
#         net_states_list = list(net_state)
#         sfc_states_list = list(sfc_state)
#
#         net_state_list = [data.x for data in net_states_list]
#         net_states_list = torch.stack(net_state_list, dim=0)
#         net_states_list = torch.flatten(net_states_list, start_dim=1)
#
#         sfc_states_list = torch.stack(sfc_states_list, dim=0)
#         sfc_states_list = torch.flatten(sfc_states_list, start_dim=1)
#
#         state = torch.cat((net_states_list, sfc_states_list), dim=1)
#         x = torch.cat((state, action), dim=1)
#         x = self.l1(x)
#         x = self.l2(F.relu(x))
#         q = self.l3(F.relu(x))
#         return q