import networkx as nx
import environment

class Greedy:
    def __init__(self, cfg, env, sfc_generator):
        self.env = env
        self.sfc_generator = sfc_generator

        self.vnf = None
        self.action = None

        self.vnf_exceeded_capacity = 0
        self.vnf_latency = 0
        self.vnf_exceeded_bandwidth = 0
        self.vnf_reliability = 1
        self.vnf_power_consumption = 0

        self.last_action = None
        self.vnf_path = None
        self.vnf_placement = []
        self.vnf_reward = 0
        self.max_reward = float('-inf')

    def place_vnf(self, node):
        vnf = self.vnf
        self.env.node_used[node] += self.env.vnf_properties[vnf]['size']
        if self.env.vnf_properties[vnf]['size'] <= self.env.node_properties[node]['capacity'] - self.env.node_used[node]:
            self.vnf_placement.append(1)
        else:
            self.vnf_placement.append(0)

    def find_route(self, src, dest):
        src, dest = str(src), str(dest)
        if nx.has_path(self.env.graph, src, dest):
            return nx.shortest_path(self.env.graph, src, dest)
        else:
            return False

    def compute_vnf_capacity(self, node):
        self.vnf_exceeded_capacity = self.env.node_used[node] - self.env.node_properties[node]['capacity']

    def compute_vnf_latency(self, path):
        if path:
            for i in range(len(path) - 1):
                if int(path[i]) > int(path[i + 1]):
                    index = self.env.link_index[(path[i + 1], path[i])]
                else:
                    index = self.env.link_index[(path[i], path[i + 1])]
                self.vnf_latency = self.env.link_properties[index]['latency']
        else:
            self.vnf_latency = self.env.path_penalty

    def compute_vnf_bandwidth(self, path):
        vnf = self.vnf
        if path:
            for i in range(len(path) - 1):
                if int(path[i]) > int(path[i + 1]):
                    index = self.env.link_index[(path[i + 1], path[i])]
                else:
                    index = self.env.link_index[(path[i], path[i + 1])]
                self.env.link_used[index] += self.env.vnf_properties[vnf]['bandwidth']
                self.env.link_occupied[index] = 1
                self.vnf_exceeded_bandwidth = self.env.link_used[index] - self.env.link_properties[index]['bandwidth']
        else:
            self.vnf_exceeded_bandwidth = self.env.path_penalty

    def compute_power(self, node):
        vnf = self.vnf
        if self.env.node_occupied[node]:
            self.vnf_power_consumption = self.env.p_unit * environment.VNF_SIZE[vnf]
        else:
            self.vnf_power_consumption = self.env.p_min + self.env.p_unit * environment.VNF_SIZE[vnf]
            self.env.node_occupied[node] = 1
        self.env.power_consumption += self.env.p_link * self.vnf_latency

    def compute_reliability(self, node, path):
        self.vnf_reliability *= self.env.node_properties[node]['reliability']
        if path:
            for i in range(len(path) - 1):
                if int(path[i]) > int(path[i + 1]):
                    index = self.env.link_index[(path[i + 1], path[i])]
                else:
                    index = self.env.link_index[(path[i], path[i + 1])]
                self.vnf_reliability *= self.env.link_properties[index]['reliability']
        else:
            pass

    def compute_vnf_reward(self):
        vnf = self.vnf
        vnf_reward = self.env.vnf_properties[vnf]['size'] \
                         * (self.env.vnf_properties[vnf]['bandwidth'] / 20) \
                         / (self.env.vnf_properties[vnf]['latency'] / 20)

        self.vnf_reward += self.env.lambda_placement * self.vnf_placement[-1] * vnf_reward  # vnf placement reward

        self.vnf_reward -= self.env.lambda_power * self.vnf_power_consumption    # vnf power penalty

        self.vnf_reward -= self.env.lambda_capacity * self.vnf_exceeded_capacity    # vnf exceeded capacity penalty

        self.vnf_reward -= self.env.lambda_latency * self.vnf_latency   # vnf latency penalty

        self.vnf_reward -= self.env.lambda_bandwidth * self.vnf_exceeded_bandwidth  # vnf exceeded bandwidth penalty

        self.vnf_reward += self.env.lambda_reliability * self.vnf_reliability    # vnf reliability penalty

        return self.vnf_reward

    def select_action(self, vnf, dest=None):
        self.vnf = vnf
        for node in self.env.graph.nodes():
            node = int(node)
            if not dest:
                path = self.find_route(self.last_action, node)
            else:
                path = self.find_route(self.last_action, node)[:-1] + self.find_route(node, dest)

            self.place_vnf(node)
            self.compute_vnf_capacity(node)
            self.compute_vnf_latency(path)
            self.compute_vnf_bandwidth(path)
            self.compute_power(node)
            self.compute_reliability(node, path)

            reward = self.compute_vnf_reward()
            if reward > self.max_reward:
                self.action = node
                self.max_reward = reward

            self.clear_vnf_state()

        self.last_action = self.action
        self.action = None
        self.max_reward = float('-inf')

        return self.last_action

    def test(self, env, sfc_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):
            env.clear_sfc()
            self.clear_sfc_state()
            source_dest_node_pair = source_dest_node_pairs[i]
            reliability_requirement = reliability_requirement_list[i]
            placement = []
            self.last_action = int(source_dest_node_pair[0])
            self.vnf_placement.clear()
            for vnf in sfc_list[i][:-1]:
                placement.append(self.select_action(vnf))
            placement.append(self.select_action(sfc_list[i][-1], dest=int(source_dest_node_pair[1])))
            sfc = source_dest_node_pair.to(dtype=int).tolist() + sfc_list[i] + reliability_requirement.tolist()
            env.step(sfc, placement)

    def clear_vnf_state(self):
        self.vnf_exceeded_capacity = 0
        self.vnf_latency = 0
        self.vnf_exceeded_bandwidth = 0
        self.vnf_reliability = 1
        self.vnf_power_consumption = 0

        self.vnf_path = None
        self.vnf_reward = 0

    def clear_sfc_state(self):
        self.vnf = None
        self.action = None

        self.vnf_exceeded_capacity = 0
        self.vnf_latency = 0
        self.vnf_exceeded_bandwidth = 0
        self.vnf_reliability = 1
        self.vnf_power_consumption = 0

        self.last_action = None
        self.vnf_path = None
        self.vnf_placement.clear()
        self.vnf_reward = 0
        self.max_reward = float('-inf')
