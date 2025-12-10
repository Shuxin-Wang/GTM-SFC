import networkx as nx
import environment
import random

class Greedy:
    def __init__(self, cfg, env, sfc_generator):
        self.env = env
        self.sfc_generator = sfc_generator

        self.node_used = self.env.node_used.copy()
        self.link_used = self.env.link_used.copy()
        self.link_occupied = self.env.link_occupied.copy()
        self.node_occupied = self.env.node_occupied.copy()
        self.power_consumption = self.env.power_consumption

        self.vnf = None
        self.vnf_exceeded_capacity = 0
        self.vnf_latency = 0
        self.vnf_exceeded_bandwidth = 0
        self.vnf_reliability = 1
        self.vnf_power_consumption = 0
        self.vnf_path = None
        self.vnf_placement = []
        self.vnf_reward = 0

        self.last_action = None

    def place_vnf(self, node):
        vnf = self.vnf
        if self.env.vnf_properties[vnf]['size'] <= self.env.node_properties[node]['capacity'] - self.env.node_used[node]:
            self.vnf_placement.append(1)
        else:
            self.vnf_placement.append(0)
        self.env.node_used[node] += self.env.vnf_properties[vnf]['size']

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
                self.vnf_latency += self.env.link_properties[index]['latency']
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
                exceeded_bandwidth = self.env.link_used[index] - self.env.link_properties[index]['bandwidth']
                if exceeded_bandwidth > self.vnf_exceeded_bandwidth:
                    self.vnf_exceeded_bandwidth = exceeded_bandwidth
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

    def save_state(self):
        self.node_used = self.env.node_used.copy()
        self.link_used = self.env.link_used.copy()
        self.link_occupied = self.env.link_occupied.copy()
        self.node_occupied = self.env.node_occupied.copy()
        self.power_consumption = self.env.power_consumption

    def load_state(self):
        self.env.node_used = self.node_used.copy()
        self.env.link_used = self.link_used.copy()
        self.env.link_occupied = self.link_occupied.copy()
        self.env.node_occupied = self.node_occupied.copy()
        self.env.power_consumption = self.power_consumption

    def select_action(self, vnf, dest=None):
        self.vnf = vnf

        self.save_state()

        best_node = None
        max_reward = float('-inf')

        for node in self.env.graph.nodes():
            node = int(node)
            if not dest:
                path = self.find_route(self.last_action, node)
            else:
                p1 = self.find_route(self.last_action, node)
                p2 = self.find_route(node, dest)
                if p1 is False or p2 is False:
                    path = False
                else:
                    path = p1[:-1] + p2

            self.place_vnf(node)
            self.compute_vnf_capacity(node)
            self.compute_vnf_latency(path)
            self.compute_vnf_bandwidth(path)
            self.compute_power(node)
            self.compute_reliability(node, path)

            reward = self.compute_vnf_reward()

            if reward > max_reward:
                best_node = node
                max_reward = reward

            self.load_state()
            self.clear_vnf_state()

        if best_node is None:
            best_node = int(next(iter(self.env.graph.nodes()))) # select default node to place vnf

        if not dest:
            final_path = self.find_route(self.last_action, best_node)
        else:
            p1 = self.find_route(self.last_action, best_node)
            p2 = self.find_route(best_node, dest)
            final_path = False if (p1 is False or p2 is False) else p1[:-1] + p2

        self.vnf = vnf
        self.place_vnf(best_node)
        self.compute_vnf_capacity(best_node)
        self.compute_vnf_latency(final_path)
        self.compute_vnf_bandwidth(final_path)
        self.compute_power(best_node)
        self.compute_reliability(best_node, final_path)

        self.last_action = best_node

        return best_node

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

        self.vnf_exceeded_capacity = 0
        self.vnf_latency = 0
        self.vnf_exceeded_bandwidth = 0
        self.vnf_reliability = 1
        self.vnf_power_consumption = 0

        self.last_action = None
        self.vnf_path = None
        self.vnf_placement.clear()
        self.vnf_reward = 0

class FirstFit:
    def __init__(self, cfg, env, sfc_generator):
        self.env = env
        self.sfc_generator = sfc_generator

        self.node_used = self.env.node_used.copy()
        self.link_used = self.env.link_used.copy()
        self.link_occupied = self.env.link_occupied.copy()
        self.node_occupied = self.env.node_occupied.copy()
        self.power_consumption = self.env.power_consumption

        self.vnf = None
        self.vnf_placement = []

        self.last_action = None

    def place_vnf(self, node):
        vnf = self.vnf
        if self.env.vnf_properties[vnf]['size'] <= self.env.node_properties[node]['capacity'] - self.env.node_used[node]:
            self.vnf_placement.append(1)
        else:
            self.vnf_placement.append(0)
        self.env.node_used[node] += self.env.vnf_properties[vnf]['size']

    def find_route(self, src, dest):
        src, dest = str(src), str(dest)
        if nx.has_path(self.env.graph, src, dest):
            return nx.shortest_path(self.env.graph, src, dest)
        else:
            return False

    def update_bandwidth(self, path):
        vnf = self.vnf
        for i in range(len(path) - 1):
            if int(path[i]) > int(path[i + 1]):
                index = self.env.link_index[(path[i + 1], path[i])]
            else:
                index = self.env.link_index[(path[i], path[i + 1])]
            self.env.link_used[index] += self.env.vnf_properties[vnf]['bandwidth']
            self.env.link_occupied[index] = 1

    def capacity_exceeded(self, node):
        vnf_exceeded_capacity = self.env.node_used[node] - self.env.node_properties[node]['capacity']
        if vnf_exceeded_capacity > 0:
            return True
        else:
            return False

    def bandwidth_exceeded(self, path):
        for i in range(len(path) - 1):
            if int(path[i]) > int(path[i + 1]):
                index = self.env.link_index[(path[i + 1], path[i])]
            else:
                index = self.env.link_index[(path[i], path[i + 1])]
            exceeded_bandwidth = self.env.link_used[index] - self.env.link_properties[index]['bandwidth']
            if exceeded_bandwidth > 0:
                return True
        return False

    def save_state(self):
        self.node_used = self.env.node_used.copy()
        self.link_used = self.env.link_used.copy()
        self.link_occupied = self.env.link_occupied.copy()
        self.node_occupied = self.env.node_occupied.copy()

    def load_state(self):
        self.env.node_used = self.node_used.copy()
        self.env.link_used = self.link_used.copy()
        self.env.link_occupied = self.link_occupied.copy()
        self.env.node_occupied = self.node_occupied.copy()

    def select_action(self, vnf, dest=None):
        self.vnf = vnf

        self.save_state()

        nodes = list(self.env.graph.nodes())
        max_iteration = len(nodes)
        fit_node = None
        final_path = None

        for _ in range(max_iteration):
            node = int(random.choice(nodes))
            if not dest:
                path = self.find_route(self.last_action, node)
            else:
                p1 = self.find_route(self.last_action, node)
                p2 = self.find_route(node, dest)
                if p1 is False or p2 is False:
                    path = False
                else:
                    path = p1[:-1] + p2

            if not path:
                continue

            self.place_vnf(node)
            self.update_bandwidth(path)
            if not self.capacity_exceeded(node) and not self.bandwidth_exceeded(path):
                fit_node = node
                final_path = path
                break

            self.load_state()

        if fit_node is None:
            self.load_state()
            fit_node = int(next(iter(self.env.graph.nodes())))  # choose default node to place VNF
            if not dest:
                final_path = self.find_route(self.last_action, fit_node)
            else:
                p1 = self.find_route(self.last_action, fit_node)
                p2 = self.find_route(fit_node, dest)
                if p1 is False or p2 is False:
                    final_path = []  # 或 None，但要确保 update_bandwidth 能处理
                else:
                    final_path = p1[:-1] + p2

            self.place_vnf(fit_node)
            self.update_bandwidth(final_path)

        self.last_action = fit_node

        return fit_node

    def test(self, env, sfc_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):
            env.clear_sfc()
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
