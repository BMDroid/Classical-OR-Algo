'''
@Description: TSP class
@Author: BO Jianyuan
@Date: 2020-02-27 19:12:08
LastEditors: BO Jianyuan
LastEditTime: 2021-01-28 17:30:19
'''

import math
import numpy as np


class TSP(object):
    def __init__(self, tsp, content=[]):
        def _content_to_coords(content):
            coords = {}
            for node in content:
                if len(node) == 3:
                    coords[int(node[0])] = (np.float(node[1]),
                                            np.float(node[2]))
            return coords

        def _update_coords(coords):
            keys = list(coords.keys())
            for key in keys:
                coords[key - 1] = coords.pop(key)
            return coords

        if tsp:
            if tsp.edge_weight_type == "EUC_2D" or tsp.edge_weight_type is None:  # euclidean distance instance with coords provided
                self._name = tsp.name
                self._coords = _update_coords(tsp.node_coords)
                self._nodes = self._coords.keys()
                self._g = self.get_distance()
                self._no_nodes = tsp.dimension
                self._no_edges = tsp.dimension * (tsp.dimension - 1) / 2
                self._path = []
                self._cost = None
            elif tsp.edge_weight_type == "EXPLICIT":
                self._name = tsp.name
                self._coords = None
                self._nodes = list(range(1, tsp.dimension + 1))
                self._g = self.get_distance(tsp.edge_weight_format,
                                            tsp.edge_weights)
                self._no_nodes = tsp.dimension
                self._no_edges = tsp.dimension * (tsp.dimension - 1) / 2
                self._path = []
                self._cost = None
        else:
            self._name = ""
            self._coords = _content_to_coords(content)
            self._nodes = self._coords.keys()
            self._g = self.get_distance()
            self._no_nodes = len(self._nodes)
            self._no_edges = len(self._nodes) * (len(self._nodes) - 1) / 2
            self._perturbed_g = self.get_perturbed(True)
            self._removed_diag = self.get_removed_diag()
            self._min_edge = self._removed_diag[self._removed_diag != 0].min()
            self._max_edge = self._removed_diag.max()

    def get_distance(self, edge_weight_format=None, edge_weights=None):
        if self._coords:
            return np.array([[
                int(
                    math.sqrt(
                        (self._coords[start][0] - self._coords[end][0])**2 +
                        (self._coords[start][1] - self._coords[end][1])**2) +
                    .5) for start in self._nodes
            ] for end in self._nodes])
        elif edge_weight_format == "UPPER_ROW":
            upper_weights = edge_weights
            upper_weights.append([])
            weights = [[0] * (i + 1) + lst
                       for i, lst in enumerate(upper_weights)]
            weights = np.array(weights)
            weights = weights + weights.T
            return weights
        elif edge_weight_format == "LOWER_DIAG_ROW":
            lower_weights = sum(edge_weights, [])
            lower_weights.insert(0, 0)
            k = sum(1 for x in lower_weights if x > 0)
            n = int((1 + math.sqrt(1 + 4 * (2 * k))) / 2)
            weights = []
            for i in range(n):
                weights.insert(0, [0] * (n - i - 1) + lower_weights[0:i + 1])
                lower_weights = lower_weights[i + 1:]
            weights = np.array(weights)
            weights = weights + weights.T
            return weights
        else:  # full matrix
            return np.array(edge_weights)

    def get_greedy_heuristic(self):
        graph = self._perturbed_g
        # https://github.com/theyusko/tsp-heuristics/blob/master/algo/nearest_neighbor.py
        node_no = graph.shape[0]
        min_distance = np.zeros(
            (node_no, ),
            dtype=float)  # distances with starting node as min_distance[i]
        travel_route = [[0 for x in range(0, node_no)]
                        for y in range(0, node_no)]
        # Step 1
        for start_node in range(0, node_no):
            # Step 3
            unvisited = np.ones((node_no, ),
                                dtype=int)  # all nodes are unvisited
            unvisited[start_node] = 0
            travel_route[start_node][
                0] = start_node  # travel route starts with start_node
            node = start_node
            iteration = 1
            while TSP.check_unvisited_node(unvisited) and iteration < node_no:
                # Step 2
                closest_arc = float('inf')
                closest_node = node_no
                for node2 in range(0, node_no):
                    if unvisited[node2] == 1 and 0 < graph[node][
                            node2] < closest_arc:
                        closest_arc = graph[node][node2]
                        closest_node = node2
                if closest_node >= node_no:
                    min_distance[start_node] = float('inf')
                    break
                node = closest_node
                unvisited[node] = 0
                min_distance[
                    start_node] = min_distance[start_node] + closest_arc
                travel_route[start_node][iteration] = node
                iteration = iteration + 1
            if not math.isinf(min_distance[start_node]):
                last_visited = travel_route[start_node][node_no - 1]
                if graph[last_visited][start_node] > 0:
                    min_distance[start_node] = min_distance[
                        start_node] + graph[last_visited][start_node]
                else:
                    min_distance[start_node] = float('inf')
        [shortest_min_distance,
         shortest_travel_route] = TSP.find_best_route(node_no, travel_route,
                                                      min_distance)
        return shortest_min_distance, shortest_travel_route

    def get_two_opt(self, graph, route):
        # graph = self._perturbed_g
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    if TSP.cost_change(graph, best[i - 1], best[i],
                                       best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
        dist = 0
        for i in range(len(route)):
            dist += graph[route[i]][route[(i + 1) % len(route)]]
        return dist, best

    @staticmethod
    def find_best_route(node_no, travel_route, min_distance):
        shortest_travel_route = travel_route[0]
        shortest_min_distance = min_distance.item(0)
        for start_node in range(0, node_no):
            if min_distance[start_node] < shortest_min_distance:
                shortest_min_distance = min_distance.item(start_node)
                shortest_travel_route = travel_route[start_node]
        return shortest_min_distance, shortest_travel_route

    @staticmethod
    def check_unvisited_node(unvisited):
        for u in unvisited:
            if u == 1:
                return True
        return False

    @staticmethod
    def cost_change(graph, n1, n2, n3, n4):
        return graph[n1][n3] + graph[n2][n4] - graph[n1][n2] - graph[n3][n4]