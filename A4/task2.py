import sys
import time
from collections import defaultdict, Counter


class Node:
    def __init__(self, id):
        self.id = id
        self.degree = 0
        self.neighbors = set()

    def inc_degree(self):
        self.degree += 1


def get_input():
    run = 'non'
    if run == 'local':
        ip = 'power_input.txt'
        op1 = 'output1.txt'
        op2 = 'output2.txt'
    else:
        ip = sys.argv[1]
        op1 = sys.argv[2]
        op2 = sys.argv[3]
    return ip, op1, op2


def preprocess(path):
    edge_count = 0
    node_map = {}
    with open(path) as fp:
        for line in fp.readlines():
            edge_count += 1
            src, dst = line.split()
            if src not in node_map:
                node_map[src] = Node(src)
            if dst not in node_map:
                node_map[dst] = Node(dst)
            src_node = node_map[src]
            dst_node = node_map[dst]
            src_node.inc_degree()
            dst_node.inc_degree()
            src_node.neighbors.add(dst_node)
            dst_node.neighbors.add(src_node)
    return node_map, edge_count


def get_partial_edge_betweenness(head, node_map):
    """
    Edge betweenness for a node
    :param head: A node in the graph
    :param node_map: node_id -> Node
    :return: Betweenness from the head
    """

    for node in node_map.values():
        # node.val is calculated from it's child nodes while going up
        node.val = 1
        # node.shortest path is populated while performing BFS from head to leaf
        node.shortest_paths = 0

    head.shortest_paths = 1

    # Child may have more than 1 parent
    parents_map = defaultdict(set)
    level_cur = {head}
    level_next = set()
    levels = [{head}]
    visited = {head}
    while level_cur:
        head = level_cur.pop()
        for child in head.neighbors:
            if child not in visited:
                level_next.add(child)
                visited.add(child)
        if not level_cur:
            if level_next:
                for child in level_next:
                    for node in child.neighbors:
                        if node in levels[-1]:
                            parents_map[child].add(node)
                            child.shortest_paths += node.shortest_paths
                levels.append(level_next.copy())

            level_cur, level_next = level_next, level_cur

    edge_value = {}

    for level_idx in range(len(levels)-1, -1, -1):
        for node in levels[level_idx]:
            parents = parents_map[node]
            for parent in parents:
                edge_tuple = min(node.id, parent.id), max(node.id, parent.id)
                temp = node.val * (parent.shortest_paths/node.shortest_paths)
                edge_value[edge_tuple] = temp
                parent.val += temp

    return edge_value


def get_edge_betweenness(node_map):
    edge_map = Counter()
    for node in node_map.values():
        node_edge_map = get_partial_edge_betweenness(node, node_map)
        edge_map.update(node_edge_map)
    for edge in edge_map:
        edge_map[edge] /= 2
    return edge_map


def get_modularity(node_map, m, removed_edges):
    """
    :return: Modularity for all communities, communities
    """
    remaining_nodes = set(node_map.values())
    communities = []

    while remaining_nodes:
        communities.append([])
        community_head = remaining_nodes.pop()
        queue = [community_head]
        visited = {community_head}

        while queue:
            node = queue.pop()
            communities[-1].append(node)
            for child in node.neighbors:
                if child not in visited:
                    queue.append(child)
                    visited.add(child)

                    remaining_nodes.remove(child)
    modularity = 0

    for community in communities:
        sorted_community = sorted(community, key=lambda x: x.id)
        total = 0
        # pairs = combinations(community, 2)
        # for src, dst in pairs:
        for src in sorted_community:
            for dst in sorted_community:
                if dst in src.neighbors or (src, dst) in removed_edges or (dst, src) in removed_edges:
                    temp = 1 - (1 / (2 * m)) * src.degree * dst.degree
                else:
                    temp = -(1 / (2 * m)) * src.degree * dst.degree
                total += temp

        modularity += total

    return modularity/(2*m), communities


def process(input_path, output_betweenness, output):
    all_removed_edges = set()
    node_map, m = preprocess(input_path)
    edge_map = get_edge_betweenness(node_map)

    with open(output_betweenness, 'w') as fp:
        for a, b in sorted(edge_map.items(), key=lambda x: (-x[1], x[0])):
            fp.write(str(a) + ', ' + str(b) + '\n')

    for node1 in node_map.values():
        for node2 in node_map.values():
            if node2 in node1.neighbors and node1 not in node2.neighbors:
                print(node1, node2)

    max_modularity = 0
    best_communities = None

    for i in range(m):
        edge_src, edge_dst = edge_map.most_common(1)[0][0]
        edge_src_node = node_map[edge_src]
        edge_dst_node = node_map[edge_dst]
        edge_src_node.neighbors.remove(edge_dst_node)
        edge_dst_node.neighbors.remove(edge_src_node)
        all_removed_edges.add((edge_src_node, edge_dst_node))

        modularity, communities = get_modularity(node_map, m, all_removed_edges)
        if modularity > max_modularity:
            max_modularity = modularity
            best_communities = communities[::]

        edge_map = get_edge_betweenness(node_map)

    for i in range(len(best_communities)):
        best_communities[i] = sorted([node.id for node in best_communities[i]])

    best_communities.sort(key = lambda x: (len(x), x))
    with open(output, 'w') as fp:
        for row in best_communities:
            fp.write("'" + "', '".join(row) + "'\n")


if __name__ == '__main__':
    time_start = time.time()
    input_file_path, betweenness_output_path, output_file_path = get_input()
    process(input_file_path, betweenness_output_path, output_file_path)
    time_end = time.time()
    print("Duration {}".format(time_end-time_start))