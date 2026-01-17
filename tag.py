import json
import pickle
import string
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from nltk.corpus import stopwords
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def bfs_edges(
    G,
    source: int = 0,
    reverse: bool = False,
    depth_limit: Optional[int] = None,
    sort_neighbors_key: Optional[str] = None,
) -> Iterable:
    """
    Perform a breadth-first search (BFS) on a graph and yield edges.

    Args:
        G: The graph to search.
        source: The starting node for the search.
        reverse: If True and G is directed, traverse predecessors instead of successors.
        depth_limit: Maximum depth to search (default: None, which means no limit).
        sort_neighbors_key: If provided, sort neighbors based on this node attribute.

    Yields:
        Tuples of (parent, child, depth) for each edge traversed in the BFS.
    """
    # Determine the appropriate neighbor function based on graph type and direction
    if reverse and G.is_directed():
        neighbors = G.predecessors
    else:
        neighbors = G.neighbors

    # Define a function to get and optionally sort neighbors
    def get_neighbors(node: Any) -> Iterable:
        if sort_neighbors_key:
            return sorted(
                neighbors(node),
                key=lambda x: G.nodes[x].get(sort_neighbors_key, 0),
                reverse=True,
            )
        else:
            return neighbors(node)

    # Initialize BFS
    visited = {source}
    queue = deque([(source, 1)])  # (node, depth)

    while queue:
        parent, depth = queue.popleft()

        if depth_limit is not None and depth >= depth_limit:
            break

        for child in get_neighbors(parent):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))
                yield parent, child, depth


def neighborhood_corpus(source: int, subgraph: nx.Graph, all_node_text: Dict[str, str]) -> str:
    """
    Generate the neighborhood corpus based on a breadth-first search.

    Args:
        source (int): The source node ID
        subgraph (nx.Graph): The NetworkX subgraph
        all_node_text (Dict[str, str]): Dictionary mapping node IDs to their text content

    Returns:
        str: A formatted string representing the neighborhood corpus
    """
    bfs_edges_list: List[Tuple[int, int, int]] = list(bfs_edges(subgraph, source=source))
    mapping: Dict[int, str] = {source: "[Root]"}  # Node id -> Node version
    prompt_dict: Dict[str, str] = {}  # Node version -> Node prompt

    # Generate initial mapping and prompt dictionary
    old_depth = 0
    cur_depth_count: Dict[str, int] = {}

    for parent, child, depth in bfs_edges_list:
        if depth != old_depth:
            cur_depth_count = {}
            old_depth = depth

        cur_depth_count[parent] = cur_depth_count.get(parent, 0) + 1

        if depth == 1:
            mapping[child] = f"{cur_depth_count[parent]}"
        else:
            mapping[child] = f"{mapping[parent]}.{cur_depth_count[parent]}"

        prompt_dict[mapping[child]] = f"Node-{mapping[child]}: {all_node_text[child]}\n"

    # Generate the final prompt
    reverse_mapping = {v: k for k, v in mapping.items()}  # Node version -> Node id
    visited_sections: Set[int] = {source}
    prompt = f"ROOT: {all_node_text[source]}\n"

    for key in sorted(
        prompt_dict.keys(),
        # sorted by version number
        key=lambda version: [int(part) for part in version.split(".")],
    ):
        child = reverse_mapping[key]
        visited_sections.add(child)

        predecessors = set(subgraph.predecessors(child))
        visited_pred = predecessors & visited_sections

        parent_key = ".".join(key.split(".")[:-1])
        if parent_key:
            parent = reverse_mapping[parent_key]
            visited_pred.discard(parent)  # discard direct predecessor if parent exists

        visited_pred.discard(source)  # always discard source node

        prompt += prompt_dict[key]

        if visited_pred:
            cited_nodes = sorted(
                [mapping[node] for node in visited_pred],
                key=lambda version: [int(part) for part in version.split(".")],
            )
            cited_sections = ", ".join(["Node-" + node for node in cited_nodes])

            prompt += f"(This node also have linkage with {cited_sections})\n"

    return prompt


class TAG:
    def __init__(self):
        self.nodes = []
        self.edges = [[], []]
        self.prediction = None
        self.ground_truth = None

    def set_labels(self, prediction, ground_truth):
        self.prediction = prediction
        self.ground_truth = ground_truth

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, src, dst):
        self.edges[0].append(src)
        self.edges[1].append(dst)

    def save(self, path):
        # write object to a file
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        # load object from a file
        with open(path, "rb") as f:
            return pickle.load(f)

    def to_filtered_networkx(self, threshold_range=None, keep_empty_nodes=True):
        # construct a full networkx graph
        data = Data(
            num_nodes=len(self.nodes),
            edge_index=torch.tensor(self.edges, dtype=torch.long),
            is_undirected=False,
        )
        G = to_networkx(data)

        # filter out nodes that do not have features within the threshold range
        # unless keep_empty_nodes is True
        for node_index, node in enumerate(self.nodes):
            if not keep_empty_nodes and not node.has_feature_within_range(threshold_range):
                G.remove_node(node_index)

        return G

    def get_importance_scores(self):
        scores = []
        for node in self.nodes:
            scores.extend(node.get_valid_scores())

        return np.array(scores)

    def text(self, p_range=None, style="document", **kwargs):
        threshold_range = (
            percentage_to_range(data=self.get_importance_scores(), percentage_range=p_range)
            if p_range is not None
            else None
        )
        # print("threshold_range:", threshold_range)

        if style == "document":
            # convert graph into a document-style corpus
            G = self.to_filtered_networkx(threshold_range)
            all_node_text = [node.to_string(threshold_range, **kwargs) for node in self.nodes]

            return neighborhood_corpus(source=0, subgraph=G, all_node_text=all_node_text)
        elif style == "json":
            # directly convert graph into a json-style document
            corpus_dict = {}
            valid_nodes = []  # nodes has features with importance score within the threshold range

            for node_index, node in enumerate(self.nodes):
                # keep all nodes even if they do not have features within the threshold range
                valid_nodes.append(node_index)
                corpus_dict[str(node_index)] = {
                    "neighbors": [],
                    "node_features": node.to_string(threshold_range, **kwargs),
                }

            for src, dst in zip(*self.edges):
                if src in valid_nodes:
                    corpus_dict[str(src)]["neighbors"].append(str(dst))

            return json.dumps(corpus_dict)

    def structure(self, p_range=None):
        threshold_range = (
            percentage_to_range(data=self.get_importance_scores(), percentage_range=p_range)
            if p_range is not None
            else None
        )
        G = self.to_filtered_networkx(threshold_range)
        return G

    def get_masked_tokens(self, p_range=None, return_scores=False):
        # tokens whose importance scores are not in p_range are masked
        threshold_range = (
            percentage_to_range(data=self.get_importance_scores(), percentage_range=p_range)
            if p_range is not None
            else None
        )

        masked_tokens = []
        importance_scores = []
        for node in self.nodes:
            m, s = node.get_masked_tokens(threshold_range)
            masked_tokens.extend(m)
            importance_scores.extend(s)
        if return_scores:
            return masked_tokens, importance_scores

        return masked_tokens


class Node:
    def __init__(self):
        self.features = []
        self.importance_scores = []

        self.tmp_subwords_counter = 0

    def append(self, token, score):
        self.features.append(token)
        self.importance_scores.append(score)
        self.tmp_subwords_counter = 1

    def concat(self, token, score):
        self.features[-1] += token

        self.importance_scores[-1] *= self.tmp_subwords_counter
        self.importance_scores[-1] += score
        self.tmp_subwords_counter += 1
        self.importance_scores[-1] /= self.tmp_subwords_counter

    def has_feature_within_range(self, threshold_range):
        if threshold_range is None:
            return True
        return any(score >= threshold_range[0] and score <= threshold_range[1] for score in self.importance_scores)

    @staticmethod
    def is_valid(token):
        try:
            return token not in stopwords.words("english") and token not in string.punctuation
        except LookupError:
            import nltk
            nltk.download('stopwords')
            return token not in stopwords.words("english") and token not in string.punctuation

    def get_valid_scores(self):
        scores = []
        for token, score in zip(self.features, self.importance_scores):
            if self.is_valid(token):
                scores.append(score)
        return scores

    def get_masked_tokens(self, threshold_range=None):
        if threshold_range is None:
            threshold_range = [
                min(self.importance_scores),
                max(self.importance_scores) + 1,
            ]

        masked_tokens = []
        importance_scores = []

        for token, score in zip(self.features, self.importance_scores):
            if not self.is_valid(token):
                continue

            if score < threshold_range[0] or score > threshold_range[1]:
                masked_tokens.append(token)
                importance_scores.append(score)

        return masked_tokens, importance_scores

    def to_string(
        self,
        threshold_range=None,
        splitter=" ",
        masker="[MASK]",
        merge=False,
        with_score=True,
        p_range=None,
    ):
        if threshold_range is None and p_range is None:
            threshold_range = [
                min(self.importance_scores),
                max(self.importance_scores) + 1,
            ]

        if p_range is not None and threshold_range is None:
            threshold_range = (
                percentage_to_range(data=self.importance_scores, percentage_range=p_range)
                if p_range is not None
                else None
            )

        if with_score:
            token_template = "{token}({score:.2f})"
        else:
            token_template = "{token}"

        result = ""
        is_last_token_masker = False

        for token, score in zip(self.features, self.importance_scores):
            if not self.is_valid(token):
                continue

            if score >= threshold_range[0] and score <= threshold_range[1]:
                result += token_template.format(token=token, score=score) + splitter
                is_last_token_masker = False
            elif masker != "":
                if not merge or (merge and not is_last_token_masker):
                    result += masker + splitter
                    is_last_token_masker = True

        return result


# helper function
def percentage_to_range(data, percentage_range):
    """
    Convert a percentage range to a threshold range.
    """
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as exc:
            raise ValueError(
                'data must be a numpy array or an array-like object that can be converted to a numpy array'
            ) from exc

    lower_bound = np.percentile(data, percentage_range[0])
    upper_bound = np.percentile(data, percentage_range[1])
    return (lower_bound, upper_bound)
