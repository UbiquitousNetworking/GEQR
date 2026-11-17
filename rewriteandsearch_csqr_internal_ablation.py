import os
import pickle
import time
import argparse
import random
import networkx as nx
import numpy as np
# from numpy import *
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
from index_tree_schema import KaryTree, TreeNode
import torch.nn as nn
# 浣犲凡鏈夌殑宸ュ叿鍑芥暟
from utils import (
    f1_score_calculation,
    evaluation
)

# ===================== 鍙傛暟瑙ｆ瀽 =====================
# DEFAULT_TREE_BASE_NAMES = ("louvain", "kclique", "leiden")
DEFAULT_TREE_BASE_NAMES = ("louvain","leiden", "kclique")
DEFAULT_TREE_VARIANT_COUNT = 10


def build_default_candidate_tree_names():
    names = list(DEFAULT_TREE_BASE_NAMES)
    for variant_idx in range(1, DEFAULT_TREE_VARIANT_COUNT + 1):
        names.extend(f"{base_name}{variant_idx}" for base_name in DEFAULT_TREE_BASE_NAMES)
    return ",".join(names)


def parse_candidate_tree_names(candidate_tree_names):
    if not candidate_tree_names:
        candidate_tree_names = build_default_candidate_tree_names()
    return [
        name.strip()
        for name in candidate_tree_names.split(",")
        if name.strip()
    ]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='texas')
    parser.add_argument('--embedding_tensor_name', type=str)
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/')
    parser.add_argument('--queryfilenum', type=int, default=2)
    parser.add_argument('--query_file', type=str, default=None,
                        help='optional query file override; ground truth construction is unchanged')
    parser.add_argument('--query_artifact_output', type=str, default=None,
                        help='optional .npz path for per-query predictions and gate features')
    parser.add_argument('--alpha', type=float, default=0.3, help='semantic rewrite weight')
    parser.add_argument('--belta', type=float, default=0.6, help='structure rewrite weight')
    parser.add_argument('--semrewrite', type=int, default=1)
    parser.add_argument('--structrewrite', type=int, default=1)
    parser.add_argument('--embtype', type=str, default='meanembedding', choices=['meanembedding', 'medoidembedding', 'centroidembedding'])
    parser.add_argument('--candidate_tree_names', type=str, default=build_default_candidate_tree_names())
    parser.add_argument('--candidate_tree_dir', type=str, default=None)
    parser.add_argument('--candidate_tree_merge', type=str, default='ranked_consensus', choices=['adaptive', 'union', 'intersection', 'consensus', 'ranked_consensus'])
    parser.add_argument('--candidate_consensus_min', type=int, default=2)
    parser.add_argument('--context_max_nodes', type=int, default=96,
                        help='maximum non-query nodes retained after node-level ranking; <=0 disables the cap')
    parser.add_argument('--context_reject_above', type=int, default=256,
                        help='reject context before truncation when the consensus pool exceeds this size; <=0 disables')
    parser.add_argument('--context_min_parent_margin', type=float, default=0.06,
                        help='reject context when mean selected-split cosine margin is below this value')
    parser.add_argument('--context_min_tree_agreement', type=float, default=-1.0,
                        help='reject context when mean pairwise Jaccard agreement between trees is below this value')
    parser.add_argument('--candidate_similarity_weight', type=float, default=0.25,
                        help='weight of query cosine in node-level consensus ranking')
    parser.add_argument('--merged_centrality', type=str, default='degree', choices=['degree', 'closeness'])
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--uncertainty_gamma', type=float, default=0.0)
    parser.add_argument('--topkvalue', type=int, default=5, help='number of representative nodes used for structure rewrite')
    parser.add_argument('--representative_relevance_weight', '--lambda_rep', type=float, default=0.5,
                        help='lambda in representative scoring: lambda * query cosine + (1-lambda) * normalized centrality; must be in [0,1]')
    parser.add_argument('--representative_center', type=str, default='median', choices=['mean', 'median'],
                        help='robust aggregation used for the selected representative nodes')
    parser.add_argument('--fusion_mode', type=str, default='normalized', choices=['normalized', 'sequential'])
    parser.add_argument('--query_weight', type=float, default=0.5)
    parser.add_argument('--context_weight', type=float, default=0.25)
    parser.add_argument('--representative_weight', type=float, default=0.25)
    parser.add_argument(
        '--context_provider', type=str, default='csqr',
        choices=['none', 'random', '1hop', '2hop', 'ppr', 'csqr'],
        help='provider used to construct the query context; all downstream steps are unchanged'
    )
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--ppr_alpha', type=float, default=0.85)
    parser.add_argument('--ppr_tol', type=float, default=1.0e-6)
    parser.add_argument('--ppr_max_iter', type=int, default=100)
    parser.add_argument(
        '--context_score_terms', type=str, default='size,relevance,coherence',
        help='comma-separated subset of size,relevance,coherence; active terms are re-averaged'
    )
    args = parser.parse_args()
    if not 0.0 <= args.representative_relevance_weight <= 1.0:
        parser.error('--representative_relevance_weight must be in [0,1]')
    return args

def _unique_preserve_order(seq):
    """鎸夊嚭鐜伴『搴忎繚鐣欏敮涓€鍏冪礌"""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def ESGSearch_CandidatePruned(
    query_index,
    graph_score,
    G,
    candidate_nodes=None,
    uncertainty=None,
    gamma=0.0,
    tau=0.5,
):
    scores = np.asarray(graph_score, dtype=float)
    n = scores.size
    qset = set(int(q) for q in query_index)

    # Exact fast path for the production setting used by the DBLP experiments:
    # one query node, the full graph as candidate pool, and no uncertainty
    # penalty. Stable NumPy sorting preserves the original node-id tie order;
    # the vectorized prefix objective replaces an O(n) Python dictionary plus
    # an O(n) Python evaluation loop for every query.
    if (
        len(qset) == 1
        and (candidate_nodes is None or len(candidate_nodes) == 0)
        and (uncertainty is None or len(uncertainty) == 0)
        and float(gamma) == 0.0
    ):
        query_node = next(iter(qset))
        if not (0 <= query_node < n):
            return [query_node], 0, float("-inf")
        ranked = np.argsort(-scores, kind="stable")
        ranked = ranked[ranked != query_node]
        # cumsum is sequential here, matching the original prefix updates.
        ranked_prefix = np.cumsum(scores[ranked], dtype=np.float64)
        prefix_sum = float(scores[query_node]) + ranked_prefix
        sizes = np.arange(2, n + 1, dtype=np.float64)
        total_sum = float(np.cumsum(scores, dtype=np.float64)[-1]) if n else 0.0
        total_mean = total_sum / max(1, n)
        esg_after_add = (prefix_sum - total_mean * sizes) / np.power(sizes, tau)
        esg_at_query = (float(scores[query_node]) - total_mean) / (1.0 ** tau)
        best_after = int(np.argmax(esg_after_add)) if esg_after_add.size else -1
        if best_after >= 0 and float(esg_after_add[best_after]) > esg_at_query:
            best_k = best_after + 1
            best_nodes = [query_node] + ranked[:best_k].astype(int).tolist()
        else:
            best_k = 0
            best_nodes = [query_node]
        best_threshold = min(float(scores[v]) for v in best_nodes)
        return best_nodes, best_k, best_threshold

    if candidate_nodes is None or len(candidate_nodes) == 0:
        candidate_nodes = range(n)
    pool = _unique_preserve_order([int(v) for v in candidate_nodes if 0 <= int(v) < n] + list(qset))
    if not pool:
        return list(qset), 0, float("-inf")

    if uncertainty is None:
        uncertainty = {}
    calibrated = {}
    for v in pool:
        calibrated[v] = float(scores[v]) - float(gamma) * float(uncertainty.get(v, 0.0))

    total_sum = sum(calibrated[v] for v in pool)
    total_mean = total_sum / max(1, len(pool))
    ranked = sorted((v for v in pool if v not in qset), key=lambda v: calibrated[v], reverse=True)

    best_nodes = list(qset)
    best_esg = -float("inf")
    best_k = 0
    prefix_nodes = list(qset)
    prefix_sum = sum(calibrated[v] for v in qset if v in calibrated)

    def evaluate(k, nodes, score_sum):
        size = len(nodes)
        if size <= 0:
            return
        esg = (score_sum - total_mean * size) / (size ** tau)
        nonlocal best_nodes, best_esg, best_k
        if esg > best_esg:
            best_esg = esg
            best_k = k
            best_nodes = list(nodes)

    evaluate(0, prefix_nodes, prefix_sum)
    for k, v in enumerate(ranked, start=1):
        prefix_nodes.append(v)
        prefix_sum += calibrated[v]
        evaluate(k, prefix_nodes, prefix_sum)

    if best_esg == -float("inf"):
        return list(qset), 0, float("-inf")
    best_threshold = min(calibrated[v] for v in best_nodes) if best_nodes else float("-inf")
    return best_nodes, best_k, best_threshold


def ESGSearch_FullGraphSingleQueryBatch(query_nodes, graph_scores, tau=0.5):
    """Vectorized equivalent of the full-graph, single-query ESG fast path."""
    scores = np.asarray(graph_scores, dtype=np.float64)
    query_nodes = np.asarray(query_nodes, dtype=np.int64)
    if scores.ndim != 2 or len(query_nodes) != scores.shape[0]:
        raise ValueError("Batch ESG inputs have incompatible shapes")
    batch_size, num_nodes = scores.shape
    if num_nodes <= 1:
        return [[int(node)] for node in query_nodes]
    if np.any(query_nodes < 0) or np.any(query_nodes >= num_nodes):
        raise ValueError("Batch ESG query node is outside the score matrix")

    ranked = np.argsort(-scores, axis=1, kind="stable")
    keep = ranked != query_nodes[:, None]
    ranked = ranked[keep].reshape(batch_size, num_nodes - 1)
    ranked_scores = np.take_along_axis(scores, ranked, axis=1)
    prefix_sum = scores[np.arange(batch_size), query_nodes, None] + np.cumsum(
        ranked_scores, axis=1, dtype=np.float64
    )
    sizes = np.arange(2, num_nodes + 1, dtype=np.float64)[None, :]
    total_sum = np.cumsum(scores, axis=1, dtype=np.float64)[:, -1]
    total_mean = total_sum / float(num_nodes)
    esg_after_add = (prefix_sum - total_mean[:, None] * sizes) / np.power(sizes, tau)
    best_after = np.argmax(esg_after_add, axis=1)
    best_after_score = esg_after_add[np.arange(batch_size), best_after]
    query_score = scores[np.arange(batch_size), query_nodes]
    esg_at_query = query_score - total_mean
    use_after = best_after_score > esg_at_query
    results = []
    for row in range(batch_size):
        best_k = int(best_after[row]) + 1 if use_after[row] else 0
        results.append(
            [int(query_nodes[row])] + ranked[row, :best_k].astype(int).tolist()
        )
    return results

def load_candidategraphs(candidategraphfile_path, vec_length, embedding_tensor):
    vectors = []
    # 璇诲€欓€夊瓙鍥炬椂鍔犺繘搴︽潯
    with open(os.path.join(candidategraphfile_path), 'r') as f:
        for line in tqdm(f, desc="Loading candidate graphs"):
            line = line.strip()
            if not line:
                continue  # 绌鸿灏辫烦杩?
            vec = [0] * vec_length
            for idx in line.split():
                vec[int(idx)] = 1
            vectors.append(vec)

    # 璇诲畬鏂囦欢鍚庡啀涓€娆℃€ц浆鎴?tensor锛堥伩鍏嶅湪寰幆閲屽弽澶嶅垱寤猴級
    device = embedding_tensor.device
    candidategraphsvec = torch.tensor(vectors, dtype=torch.float32, device=device)
    candidategraphs_num = torch.sum(candidategraphsvec, dim=1)
    candidategraphs_feature = torch.mm(candidategraphsvec, embedding_tensor)
    candidategraphscenterfeature = torch.div(candidategraphs_feature, candidategraphs_num.view(-1, 1))

    return candidategraphsvec, candidategraphscenterfeature


def default_candidate_tree_paths(args, dataset):
    names = parse_candidate_tree_names(args.candidate_tree_names)
    tree_dir = getattr(args, "candidate_tree_dir", None) or os.path.join("dataset", dataset, "tree")
    return [
        os.path.join(tree_dir, f"{name}tree.pkl")
        for name in names
    ]


def load_kary_tree_safe(file_path):
    if not os.path.exists(file_path):
        print(f"Tree file does not exist: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            tree = pickle.load(f)
        if hasattr(tree, "build_leaf_id_map"):
            tree.build_leaf_id_map()
        return tree
    except Exception as e:
        print(f"Failed to load tree {file_path}: {e}")
        return None


def load_candidate_trees(args, dataset):
    tree_configs = [
        {"name": name, "weight": 1.0}
        for name in parse_candidate_tree_names(getattr(args, "candidate_tree_names", None))
    ]

    loaded_trees = []
    tree_dir = getattr(args, "candidate_tree_dir", None) or os.path.join("dataset", dataset, "tree")
    for tree_cfg in tree_configs:
        tree_name = tree_cfg["name"]
        tree_path = os.path.join(tree_dir, f"{tree_name}tree.pkl")
        tree = load_kary_tree_safe(tree_path)
        if tree is None:
            continue
        new_tree_cfg = tree_cfg.copy()
        new_tree_cfg["path"] = tree_path
        new_tree_cfg["tree_obj"] = tree
        loaded_trees.append(new_tree_cfg)

    if loaded_trees:
        total_weight = sum(t["weight"] for t in loaded_trees)
        for t in loaded_trees:
            t["norm_weight"] = t["weight"] / total_weight if total_weight > 0 else 1.0 / len(loaded_trees)
    return loaded_trees


def get_preloaded_tree(candidate_trees, tree_path):
    target = os.path.normcase(os.path.abspath(tree_path))
    for tree_cfg in candidate_trees:
        loaded_path = os.path.normcase(os.path.abspath(tree_cfg.get("path", "")))
        if loaded_path == target:
            return tree_cfg["tree_obj"]
    return None



def _to_numpy_embedding(embedding_tensor, expected_num_nodes=None):
    if embedding_tensor is None:
        return None
    if isinstance(embedding_tensor, np.ndarray):
        arr = embedding_tensor
    elif torch.is_tensor(embedding_tensor):
        arr = embedding_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(embedding_tensor)
    if arr.ndim != 2:
        raise ValueError(f"embedding must be 2-D, got {arr.shape}")
    # CSQR tree caches require row=node_id. Do not infer orientation from
    # row/column size: small graphs such as texas have num_nodes < emb_dim.
    if expected_num_nodes is not None:
        expected_num_nodes = int(expected_num_nodes)
        if arr.shape[0] == expected_num_nodes:
            pass
        elif arr.shape[1] == expected_num_nodes:
            arr = arr.T
        else:
            raise ValueError(
                f"embedding shape {arr.shape} does not match expected node count {expected_num_nodes}"
            )
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] <= 64:
        arr = arr.T
    return arr.astype(np.float32, copy=False)


def _node_ids(node):
    return [int(v) for v in getattr(node, "node_ids", [])]


def _cosine_np(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _coverage_center(ids, emb_np):
    ids = [v for v in ids if 0 <= int(v) < emb_np.shape[0]]
    if not ids:
        return None
    return emb_np[ids].mean(axis=0).astype(np.float32, copy=False)


def ensure_csqr_tree_cache(tree, emb_np):
    """Populate CSQR cached fields on every tree node if they are missing.

    Cached fields:
      ctx_center: c_u, mean embedding of graph nodes covered by tree node u
      ctx_size_score: s_size(u), soft preference for middle-level tree nodes
      ctx_coh_score: s_coh(u), embedding coherence inside V(u)
    """
    if tree is None or getattr(tree, "root", None) is None or emb_np is None:
        return tree
    if getattr(tree, "cache_version", None) == "csqr-context-cache-v1":
        center = np.asarray(getattr(tree.root, "ctx_center", None), dtype=np.float32).reshape(-1)
        if center.shape[0] != emb_np.shape[1]:
            raise ValueError("Precomputed index cache has the wrong embedding dimension")
        if hasattr(tree, "build_leaf_id_map"):
            tree.build_leaf_id_map()
        return tree
    n = max(1, int(emb_np.shape[0]))
    log_n = float(np.log(n)) if n > 1 else 0.0

    def fill(node):
        for child in getattr(node, "children", []):
            fill(child)
        ids = _node_ids(node)
        center = getattr(node, "ctx_center", None)
        if center is not None:
            center = np.asarray(center, dtype=np.float32).reshape(-1)
            if center.shape[0] != emb_np.shape[1]:
                center = None
                node.ctx_center = None
        if center is None:
            center = getattr(node, "meanembedding", None)
            if center is not None:
                center = np.asarray(center, dtype=np.float32).reshape(-1)
                if center.shape[0] != emb_np.shape[1]:
                    center = None
        if center is None:
            center = _coverage_center(ids, emb_np)
        if center is not None:
            center = np.asarray(center, dtype=np.float32).reshape(-1)
            node.ctx_center = center
            # Keep legacy field synchronized for old code paths.
            if getattr(node, "meanembedding", None) is None:
                node.meanembedding = center
        size = max(1, len(ids))
        if log_n > 0.0:
            rho = float(np.log(size) / log_n)
        else:
            rho = 0.5
        rho = max(0.0, min(1.0, rho))
        node.ctx_size_score = float(4.0 * rho * (1.0 - rho))
        if center is None or not ids:
            node.ctx_coh_score = 0.0
        else:
            valid = [v for v in ids if 0 <= int(v) < emb_np.shape[0]]
            if not valid:
                node.ctx_coh_score = 0.0
            else:
                x = emb_np[valid].astype(np.float32, copy=False)
                c_norm = np.linalg.norm(center) + 1e-12
                x_norm = np.linalg.norm(x, axis=1) + 1e-12
                sims = (x @ center) / (x_norm * c_norm)
                node.ctx_coh_score = float(np.mean(sims)) if sims.size else 0.0
        return node

    fill(tree.root)
    if hasattr(tree, "build_leaf_id_map"):
        tree.build_leaf_id_map()
    return tree


def ensure_csqr_caches(candidate_trees, embedding_tensor, expected_num_nodes=None):
    emb_np = _to_numpy_embedding(embedding_tensor, expected_num_nodes=expected_num_nodes)
    if emb_np is None:
        return candidate_trees
    for tree_cfg in candidate_trees or []:
        ensure_csqr_tree_cache(tree_cfg.get("tree_obj"), emb_np)
    return candidate_trees


def _query_embedding_from_row(query_row, emb_np):
    if query_row is None:
        return None
    if torch.is_tensor(query_row):
        ids = torch.nonzero(query_row).reshape(-1).detach().cpu().tolist()
    else:
        arr = np.asarray(query_row)
        ids = np.nonzero(arr)[0].tolist()
    ids = [int(v) for v in ids if 0 <= int(v) < emb_np.shape[0]]
    if not ids:
        return None
    return emb_np[ids].mean(axis=0).astype(np.float32, copy=False)


def parse_context_score_terms(text):
    allowed = {'size', 'relevance', 'coherence'}
    terms = [term.strip().lower() for term in str(text).split(',') if term.strip()]
    if not terms:
        raise ValueError('--context_score_terms must contain at least one scoring term')
    unknown = sorted(set(terms) - allowed)
    if unknown:
        raise ValueError('Unknown context scoring terms: {}'.format(','.join(unknown)))
    return _unique_preserve_order(terms)


def csqr_context_score(node, query_vec, score_terms):
    center = getattr(node, "ctx_center", None)
    if center is None:
        center = getattr(node, "meanembedding", None)
    size_score = float(getattr(node, "ctx_size_score", 0.0))
    coh_score = float(getattr(node, "ctx_coh_score", 0.0))
    rel_score = _cosine_np(query_vec, center)
    # Map cosine terms to [0, 1] so that all three components share one scale.
    rel_score = (rel_score + 1.0) / 2.0
    coh_score = (coh_score + 1.0) / 2.0
    values = {
        'size': size_score,
        'relevance': rel_score,
        'coherence': coh_score,
    }
    return sum(values[term] for term in score_terms) / float(len(score_terms))


def collect_ancestor_path(leaf_node):
    path = []
    node = leaf_node
    while node is not None and hasattr(node, "node_ids"):
        path.append(node)
        node = getattr(node, "parent", None)
    return path


def select_csqr_context_node(tree, query_node, query_vec, score_terms):
    leaf = tree.find_leaf_by_id(int(query_node)) if hasattr(tree, "find_leaf_by_id") else None
    if leaf is None:
        return None
    path = collect_ancestor_path(leaf)
    if not path:
        return None
    return max(path, key=lambda node: csqr_context_score(node, query_vec, score_terms))


def selected_parent_margin(node, query_vec):
    """Cosine margin between the selected node and its best sibling."""
    parent = getattr(node, "parent", None)
    siblings = list(getattr(parent, "children", [])) if parent is not None else []
    if len(siblings) <= 1:
        return None
    selected_center = getattr(node, "ctx_center", None)
    selected_similarity = _cosine_np(query_vec, selected_center)
    alternatives = [
        _cosine_np(query_vec, getattr(sibling, "ctx_center", None))
        for sibling in siblings
        if sibling is not node
    ]
    return selected_similarity - max(alternatives) if alternatives else None


def mean_pairwise_jaccard(regions):
    regions = [set(region) for region in regions if region]
    if len(regions) <= 1:
        return 1.0 if regions else 0.0
    values = []
    for left_index in range(len(regions)):
        for right_index in range(left_index + 1, len(regions)):
            union = regions[left_index] | regions[right_index]
            values.append(len(regions[left_index] & regions[right_index]) / max(1, len(union)))
    return float(np.mean(values)) if values else 0.0


def adaptive_frequency_merge(per_tree_regions):
    nonempty_regions = [set(region) for region in per_tree_regions if region]
    if not nonempty_regions:
        return set(), {}
    freq = {}
    for region in nonempty_regions:
        for node in region:
            freq[node] = freq.get(node, 0) + 1
    denom = max(1, len(nonempty_regions))
    freq_norm = {node: count / denom for node, count in freq.items()}
    avg_freq = sum(freq_norm.values()) / max(1, len(freq_norm))
    merged = {node for node, score in freq_norm.items() if score >= avg_freq}
    return merged, {node: int(freq[node]) for node in freq}

def candidate_region_from_tree(tree, query_node, query_vec, score_terms):
    node = select_csqr_context_node(tree, query_node, query_vec, score_terms)
    return list(getattr(node, "node_ids", [])) if node is not None else []


def merge_tree_regions(args, per_tree_regions):
    nonempty_regions = [set(region) for region in per_tree_regions if region]
    if not nonempty_regions:
        return set(), {}

    freq = {}
    for region in nonempty_regions:
        for node in region:
            freq[node] = freq.get(node, 0) + 1

    if args.candidate_tree_merge == "intersection":
        return set.intersection(*nonempty_regions), freq

    if args.candidate_tree_merge == "consensus":
        min_count = max(1, int(getattr(args, "candidate_consensus_min", 2)))
        min_count = min(min_count, len(nonempty_regions))
        return {node for node, count in freq.items() if count >= min_count}, freq

    if args.candidate_tree_merge == "ranked_consensus":
        min_count = max(1, int(getattr(args, "candidate_consensus_min", 2)))
        min_count = min(min_count, len(nonempty_regions))
        return {node for node, count in freq.items() if count >= min_count}, freq

    if args.candidate_tree_merge == "adaptive":
        return adaptive_frequency_merge(per_tree_regions)

    return set(freq.keys()), freq


def retrieve_csqr_context(
    args, candidate_trees, query_node_ids, query_vec, embedding=None,
    return_diagnostics=False,
):
    """Return gated, ranked CSQR non-query context nodes and optional diagnostics."""
    if not query_node_ids:
        empty = {
            "mean_parent_margin": 0.0,
            "tree_agreement": 0.0,
            "raw_context_size": 0,
            "selected_context_size": 0,
            "gate_applied": True,
            "gate_reason": "empty_query",
        }
        return ([], empty) if return_diagnostics else []
    primary_query_node = int(query_node_ids[0])
    per_tree_regions = []
    margins = []
    for tree_cfg in candidate_trees:
        tree = tree_cfg["tree_obj"]
        selected_node = select_csqr_context_node(
            tree, primary_query_node, query_vec, args.context_score_terms_parsed
        )
        region = list(getattr(selected_node, "node_ids", [])) if selected_node is not None else []
        if region:
            per_tree_regions.append(set(int(v) for v in region))
            margin = selected_parent_margin(selected_node, query_vec)
            if margin is not None and np.isfinite(margin):
                margins.append(float(margin))
    merged_set, frequencies = merge_tree_regions(args, per_tree_regions)
    query_set = set(int(v) for v in query_node_ids)
    candidates = [int(v) for v in merged_set if int(v) not in query_set]
    mean_margin = float(np.mean(margins)) if margins else 0.0
    agreement = mean_pairwise_jaccard(per_tree_regions)
    raw_size = len(candidates)
    gate_reasons = []
    if mean_margin < float(getattr(args, "context_min_parent_margin", 0.0)):
        gate_reasons.append("low_parent_margin")
    if agreement < float(getattr(args, "context_min_tree_agreement", 0.0)):
        gate_reasons.append("low_tree_agreement")
    reject_above = int(getattr(args, "context_reject_above", 0))
    if reject_above > 0 and raw_size > reject_above:
        gate_reasons.append("oversized_context")

    if gate_reasons:
        selected = []
    elif args.candidate_tree_merge == "ranked_consensus":
        similarity_weight = min(1.0, max(0.0, float(args.candidate_similarity_weight)))
        tree_count = max(1, len(per_tree_regions))
        scored = []
        for node_id in candidates:
            vote_score = frequencies.get(node_id, 0) / tree_count
            node_vector = embedding[node_id] if embedding is not None else None
            similarity_score = (_cosine_np(query_vec, node_vector) + 1.0) / 2.0
            score = (1.0 - similarity_weight) * vote_score + similarity_weight * similarity_score
            scored.append((score, frequencies.get(node_id, 0), similarity_score, node_id))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected = [item[3] for item in scored]
    else:
        selected = sorted(candidates)

    max_nodes = int(getattr(args, "context_max_nodes", 0))
    if max_nodes > 0:
        selected = selected[:max_nodes]
    diagnostics = {
        "mean_parent_margin": mean_margin,
        "tree_agreement": agreement,
        "raw_context_size": raw_size,
        "selected_context_size": len(selected),
        "gate_applied": bool(gate_reasons),
        "gate_reason": ",".join(gate_reasons),
    }
    return (selected, diagnostics) if return_diagnostics else selected


def personalized_pagerank_scores(G, query_node_ids, args):
    sources = [int(v) for v in query_node_ids if int(v) in G]
    if not sources:
        return {}
    mass = 1.0 / len(sources)
    personalization = {v: mass for v in sources}
    return nx.pagerank(
        G,
        alpha=float(args.ppr_alpha),
        personalization=personalization,
        tol=float(args.ppr_tol),
        max_iter=int(args.ppr_max_iter),
    )


def rank_nodes_by_ppr(nodes, ppr_scores):
    return sorted(
        (int(v) for v in nodes),
        key=lambda v: (-float(ppr_scores.get(v, 0.0)), v),
    )


def retrieve_baseline_context(
    args, G, query_node_ids, target_nonquery_size, query_index, num_nodes
):
    """Retrieve at most the CSQR-matched number of non-query context nodes."""
    provider = args.context_provider
    query_set = set(int(v) for v in query_node_ids)
    target = max(0, int(target_nonquery_size))
    if provider == 'none' or target == 0:
        return []

    all_nonquery = [v for v in range(int(num_nodes)) if v not in query_set]
    if provider == 'random':
        # A per-query RNG makes results invariant to loop ordering and reproducible.
        rng = random.Random(int(args.random_seed) + 1000003 * int(args.queryfilenum) + int(query_index))
        sample_size = min(target, len(all_nonquery))
        return rng.sample(all_nonquery, sample_size)

    ppr_scores = personalized_pagerank_scores(G, query_node_ids, args)
    if provider == 'ppr':
        return rank_nodes_by_ppr(all_nonquery, ppr_scores)[:target]

    hops = 1 if provider == '1hop' else 2
    candidates = set()
    frontier = {int(v) for v in query_node_ids if int(v) in G}
    visited = set(frontier)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(int(v) for v in G.neighbors(node))
        next_frontier -= visited
        candidates.update(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return rank_nodes_by_ppr(candidates, ppr_scores)[:target]


def expand_candidate_pool(G, base_nodes, hops):
    pool = set(int(v) for v in base_nodes if int(v) in G)
    frontier = set(pool)
    for _ in range(max(0, hops)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.neighbors(node))
        next_frontier -= pool
        pool.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(pool)


def Closeness_centrality_subgraph(G, C):
        """
        G: 鍘熷浘
        C: 绀惧尯鑺傜偣闆嗗悎
        杩斿洖 cent[v]锛氳嫢 v 涓嶅湪鍥句腑锛屽垯涓績搴?= 0
        """
        # C = list(C)
        cent = {}

        # 鏋勫缓瀛愬浘锛堝彧鍖呭惈 C 涓湪 G 閲岀殑鑺傜偣锛?
        C_in = [v for v in C if v in G]
        G_sub = G.subgraph(C_in).copy()

        for v in C:
            if v not in G_sub:
                # 涓嶅湪鍥句腑 鈫?涓績搴︿负 0
                cent[v] = 0.0
                continue

            dist = nx.single_source_shortest_path_length(G_sub, v)
            s = 0.0
            for u in G_sub.nodes():
                if u == v:
                    continue
                if u in dist:
                    s += dist[u]

            cent[v] = 0.0 if s == 0 else 1.0 / s

        return cent


def Degree_centrality_subgraph(G, C):
    cent = {}

    C_in = [v for v in C if v in G]
    G_sub = G.subgraph(C_in)
    n = len(C_in)

    for v in C:
        if v in G_sub:
            deg = G_sub.degree(v)
            if n > 1:
                cent[v] = float(deg) / (n - 1)  # 瑙勮寖鍖栧害涓績鎬?
            else:
                cent[v] = 0.0
        else:
            cent[v] = 0.0

    return cent

def merged_subgraph_centrality(G, nodes, method='degree'):
    if method == 'closeness':
        return Closeness_centrality_subgraph(G, nodes)
    return Degree_centrality_subgraph(G, nodes)

def query_bias(q, C, cent_dict):
        """
        q: 鏌ヨ鑺傜偣
        C: 绀惧尯鑺傜偣闆嗗悎
        cent_dict: {v: Cent_harm(v)} 瀛楀吀
        """
        max_cent = max(cent_dict[v] for v in C)
        return 1 - cent_dict[q] / max_cent
def load_queryandaddnode(path, dataset, addnode,vec_length):
        """
        璇诲彇 dataset.query锛屾瘡涓€琛屽搴?query 涓€涓鐑悜閲忥紝
        鍐嶆妸 addnode[j] 瀵瑰簲鐨勪綅缃篃缃?1銆?
        绗?j 琛屽搴?addnode[j]銆?
        """
        query = []
        file_query = open(path + dataset + '/' + dataset + ".query", 'r')
        j=0
        for line in file_query:
            vec = [0 for i in range(vec_length)]
            line = line.strip()
            line = line.split(" ")
            vec[int(addnode[j])] = 1
            j=j+1
            for i in line:
                vec[int(i)] = 1
            query.append(vec)
        return torch.Tensor(query)
def load_query_rewrite_gt_nocom(path, dataset, vec_length,queryfilenum, query_file=None):
    # load query and ground truth
    query = []
    gt = []
    query_path = query_file or (
        path + dataset + '/transductive' + '/' + dataset + "-query-{}".format(queryfilenum) + ".txt"
    )
    file_query = open(query_path, 'r')
    file_gt = open(path + dataset + '/' + "true_communities.txt", 'r')
    coms=[]
    for line in file_gt:
        line = line.strip()
        line = line.split(" ")
        coms.append(line)
    for line in file_query:
        vecquery = [0 for i in range(vec_length)]
        vecgt = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        for i in line:
            vecquery[int(i)] = 1
            for com in coms:
                if i in com:
                    for j in com:
                        vecgt[int(j)] = 1
        query.append(vecquery)
        gt.append(vecgt)
    return torch.Tensor(query), torch.Tensor(gt)
def load_kary_tree(pkl_file_path):
    try:
        with open(pkl_file_path, 'rb') as f:
            loaded_k_tree = pickle.load(f)
        return loaded_k_tree
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"鍔犺浇鏂囦欢鏃跺嚭閿? {e}")
        return None
def bias_of_node(q, C_q, cent):
    """
    q:   鑺傜偣锛堝崟涓妭鐐癸紝蹇呴』鍦?C_q 閲岄潰锛?
    C_q: 鑺傜偣闆嗗悎/鍒楄〃锛屼緥濡?[1, 2, 3, 4]
    cent: 鎺ヨ繎涓績搴﹀瓧鍏革紝node -> closeness
    """
    # 鍙?C_q 涓墍鏈夎妭鐐圭殑鏈€澶ф帴杩戜腑蹇冨害
    max_cent = max(cent[v] for v in C_q if v in cent)

    if max_cent == 0:
        # 閬垮厤琚?0 闄わ紝瑙嗘儏鍐佃繑鍥?0 鎴栬€?1锛岀湅浣犺鏂囩殑绾﹀畾
        return 0.0

    return 1.0 - cent[q] / max_cent


def fmt(x: float) -> str:
    return f"{x:.1f}"  # 淇濈暀1浣嶅皬鏁帮細0.3 鑰屼笉鏄?0.30000000000000004

# ===================== 涓荤▼搴?=====================
if __name__ == "__main__":
    args = parse_args()
    args.context_score_terms_parsed = parse_context_score_terms(args.context_score_terms)
    args.topkvalue = max(1, int(args.topkvalue))
    print(args)
    print("Context score terms: {}".format(','.join(args.context_score_terms_parsed)))
    print(f"structure_rewrite_topk: {args.topkvalue}")

    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    embed_path = os.path.join(args.EmbeddingPath, args.embedding_tensor_name + '.npy')
    if not os.path.isfile(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")

    embedding_tensor = torch.from_numpy(np.load(embed_path)).float().cuda()
    query, labels = load_query_rewrite_gt_nocom(
        "./dataset/", args.dataset, embedding_tensor.shape[0], args.queryfilenum,
        query_file=args.query_file,
    )
    query = query.cuda()
    labels = labels.cuda()
    G = nx.read_edgelist(
        "dataset/{}/{}.edges".format(args.dataset, args.dataset),
        nodetype=int,
        create_using=nx.Graph()
    )
    expected_num_nodes = int(embedding_tensor.shape[0])
    G.add_nodes_from(range(expected_num_nodes))

    candidate_trees = load_candidate_trees(args, args.dataset)
    ensure_csqr_caches(candidate_trees, embedding_tensor, expected_num_nodes=expected_num_nodes)
    if candidate_trees:
        print("Loaded CSQR context-index trees:")
        for tree_cfg in candidate_trees:
            print(f"  - {tree_cfg['name']} ({tree_cfg['path']}), weight={tree_cfg['weight']}")

    base_start = time.time()
    query_feature = torch.mm(query, embedding_tensor)
    query_num = torch.sum(query, dim=1).clamp_min(1).view(-1, 1)
    query_feature = F.normalize(torch.div(query_feature, query_num), dim=1)

    E = embedding_tensor
    if E.dim() == 2 and E.shape[0] != query_feature.shape[1]:
        E_norm = F.normalize(E, dim=1)
        score_q = torch.mm(query_feature, E_norm.t())
    else:
        E_norm = F.normalize(E, dim=0)
        score_q = torch.mm(query_feature, E_norm)

    base_time = time.time() - base_start
    context_centers = []
    representative_centers = []
    emb_np = _to_numpy_embedding(embedding_tensor, expected_num_nodes=expected_num_nodes)

    # CSQR determines the per-query context budget for every provider. This
    # reference pass is experimental-control setup and is excluded from the
    # online time of non-CSQR providers.
    csqr_reference_start = time.time()
    csqr_nonquery_contexts = []
    csqr_context_diagnostics = []
    query_node_id_rows = []
    query_vec_rows = []
    for i in range(query.shape[0]):
        query_node_ids = torch.nonzero(query[i]).reshape(-1).detach().cpu().tolist()
        query_node_ids = [int(v) for v in query_node_ids]
        query_vec = _query_embedding_from_row(query[i], emb_np)
        query_node_id_rows.append(query_node_ids)
        query_vec_rows.append(query_vec)
        context_nodes, context_diagnostics = retrieve_csqr_context(
            args, candidate_trees, query_node_ids, query_vec,
            embedding=emb_np, return_diagnostics=True,
        )
        csqr_nonquery_contexts.append(context_nodes)
        csqr_context_diagnostics.append(context_diagnostics)
    csqr_reference_time = time.time() - csqr_reference_start

    provider_start = time.time()
    if args.context_provider == 'csqr':
        provider_nonquery_contexts = csqr_nonquery_contexts
    else:
        provider_nonquery_contexts = [
            retrieve_baseline_context(
                args,
                G,
                query_node_id_rows[i],
                len(csqr_nonquery_contexts[i]),
                i,
                embedding_tensor.shape[0],
            )
            for i in range(query.shape[0])
        ]
    provider_time = (
        csqr_reference_time
        if args.context_provider == 'csqr'
        else time.time() - provider_start
    )

    downstream_start = time.time()

    for i in tqdm(range(query.shape[0]), desc="Context Calibration", total=query.shape[0]):
        query_node_ids = query_node_id_rows[i]
        if len(query_node_ids) == 0:
            context_centers.append(query_feature[i].clone())
            representative_centers.append(query_feature[i].clone())
            continue

        context_ids = _unique_preserve_order(provider_nonquery_contexts[i] + query_node_ids)
        valid_context_ids = [v for v in context_ids if 0 <= v < embedding_tensor.shape[0]]
        if valid_context_ids:
            center = F.normalize(embedding_tensor[valid_context_ids].mean(dim=0), dim=0)
        else:
            center = query_feature[i].clone()
        context_centers.append(center)

        representative_ids = [v for v in valid_context_ids if v not in set(query_node_ids)] or valid_context_ids
        if representative_ids:
            cent_dict = merged_subgraph_centrality(G, representative_ids, args.merged_centrality)
            query_vec_t = query_feature[i].unsqueeze(0)
            node_embeddings = embedding_tensor[representative_ids]
            cos_sim_vec = F.cosine_similarity(
                node_embeddings,
                query_vec_t.expand_as(node_embeddings),
                dim=1
            )
            cent_values = torch.tensor(
                [cent_dict.get(v, 0.0) for v in representative_ids],
                dtype=torch.float32,
                device=embedding_tensor.device
            )
            max_cent = torch.max(cent_values).item() if cent_values.numel() > 0 else 0.0
            cent_score = cent_values / max_cent if max_cent > 0 else torch.zeros_like(cent_values)
            rep_score = (args.representative_relevance_weight * cos_sim_vec
                         + (1.0 - args.representative_relevance_weight) * cent_score)
            topk = min(args.topkvalue, len(representative_ids))
            topk_indices = torch.topk(rep_score, k=topk, largest=True, sorted=True).indices
            topk_node_ids = [representative_ids[int(idx.item())] for idx in topk_indices]
            representative_vectors = F.normalize(embedding_tensor[topk_node_ids], dim=1)
            if args.representative_center == 'median' and len(topk_node_ids) > 1:
                representative_center = F.normalize(
                    torch.median(representative_vectors, dim=0).values,
                    dim=0,
                )
            else:
                representative_center = F.normalize(representative_vectors.mean(dim=0), dim=0)
        else:
            representative_center = center
        representative_centers.append(representative_center)

    context_centers = torch.stack(context_centers, dim=0)
    representative_centers = torch.stack(representative_centers, dim=0)

    if E.dim() == 2 and E.shape[0] != query_feature.shape[1]:
        score_context = torch.mm(context_centers, E_norm.t())
        score_struct = torch.mm(representative_centers, E_norm.t())
    else:
        score_context = torch.mm(context_centers, E_norm)
        score_struct = torch.mm(representative_centers, E_norm)

    if args.fusion_mode == 'normalized':
        raw_weights = np.asarray([
            float(args.query_weight),
            float(args.context_weight) if args.semrewrite == 1 else 0.0,
            float(args.representative_weight) if args.structrewrite == 1 else 0.0,
        ], dtype=np.float64)
        if np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
            raise ValueError('Normalized fusion weights must be non-negative with positive sum')
        normalized_weights = raw_weights / raw_weights.sum()
        query_score = (
            float(normalized_weights[0]) * score_q
            + float(normalized_weights[1]) * score_context
            + float(normalized_weights[2]) * score_struct
        )
    else:
        query_score = score_q
        if args.semrewrite == 1:
            query_score = (1 - args.alpha) * query_score + args.alpha * score_context
        if args.structrewrite == 1:
            query_score = (1 - args.belta) * query_score + args.belta * score_struct

    y_pred = torch.zeros_like(query_score)
    print("ESG prefix search starts on the full graph; no candidate pruning is used.")
    can_batch_esg = (
        float(args.uncertainty_gamma) == 0.0
        and all(len(values) == 1 for values in query_node_id_rows)
    )
    if can_batch_esg:
        selected_rows = ESGSearch_FullGraphSingleQueryBatch(
            [values[0] for values in query_node_id_rows],
            query_score.detach().cpu().numpy(),
            tau=args.tau,
        )
        for i, selected_candidates in enumerate(selected_rows):
            y_pred[i, selected_candidates] = 1
    else:
        for i in tqdm(range(query_score.shape[0]), desc="ESG Prefix Search", total=query_score.shape[0]):
            query_index = torch.nonzero(query[i]).reshape(-1).detach().cpu().tolist()
            query_index = [int(v) for v in query_index]
            if not query_index:
                continue
            selected_candidates, best_k, best_threshold = ESGSearch_CandidatePruned(
                query_index=query_index,
                graph_score=query_score[i].detach().cpu().numpy(),
                G=G,
                candidate_nodes=None,
                uncertainty=None,
                gamma=args.uncertainty_gamma,
                tau=args.tau,
            )
            for nid in selected_candidates:
                y_pred[i][nid] = 1

    downstream_time = time.time() - downstream_start
    total_time = base_time + provider_time + downstream_time
    context_sizes = [
        len(set(provider_nonquery_contexts[i]).union(query_node_id_rows[i]))
        for i in range(query.shape[0])
    ]
    avg_context_size = float(np.mean(context_sizes)) if context_sizes else 0.0
    print("Context provider: {}".format(args.context_provider))
    print("Average context size |R_q|: {:.4f}".format(avg_context_size))
    print("Context retrieval time (one query): {:.6f}".format(
        provider_time / max(1, query_feature.shape[0])
    ))
    print("The context rewrite and ESG search using time: {:.4f}".format(total_time))
    print("The context rewrite and ESG search using time (one query): {:.4f}".format(
        total_time / max(1, query_feature.shape[0])
    ))

    f1_score = f1_score_calculation(y_pred.int(), labels.int())
    print("F1 score by {}: {:.4f}".format(args.context_provider, f1_score))
    y_pred_cpu = y_pred.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    nmi, ari, jac = evaluation(y_pred_cpu, labels_cpu)
    print("NMI score by {}: {:.4f}".format(args.context_provider, nmi))
    print("ARI score by {}: {:.4f}".format(args.context_provider, ari))
    print("JAC score by {}: {:.4f}".format(args.context_provider, jac))
    if args.query_artifact_output:
        artifact_path = os.path.abspath(args.query_artifact_output)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        query_context_cosine = F.cosine_similarity(query_feature, context_centers, dim=1)
        query_representative_cosine = F.cosine_similarity(
            query_feature, representative_centers, dim=1
        )
        context_representative_cosine = F.cosine_similarity(
            context_centers, representative_centers, dim=1
        )
        query_sizes = torch.sum(query, dim=1).detach().cpu().numpy().astype(np.int32)
        mean_query_degrees = np.asarray([
            np.mean([G.degree(v) for v in row]) if row else 0.0
            for row in query_node_id_rows
        ], dtype=np.float32)
        np.savez_compressed(
            artifact_path,
            y_pred=y_pred_cpu.astype(np.uint8),
            labels=labels_cpu.astype(np.uint8),
            query_sizes=query_sizes,
            mean_query_degrees=mean_query_degrees,
            context_sizes=np.asarray(context_sizes, dtype=np.int32),
            query_context_cosine=query_context_cosine.detach().cpu().numpy().astype(np.float32),
            query_representative_cosine=query_representative_cosine.detach().cpu().numpy().astype(np.float32),
            context_representative_cosine=context_representative_cosine.detach().cpu().numpy().astype(np.float32),
            context_mean_parent_margin=np.asarray([
                item["mean_parent_margin"] for item in csqr_context_diagnostics
            ], dtype=np.float32),
            context_tree_agreement=np.asarray([
                item["tree_agreement"] for item in csqr_context_diagnostics
            ], dtype=np.float32),
            context_raw_sizes=np.asarray([
                item["raw_context_size"] for item in csqr_context_diagnostics
            ], dtype=np.int32),
            context_gate_applied=np.asarray([
                item["gate_applied"] for item in csqr_context_diagnostics
            ], dtype=np.bool_),
            context_gate_reasons=np.asarray([
                item["gate_reason"] for item in csqr_context_diagnostics
            ]),
            queryfilenum=np.asarray([args.queryfilenum], dtype=np.int32),
            context_provider=np.asarray([args.context_provider]),
        )
        print("Saved per-query artifact: {}".format(artifact_path))
