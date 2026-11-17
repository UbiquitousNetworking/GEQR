"""Build validated, reproducible and minimal CSQR partition-tree indexes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from pathlib import Path
import pickle
import platform
import sys
import time

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from index_tree_schema import CACHE_VERSION, SCHEMA_VERSION, KaryTree, TreeNode, validate_tree
from shared_leaf_vectors import STORAGE_VERSION, create_shared_vectors, share_leaf_vectors


BUILDER_VERSION = "2.1.0"
SUPPORTED_ALGORITHMS = ("louvain", "leiden", "embedding_cluster")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dblp")
    parser.add_argument("--split", default="transductive")
    parser.add_argument("--algorithms", default="louvain,leiden,embedding_cluster")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=3, help="Recorded for compatibility; k-clique is rejected for partition trees")
    parser.add_argument("--min-community-size", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-shrink-ratio", type=float, default=0.01)
    parser.add_argument("--min-quality-gain", type=float, default=1.0e-6)
    parser.add_argument("--embedding-branching-factor", type=int, default=8)
    parser.add_argument("--min-cluster-gain", type=float, default=0.01)
    parser.add_argument("--edges-file")
    parser.add_argument("--nodes-file")
    parser.add_argument("--embedding-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--shared-vectors-dir", help="Content-addressed leaf vectors shared across tree banks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_json(path: Path, payload) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, encoded)


def load_node_ids(nodes_file: Path | None, embedding_rows: int, split: str):
    if nodes_file is None:
        if split != "transductive":
            raise ValueError("Non-transductive indexes require an explicit --nodes-file")
        return list(range(int(embedding_rows))), "embedding_row_equals_node_id"
    node_ids = []
    for line in nodes_file.read_text(encoding="utf-8").splitlines():
        node_ids.extend(int(value) for value in line.split())
    if not node_ids:
        raise ValueError("--nodes-file contains no node IDs")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("--nodes-file contains duplicate node IDs")
    return sorted(node_ids), "explicit_nodes_file_embedding_row_equals_node_id"


def load_inputs(edges_path: Path, embedding_path: Path, nodes_path: Path | None, split: str):
    embedding = np.load(embedding_path)
    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be 2-D, got {embedding.shape}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding contains NaN or Inf")
    embedding = np.asarray(embedding, dtype=np.float32)
    node_ids, mapping_policy = load_node_ids(nodes_path, embedding.shape[0], split)
    if min(node_ids) < 0 or max(node_ids) >= embedding.shape[0]:
        raise ValueError("Node IDs must index embedding rows; provide a remapped embedding/nodes file")
    graph = nx.read_edgelist(edges_path, nodetype=int, create_using=nx.Graph())
    unknown = set(graph.nodes) - set(node_ids)
    if unknown:
        raise ValueError(f"Edges contain nodes outside the declared split: {sorted(unknown)[:10]}")
    graph.add_nodes_from(node_ids)
    return graph, embedding, node_ids, mapping_policy


def derive_seed(base_seed: int, depth: int, node_ids) -> int:
    digest = hashlib.sha256()
    digest.update(f"{base_seed}:{depth}:".encode("ascii"))
    digest.update(np.asarray(sorted(node_ids), dtype=np.int64).tobytes())
    return int.from_bytes(digest.digest()[:4], "little") % (2**31 - 1)


def louvain_partition(graph: nx.Graph, resolution: float, seed: int):
    return [set(values) for values in nx.community.louvain_communities(
        graph, resolution=resolution, seed=seed
    )]


def leiden_partition(graph: nx.Graph, resolution: float, seed: int):
    nodes = sorted(int(v) for v in graph.nodes)
    position = {node_id: index for index, node_id in enumerate(nodes)}
    edges = [(position[int(u)], position[int(v)]) for u, v in graph.edges]
    i_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    partition = leidenalg.find_partition(
        i_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution),
        n_iterations=-1,
        seed=int(seed),
    )
    return [set(nodes[index] for index in community) for community in partition]


def partition_graph(algorithm: str, graph: nx.Graph, resolution: float, seed: int):
    if algorithm == "louvain":
        return louvain_partition(graph, resolution, seed)
    if algorithm == "leiden":
        return leiden_partition(graph, resolution, seed)
    raise ValueError(
        "k-clique produces overlapping and incomplete covers, so it is not supported by "
        "the single-path partition-tree schema. Use a future multi-path/DAG index instead."
    )


def embedding_partition(node_ids, normalized_embedding, branching_factor: int, seed: int):
    """Cluster unit-normalized embeddings; Euclidean distance then tracks cosine similarity."""
    ids = np.asarray(sorted(int(v) for v in node_ids), dtype=np.int64)
    cluster_count = min(int(branching_factor), len(ids))
    if cluster_count <= 1:
        return [set(ids.tolist())], 0.0
    values = normalized_embedding[ids]
    parent_center = values.mean(axis=0, keepdims=True)
    parent_sse = float(np.sum((values - parent_center) ** 2))
    if parent_sse <= 1.0e-12:
        return [set(ids.tolist())], 0.0
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        n_init=3,
        max_iter=100,
        batch_size=min(2048, max(64, len(ids))),
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(values)
    communities = [
        set(ids[labels == label].tolist()) for label in sorted(set(labels.tolist()))
    ]
    gain = max(0.0, min(1.0, (parent_sse - float(model.inertia_)) / parent_sse))
    return communities, gain


def valid_partition(parent_ids, communities):
    parent = set(int(v) for v in parent_ids)
    communities = [set(int(v) for v in values) for values in communities if values]
    union = set().union(*communities) if communities else set()
    total = sum(len(values) for values in communities)
    return communities, union == parent and total == len(union)


def build_tree(
    graph, algorithm, seed, resolution, min_size, max_depth, min_shrink, min_gain,
    embedding=None, embedding_branching_factor=8, min_cluster_gain=0.01,
):
    root_ids = tuple(sorted(int(v) for v in graph.nodes))
    tree = KaryTree(TreeNode(root_ids))
    stop_counts = {}
    normalized_embedding = None
    if algorithm == "embedding_cluster":
        if embedding is None:
            raise ValueError("embedding_cluster requires the embedding matrix")
        normalized_embedding = embedding / np.maximum(
            np.linalg.norm(embedding, axis=1, keepdims=True), 1.0e-12
        )

    def stop(node, reason):
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
        for node_id in node.node_ids:
            node.add_child(TreeNode((node_id,)))

    def recurse(node, depth):
        size = len(node.node_ids)
        if size == 1:
            return
        if size <= min_size:
            stop(node, "min_community_size")
            return
        if depth >= max_depth:
            stop(node, "max_depth")
            return
        local_seed = derive_seed(seed, depth, node.node_ids)
        if algorithm == "embedding_cluster":
            raw_communities, cluster_gain = embedding_partition(
                node.node_ids, normalized_embedding, embedding_branching_factor, local_seed
            )
            if cluster_gain < min_cluster_gain:
                stop(node, "insufficient_cluster_gain")
                return
        else:
            subgraph = graph.subgraph(node.node_ids)
            if subgraph.number_of_edges() == 0:
                stop(node, "no_edges")
                return
            raw_communities = partition_graph(algorithm, subgraph, resolution, local_seed)
        communities, complete_partition = valid_partition(node.node_ids, raw_communities)
        if not complete_partition:
            raise ValueError(f"{algorithm} returned a non-partition at depth {depth}")
        if len(communities) <= 1:
            stop(node, "single_community")
            return
        largest = max(len(values) for values in communities)
        if (size - largest) / float(size) < min_shrink:
            stop(node, "insufficient_shrink")
            return
        if algorithm != "embedding_cluster":
            quality = nx.community.modularity(subgraph, communities, resolution=resolution)
            if not np.isfinite(quality) or quality < min_gain:
                stop(node, "insufficient_quality_gain")
                return
        communities.sort(key=lambda values: (min(values), len(values), tuple(sorted(values))))
        for values in communities:
            child = TreeNode(tuple(sorted(values)))
            node.add_child(child)
            recurse(child, depth + 1)

    recurse(tree.root, 0)
    return tree, stop_counts


def compute_minimal_cache(tree: KaryTree, embedding: np.ndarray):
    total_nodes = len(tree.root.node_ids)
    log_total = float(np.log(max(2, total_nodes)))
    normalized = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1.0e-12)

    def fill(node):
        if not node.children:
            node_id = int(node.node_ids[0])
            raw_sum = embedding[node_id].astype(np.float32, copy=True)
            unit_sum = normalized[node_id].astype(np.float32, copy=True)
            count = 1
        else:
            raw_sum = np.zeros(embedding.shape[1], dtype=np.float32)
            unit_sum = np.zeros(embedding.shape[1], dtype=np.float32)
            count = 0
            for child in node.children:
                child_raw, child_unit, child_count = fill(child)
                raw_sum += child_raw
                unit_sum += child_unit
                count += child_count
        center = raw_sum / float(count)
        node.ctx_center = np.asarray(center, dtype=np.float32)
        rho = float(np.log(max(1, count)) / log_total)
        rho = min(1.0, max(0.0, rho))
        node.ctx_size_score = float(4.0 * rho * (1.0 - rho))
        norm = float(np.linalg.norm(center))
        node.ctx_coh_score = float(np.dot(unit_sum / float(count), center / norm)) if norm > 1.0e-12 else 0.0
        return raw_sum, unit_sum, count

    fill(tree.root)
    tree.cache_version = CACHE_VERSION


def structural_hash(tree: KaryTree) -> str:
    digest = hashlib.sha256()
    stack = [(tree.root, 0)]
    while stack:
        node, depth = stack.pop()
        digest.update(np.asarray([depth, len(node.node_ids), len(node.children)], dtype=np.int64).tobytes())
        digest.update(np.asarray(node.node_ids, dtype=np.int64).tobytes())
        stack.extend((child, depth + 1) for child in reversed(node.children))
    return digest.hexdigest()


def dependency_versions():
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "igraph": ig.__version__,
        "leidenalg": getattr(leidenalg, "__version__", "unknown"),
        "sklearn": sklearn.__version__,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    algorithms = [value.strip().lower() for value in args.algorithms.split(",") if value.strip()]
    if any(value not in (*SUPPORTED_ALGORITHMS, "kclique") for value in algorithms):
        raise ValueError(f"Unknown algorithm in {algorithms}")
    if "kclique" in algorithms:
        raise ValueError(
            "k-clique was removed from the ordinary tree builder: its overlapping, incomplete "
            "communities require a multi-path/DAG index and query interface."
        )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integer seeds")
    if not (0.0 <= args.min_shrink_ratio < 1.0):
        raise ValueError("--min-shrink-ratio must be in [0,1)")
    if args.max_depth < 1 or args.min_community_size < 1:
        raise ValueError("Stopping parameters must be positive")
    if args.embedding_branching_factor < 2 or not (0.0 <= args.min_cluster_gain < 1.0):
        raise ValueError("Embedding clustering parameters are invalid")

    edges_path = Path(args.edges_file) if args.edges_file else root / "dataset" / args.dataset / f"{args.dataset}.edges"
    embedding_path = Path(args.embedding_path) if args.embedding_path else root / "pretrain_result" / f"{args.dataset}.npy"
    nodes_path = Path(args.nodes_file) if args.nodes_file else None
    output_dir = Path(args.output_dir) if args.output_dir else root / "dataset" / args.dataset / "tree_v2" / args.split
    edges_path = edges_path.resolve()
    embedding_path = embedding_path.resolve()
    nodes_path = nodes_path.resolve() if nodes_path else None
    output_dir = output_dir.resolve()
    for path in (edges_path, embedding_path, nodes_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Manifest already exists: {manifest_path}; use a new directory or --overwrite")

    graph, embedding, node_ids, mapping_policy = load_inputs(
        edges_path, embedding_path, nodes_path, args.split
    )
    input_files = {
        "edges": {"path": str(edges_path), "bytes": edges_path.stat().st_size, "sha256": sha256_file(edges_path)},
        "embedding": {"path": str(embedding_path), "bytes": embedding_path.stat().st_size, "sha256": sha256_file(embedding_path)},
    }
    if nodes_path:
        input_files["nodes"] = {"path": str(nodes_path), "bytes": nodes_path.stat().st_size, "sha256": sha256_file(nodes_path)}
    vector_dir = Path(args.shared_vectors_dir) if args.shared_vectors_dir else root / 'dataset' / args.dataset / 'shared_vectors'
    leaf_vectors = create_shared_vectors(embedding, vector_dir)
    fresh_manifest = {
        "storage_version": STORAGE_VERSION,
        "shared_leaf_vectors": leaf_vectors.manifest(),
        "schema_version": SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "builder_version": BUILDER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "mapping_policy": mapping_policy,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "embedding_shape": list(embedding.shape),
        "algorithms": algorithms,
        "seeds": seeds,
        "parameters": {
            "resolution": args.resolution,
            "k": args.k,
            "min_community_size": args.min_community_size,
            "max_depth": args.max_depth,
            "min_shrink_ratio": args.min_shrink_ratio,
            "min_quality_gain": args.min_quality_gain,
            "embedding_branching_factor": args.embedding_branching_factor,
            "min_cluster_gain": args.min_cluster_gain,
        },
        "dependencies": dependency_versions(),
        "inputs": input_files,
        "indexes": [],
        "rejected_duplicates": [],
    }
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "storage_version", "shared_leaf_vectors", "dataset", "split", "parameters", "inputs"):
            if manifest.get(key) != fresh_manifest.get(key):
                raise ValueError(f"Cannot resume: manifest field {key!r} does not match")
        manifest["algorithms"] = sorted(set(manifest.get("algorithms", []) + algorithms))
        manifest["seeds"] = sorted(set(manifest.get("seeds", []) + seeds))
    else:
        manifest = fresh_manifest
    known_structures = {
        record["structural_sha256"]: record["name"] for record in manifest["indexes"]
    }
    attempted_names = {
        record["name"] for record in manifest["indexes"] + manifest["rejected_duplicates"]
    }
    for algorithm in algorithms:
        for seed in seeds:
            name = f"{algorithm}_seed{seed}"
            if name in attempted_names:
                print(f"Skipping completed attempt: {name}")
                continue
            started = time.perf_counter()
            tree, stop_counts = build_tree(
                graph, algorithm, seed, args.resolution, args.min_community_size,
                args.max_depth, args.min_shrink_ratio, args.min_quality_gain,
                embedding=embedding,
                embedding_branching_factor=args.embedding_branching_factor,
                min_cluster_gain=args.min_cluster_gain,
            )
            compute_minimal_cache(tree, embedding)
            validation = validate_tree(tree, node_ids, embedding.shape[1], require_partition=True)
            structure_digest = structural_hash(tree)
            if structure_digest in known_structures:
                manifest["rejected_duplicates"].append({
                    "name": name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "duplicate_of": known_structures[structure_digest],
                    "structural_sha256": structure_digest,
                })
                print(f"Rejected duplicate {name}; same structure as {known_structures[structure_digest]}")
                continue
            known_structures[structure_digest] = name
            # leaf_id_map is rebuilt after loading; do not serialize the redundant dictionary.
            tree.leaf_id_map = {}
            share_leaf_vectors(tree, leaf_vectors)
            payload = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
            output_path = output_dir / f"{name}tree.pkl"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(output_path)
            atomic_write_bytes(output_path, payload)
            content_digest = hashlib.sha256(payload).hexdigest()
            elapsed = time.perf_counter() - started
            record = {
                "name": name,
                "algorithm": algorithm,
                "seed": seed,
                "path": str(output_path),
                "bytes": len(payload),
                "content_sha256": content_digest,
                "structural_sha256": structure_digest,
                "build_time_seconds": elapsed,
                "stop_counts": stop_counts,
                "validation": validation,
            }
            manifest["indexes"].append(record)
            atomic_write_json(manifest_path, manifest)
            print(
                f"Built {name}: F={len(payload) / 1024**2:.2f} MiB, "
                f"nodes={validation['tree_nodes']}, depth={validation['max_depth']}, "
                f"time={elapsed:.2f}s"
            )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
