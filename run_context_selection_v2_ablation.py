# -*- coding: utf-8 -*-
"""Tune and evaluate gated context selection on every DBLP graph node."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import run_index_source_ablation as adapter


ROOT = Path(__file__).resolve().parent
TREE_NAMES = [f"embedding_cluster_seed{seed}" for seed in range(10)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/dblp/tree_v2/transductive_v2/manifest.json")
    parser.add_argument("--output-dir", default="resultsrewrite/dblp/context_selection_v2")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_query_f1(prediction, label):
    prediction = np.asarray(prediction, dtype=bool)
    label = np.asarray(label, dtype=bool)
    intersection = np.logical_and(prediction, label).sum(axis=1).astype(np.float64)
    denominator = prediction.sum(axis=1) + label.sum(axis=1)
    return np.divide(2.0 * intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)


def stratified_split(node_count, community_by_node, fraction, seed):
    rng = np.random.default_rng(seed)
    validation = []
    for community in sorted(set(community_by_node.tolist())):
        nodes = np.flatnonzero(community_by_node == community)
        rng.shuffle(nodes)
        validation.extend(nodes[: int(round(len(nodes) * fraction))].tolist())
    validation = np.asarray(sorted(validation), dtype=np.int64)
    mask = np.ones(node_count, dtype=bool)
    mask[validation] = False
    return validation, np.flatnonzero(mask)


def make_chunks(base: Path, name: str, nodes, chunk_size):
    directory = base / "query_chunks" / name
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk_id, start in enumerate(range(0, len(nodes), chunk_size)):
        chunk_nodes = np.asarray(nodes[start:start + chunk_size], dtype=np.int64)
        path = directory / f"chunk_{chunk_id:04d}.txt"
        expected = "".join(f"{int(node)}\n" for node in chunk_nodes)
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            path.write_text(expected, encoding="utf-8")
        chunks.append({"id": chunk_id, "path": path, "nodes": chunk_nodes})
    return chunks


def command_for(manifest, tree_dir, query_file, query_number, artifact, config):
    tree_names = config.get("tree_names", TREE_NAMES)
    values = [
        sys.executable, "rewriteandsearch_csqr_internal_ablation.py",
        "--dataset", manifest["dataset"],
        "--queryfilenum", str(query_number),
        "--query_file", str(query_file),
        "--context_provider", "csqr",
        "--candidate_tree_names", ",".join(tree_names),
        "--candidate_tree_dir", str(tree_dir),
        "--candidate_tree_merge", config["merge"],
        "--candidate_consensus_min", str(config["votes"]),
        "--context_max_nodes", str(config["cap"]),
        "--context_reject_above", str(config.get("context_reject_above", 0)),
        "--context_min_parent_margin", str(config.get("context_min_parent_margin", -1000000)),
        "--context_min_tree_agreement", str(config.get("context_min_tree_agreement", -1)),
        "--candidate_similarity_weight", "0.25",
        "--fusion_mode", config["fusion"],
        "--query_weight", str(config["query_weight"]),
        "--context_weight", str(config["context_weight"]),
        "--representative_weight", str(config["representative_weight"]),
        "--alpha", "0.8", "--belta", "0.8",
        "--topkvalue", str(config["topk"]),
        "--representative_center", config["representative_center"],
        "--context_score_terms", "size,relevance,coherence",
        "--tau", "0.1",
        "--merged_centrality", "degree",
        "--query_artifact_output", str(artifact),
    ]
    return values


def cleanup_env(env, protected_ids):
    for key, value in list(env.items()):
        if id(value) in protected_ids:
            continue
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            env[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_config(env, main_code, protected_ids, manifest, tree_dir, output_dir, split_name, chunks, config, resume):
    artifact_dir = output_dir / "artifacts" / config["name"] / split_name
    log_dir = output_dir / "logs" / config["name"] / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        artifact = artifact_dir / f"chunk_{chunk['id']:04d}.npz"
        if resume and artifact.exists():
            continue
        sys.argv = command_for(
            manifest, tree_dir, chunk["path"],
            20000 + chunk["id"], artifact, config,
        )[1:]
        log_path = log_dir / f"chunk_{chunk['id']:04d}.log"
        with log_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            exec(main_code, env)
        if not artifact.exists():
            raise RuntimeError(f"Missing artifact {artifact}")
        cleanup_env(env, protected_ids)


def load_config_rows(output_dir, split_name, chunks, config, baseline_by_node):
    rows = []
    artifact_dir = output_dir / "artifacts" / config["name"] / split_name
    for chunk in chunks:
        path = artifact_dir / f"chunk_{chunk['id']:04d}.npz"
        with np.load(path) as artifact:
            context_f1 = per_query_f1(artifact["y_pred"], artifact["labels"])
            for index, node in enumerate(chunk["nodes"]):
                rows.append({
                    "node_id": int(node),
                    "context_f1": float(context_f1[index]),
                    "baseline_f1": float(baseline_by_node[int(node)]),
                    "margin": float(artifact["context_mean_parent_margin"][index]),
                    "agreement": float(artifact["context_tree_agreement"][index]),
                    "raw_size": int(artifact["context_raw_sizes"][index]),
                    "selected_size": int(artifact["context_sizes"][index]) - 1,
                })
    return rows


def tune_gate(rows):
    context = np.asarray([row["context_f1"] for row in rows])
    baseline = np.asarray([row["baseline_f1"] for row in rows])
    margin = np.asarray([row["margin"] for row in rows])
    agreement = np.asarray([row["agreement"] for row in rows])
    raw_size = np.asarray([row["raw_size"] for row in rows])
    best = {
        "min_margin": 1.0e9,
        "min_agreement": -1.0,
        "reject_above": 0,
        "mean_f1": float(baseline.mean()),
        "mean_delta_f1": 0.0,
        "accept_rate": 0.0,
        "positive_rate": 0.0,
        "helpful_rate": 0.0,
    }
    for min_margin in (-1.0, 0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30):
        for min_agreement in (-1.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            for reject_above in (0, 32, 64, 96, 128, 192, 256):
                accept = (margin >= min_margin) & (agreement >= min_agreement)
                if reject_above > 0:
                    accept &= raw_size <= reject_above
                hybrid = np.where(accept, context, baseline)
                candidate = {
                    "min_margin": min_margin,
                    "min_agreement": min_agreement,
                    "reject_above": reject_above,
                    "mean_f1": float(hybrid.mean()),
                    "mean_delta_f1": float((hybrid - baseline).mean()),
                    "accept_rate": float(accept.mean()),
                    "positive_rate": float(np.mean(hybrid - baseline > 0)),
                    "helpful_rate": float(np.mean(hybrid - baseline >= 0.02)),
                }
                key = (candidate["mean_f1"], candidate["accept_rate"])
                if best is None or key > (best["mean_f1"], best["accept_rate"]):
                    best = candidate
    return best


def summarize_config(rows, gate):
    context = np.asarray([row["context_f1"] for row in rows])
    baseline = np.asarray([row["baseline_f1"] for row in rows])
    margin = np.asarray([row["margin"] for row in rows])
    agreement = np.asarray([row["agreement"] for row in rows])
    raw_size = np.asarray([row["raw_size"] for row in rows])
    accept = (margin >= gate["min_margin"]) & (agreement >= gate["min_agreement"])
    if gate["reject_above"] > 0:
        accept &= raw_size <= gate["reject_above"]
    prediction_f1 = np.where(accept, context, baseline)
    delta = prediction_f1 - baseline
    return {
        "queries": len(rows),
        "mean_f1": float(prediction_f1.mean()),
        "baseline_mean_f1": float(baseline.mean()),
        "mean_delta_f1": float(delta.mean()),
        "positive_rate": float(np.mean(delta > 0)),
        "helpful_rate": float(np.mean(delta >= 0.02)),
        "accept_rate": float(accept.mean()),
        "mean_context_size_when_accepted": float(np.mean([
            rows[index]["selected_size"] + 1 for index in np.flatnonzero(accept)
        ])) if np.any(accept) else 1.0,
    }


def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (ROOT / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    node_count = int(manifest["node_count"])

    community_by_node = np.full(node_count, -1, dtype=np.int32)
    community_path = ROOT / "dataset" / manifest["dataset"] / "true_communities.txt"
    for community, line in enumerate(community_path.read_text(encoding="utf-8").splitlines()):
        for value in line.split():
            community_by_node[int(value)] = community
    if np.any(community_by_node < 0):
        raise ValueError("Ground-truth communities do not cover every node")
    validation_nodes, test_nodes = stratified_split(
        node_count, community_by_node, args.validation_fraction, args.seed
    )
    split_manifest = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "validation_nodes": len(validation_nodes),
        "test_nodes": len(test_nodes),
        "validation_community_sizes": np.bincount(community_by_node[validation_nodes]).tolist(),
        "test_community_sizes": np.bincount(community_by_node[test_nodes]).tolist(),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_chunks = make_chunks(output_dir, "validation", validation_nodes, args.chunk_size)
    test_chunks = make_chunks(output_dir, "test", test_nodes, args.chunk_size)
    all_chunks = make_chunks(output_dir, "all", np.arange(node_count), args.chunk_size)

    baseline_table = list(csv.DictReader(
        (ROOT / "resultsrewrite/dblp/all_node_query_validation/per_node.csv").open(encoding="utf-8-sig")
    ))
    baseline_by_node = {int(row["node_id"]): float(row["no_context_f1"]) for row in baseline_table}

    env, main_code = adapter.inference_namespace()
    loader = env["load_candidate_trees"]
    loader_args = argparse.Namespace(
        candidate_tree_names=",".join(TREE_NAMES),
        candidate_tree_dir=str(manifest_path.parent),
    )
    trees = loader(loader_args, manifest["dataset"])
    if len(trees) != len(TREE_NAMES):
        raise RuntimeError("Incomplete embedding-cluster tree loading")
    env["load_candidate_trees"] = lambda *unused: trees
    protected_ids = {id(trees), *(id(tree) for tree in trees)}

    selection_configs = []
    for cap in (32, 48, 64, 96):
        for votes in (2, 3):
            selection_configs.append({
                "name": f"selection_cap{cap}_votes{votes}",
                "merge": "ranked_consensus", "cap": cap, "votes": votes,
                "fusion": "normalized", "query_weight": 0.6,
                "context_weight": 0.2, "representative_weight": 0.2,
                "topk": 5, "representative_center": "median",
            })

    validation_summaries = []
    started = time.time()
    for config in selection_configs:
        print(f"Validation selection: {config['name']}", flush=True)
        run_config(env, main_code, protected_ids, manifest, manifest_path.parent, output_dir,
                   "validation", validation_chunks, config, args.resume)
        rows = load_config_rows(output_dir, "validation", validation_chunks, config, baseline_by_node)
        gate = tune_gate(rows)
        validation_summaries.append({**config, **gate})
    best_selection = max(validation_summaries, key=lambda row: (row["mean_f1"], row["accept_rate"]))

    fusion_configs = []
    for query_weight in (0.5, 0.6, 0.7):
        side_weight = (1.0 - query_weight) / 2.0
        for topk in (3, 5):
            fusion_configs.append({
                "name": f"fusion_q{int(query_weight*10)}_top{topk}",
                "merge": "ranked_consensus", "cap": best_selection["cap"],
                "votes": best_selection["votes"], "fusion": "normalized",
                "query_weight": query_weight, "context_weight": side_weight,
                "representative_weight": side_weight, "topk": topk,
                "representative_center": "median",
            })
    for config in fusion_configs:
        print(f"Validation fusion: {config['name']}", flush=True)
        run_config(env, main_code, protected_ids, manifest, manifest_path.parent, output_dir,
                   "validation", validation_chunks, config, args.resume)
        rows = load_config_rows(output_dir, "validation", validation_chunks, config, baseline_by_node)
        gate = tune_gate(rows)
        validation_summaries.append({**config, **gate})
    write_csv(output_dir / "validation_grid.csv", validation_summaries)
    best = max(validation_summaries, key=lambda row: (row["mean_f1"], row["accept_rate"]))
    best_config = {key: best[key] for key in (
        "name", "merge", "cap", "votes", "fusion", "query_weight",
        "context_weight", "representative_weight", "topk", "representative_center",
    )}
    best_gate = {key: best[key] for key in ("min_margin", "min_agreement", "reject_above")}

    print(f"Best validation config: {best_config['name']} gate={best_gate}", flush=True)
    run_config(env, main_code, protected_ids, manifest, manifest_path.parent, output_dir,
               "test", test_chunks, best_config, args.resume)
    test_rows = load_config_rows(output_dir, "test", test_chunks, best_config, baseline_by_node)
    test_summary = summarize_config(test_rows, best_gate)

    cumulative_configs = [
        {
            "name": "01_gate_diagnostics_legacy_selection", "merge": "adaptive",
            "cap": 0, "votes": 1, "fusion": "sequential", "query_weight": 0.6,
            "context_weight": 0.2, "representative_weight": 0.2,
            "topk": 1, "representative_center": "mean",
        },
        {
            "name": "02_bounded_ranked_consensus", "merge": "ranked_consensus",
            "cap": best_config["cap"], "votes": best_config["votes"],
            "fusion": "sequential", "query_weight": 0.6, "context_weight": 0.2,
            "representative_weight": 0.2, "topk": 1, "representative_center": "mean",
        },
        {
            "name": "03_normalized_fusion", "merge": "ranked_consensus",
            "cap": best_config["cap"], "votes": best_config["votes"],
            "fusion": "normalized", "query_weight": best_config["query_weight"],
            "context_weight": best_config["context_weight"],
            "representative_weight": best_config["representative_weight"],
            "topk": 1, "representative_center": "mean",
        },
        {**best_config, "name": "04_robust_multi_representative"},
    ]
    all_node_rows = []
    no_gate = {"min_margin": -1.0, "min_agreement": -1.0, "reject_above": 0}
    for config in cumulative_configs:
        print(f"All-node cumulative ablation: {config['name']}", flush=True)
        run_config(env, main_code, protected_ids, manifest, manifest_path.parent, output_dir,
                   "all", all_chunks, config, args.resume)
        rows = load_config_rows(output_dir, "all", all_chunks, config, baseline_by_node)
        ungated_summary = summarize_config(rows, no_gate)
        all_node_rows.append({"setting": config["name"], "gate": "off", **ungated_summary})
        if config["name"] == "04_robust_multi_representative":
            gated_summary = summarize_config(rows, best_gate)
            all_node_rows.append({"setting": "05_validation_selected_gate", "gate": "on", **gated_summary})
    write_csv(output_dir / "all_node_cumulative_ablation.csv", all_node_rows)

    final = {
        "elapsed_seconds": time.time() - started,
        "best_config": best_config,
        "best_gate": best_gate,
        "validation": {key: best[key] for key in (
            "mean_f1", "mean_delta_f1", "accept_rate", "positive_rate", "helpful_rate"
        )},
        "held_out_test": test_summary,
        "all_node_final": all_node_rows[-1],
        "all_node_ablation": all_node_rows,
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
