import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


PARTITIONERS = ("louvain", "leiden", "kclique")
FULL_SCORE_TERMS = "size,relevance,coherence"
METRICS = ("F1", "NMI", "ARI", "JAC")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DBLP CSQR internal ablations A/B/C/D.")
    parser.add_argument("--dataset", default="dblp")
    parser.add_argument("--experiments", default="A,B,C,D")
    parser.add_argument("--queryfilenums", default="1,2,3,4,5")
    parser.add_argument("--tree-sizes", default="1,3,10,20,30")
    parser.add_argument("--subset-repeats", type=int, default=5)
    parser.add_argument("--subset-seed", type=int, default=20260813)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--belta", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--topkvalue", type=int, default=1)
    parser.add_argument("--candidate-consensus-min", type=int, default=1)
    parser.add_argument("--merged-centrality", default="degree", choices=("degree", "closeness"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def csv_values(text, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def numbered_trees(partitioner):
    return [f"{partitioner}{index}" for index in range(1, 11)]


def select_balanced(rng, counts):
    chosen = []
    for partitioner, count in zip(PARTITIONERS, counts):
        chosen.extend(rng.sample(numbered_trees(partitioner), count))
    rng.shuffle(chosen)
    return chosen


def unique_subsets(rng, count, factory):
    subsets = []
    seen = set()
    attempts = 0
    while len(subsets) < count:
        subset = factory()
        key = tuple(sorted(subset))
        attempts += 1
        if key not in seen:
            seen.add(key)
            subsets.append(subset)
        if attempts > 10000:
            raise RuntimeError("Could not generate enough unique tree subsets")
    return subsets


def generate_manifest(args):
    rng = random.Random(args.subset_seed)
    full = [tree for p in PARTITIONERS for tree in numbered_trees(p)]
    a_counts = {
        1: None,
        3: (1, 1, 1),
        10: (3, 3, 4),
        20: (7, 7, 6),
        30: (10, 10, 10),
    }
    experiment_a = {}
    # Always materialize every supported size so the same fixed manifest can
    # later be reused for a larger or smaller --tree-sizes selection.
    for size in (1, 3, 10, 20, 30):
        if size not in a_counts:
            raise ValueError("Experiment A supports tree sizes 1,3,10,20,30")
        if size == 30:
            subsets = [full]
        elif size == 1:
            subsets = unique_subsets(rng, args.subset_repeats, lambda: [rng.choice(full)])
        else:
            counts = a_counts[size]
            subsets = unique_subsets(
                rng,
                args.subset_repeats,
                lambda counts=counts: select_balanced(rng, counts),
            )
        experiment_a[str(size)] = subsets

    # These fixed, balanced sets are determined solely by the seed/manifest,
    # before any metric is read. Mixed-10 uses the requested 3/3/4 allocation.
    experiment_b = {
        "Louvain-10": [numbered_trees("louvain")],
        "Leiden-10": [numbered_trees("leiden")],
        "k-clique-10": [numbered_trees("kclique")],
        "Mixed-10": [select_balanced(rng, (3, 3, 4))],
        "Mixed-30": [full],
    }
    return {
        "dataset": args.dataset,
        "seed": args.subset_seed,
        "numbering_policy": "Only numbered variants 1..10; unnumbered base trees are excluded.",
        "experiment_A": experiment_a,
        "experiment_B": experiment_b,
        "experiment_C": {
            "Full": FULL_SCORE_TERMS,
            "w-o-size": "relevance,coherence",
            "w-o-relevance": "size,coherence",
            "w-o-coherence": "size,relevance",
        },
        "experiment_D": {
            "Adaptive-frequency": {"merge": "adaptive", "consensus_min": 1},
            "Union-no-cross-view-filtering": {"merge": "union", "consensus_min": 1},
            "Consensus-2": {"merge": "consensus", "consensus_min": 2},
            "Consensus-3": {"merge": "consensus", "consensus_min": 3},
        },
    }


def load_or_create_manifest(path, args):
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != args.dataset or manifest.get("seed") != args.subset_seed:
            raise ValueError(
                f"Existing manifest uses dataset={manifest.get('dataset')} seed={manifest.get('seed')}; "
                "use the same settings or a different output directory"
            )
        return manifest
    manifest = generate_manifest(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def tree_index_bytes(root, dataset, tree_names):
    total = 0
    missing = []
    for name in tree_names:
        path = root / "dataset" / dataset / "tree" / f"{name}tree.pkl"
        if path.exists():
            total += path.stat().st_size
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing tree files:\n" + "\n".join(missing))
    return total


def build_configs(manifest, experiments, tree_sizes):
    full = manifest["experiment_A"].get("30")
    if not full:
        full = [[tree for p in PARTITIONERS for tree in numbered_trees(p)]]
    full_trees = full[0]
    configs = []
    if "A" in experiments:
        for size in tree_sizes:
            for subset_index, trees in enumerate(manifest["experiment_A"][str(size)], start=1):
                configs.append({
                    "experiment": "A", "setting": f"{size}-trees", "repeat": subset_index,
                    "trees": trees, "score_terms": FULL_SCORE_TERMS,
                    "merge": "adaptive", "consensus_min": 1,
                })
    if "B" in experiments:
        for setting, subsets in manifest["experiment_B"].items():
            configs.append({
                "experiment": "B", "setting": setting, "repeat": 1,
                "trees": subsets[0], "score_terms": FULL_SCORE_TERMS,
                "merge": "adaptive", "consensus_min": 1,
            })
    if "C" in experiments:
        for setting, terms in manifest["experiment_C"].items():
            configs.append({
                "experiment": "C", "setting": setting, "repeat": 1,
                "trees": full_trees, "score_terms": terms,
                "merge": "adaptive", "consensus_min": 1,
            })
    if "D" in experiments:
        for setting, merge_config in manifest["experiment_D"].items():
            configs.append({
                "experiment": "D", "setting": setting, "repeat": 1,
                "trees": full_trees, "score_terms": FULL_SCORE_TERMS,
                "merge": merge_config["merge"],
                "consensus_min": merge_config["consensus_min"],
            })
    return configs


def build_command(args, config, queryfilenum):
    return [
        args.python, "rewriteandsearch_csqr_internal_ablation.py",
        "--dataset", args.dataset,
        "--queryfilenum", str(queryfilenum),
        "--context_provider", "csqr",
        "--candidate_tree_names", ",".join(config["trees"]),
        "--candidate_tree_merge", config["merge"],
        "--candidate_consensus_min", str(config["consensus_min"]),
        "--context_score_terms", config["score_terms"],
        "--alpha", str(args.alpha),
        "--belta", str(args.belta),
        "--tau", str(args.tau),
        "--topkvalue", str(args.topkvalue),
        "--merged_centrality", args.merged_centrality,
    ]


def config_signature(args, config, queryfilenum):
    payload = {
        "dataset": args.dataset,
        "queryfilenum": queryfilenum,
        "trees": config["trees"],
        "score_terms": config["score_terms"],
        "merge": config["merge"],
        "consensus_min": config["consensus_min"],
        "alpha": args.alpha,
        "belta": args.belta,
        "tau": args.tau,
        "topkvalue": args.topkvalue,
        "merged_centrality": args.merged_centrality,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def parse_log(text):
    patterns = {
        "avg_context_size": r"Average context size \|R_q\|:\s*([-+]?\d*\.?\d+)",
        "retrieval_time_one_query": r"Context retrieval time \(one query\):\s*([-+]?\d*\.?\d+)",
        "time_one_query": r"context rewrite and ESG search using time \(one query\):\s*([-+]?\d*\.?\d+)",
    }
    row = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            row[key] = float(match.group(1))
    for metric in METRICS:
        match = re.search(rf"{metric} score by [^:]+:\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
        if match:
            row[metric] = float(match.group(1))
    return row


def run_one(command, cwd, log_path, skip_existing):
    if skip_existing and log_path.exists():
        old = log_path.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_log(old)
        if "Traceback" not in old and all(metric in parsed for metric in ("F1", "JAC")):
            return old
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    output = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(
            f"Failed ({proc.returncode}): {' '.join(command)}\nLog: {log_path}\n" +
            "\n".join(output.splitlines()[-80:])
        )
    return output


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(path, "w", newline="", encoding="utf-8-sig")
    except PermissionError:
        path = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        handle = open(path, "w", newline="", encoding="utf-8-sig")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def avg(values):
    return statistics.mean(values) if values else None


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def reference_setting(experiment):
    return {"A": "30-trees", "B": "Mixed-10", "C": "Full", "D": "Adaptive-frequency"}[experiment]


def summarize(raw_rows):
    grouped = {}
    for row in raw_rows:
        grouped.setdefault((row["experiment"], row["setting"]), []).append(row)
    summaries = []
    for (experiment, setting), rows in grouped.items():
        f1 = [row["F1"] for row in rows]
        jac = [row["JAC"] for row in rows]
        ref_name = reference_setting(experiment)
        refs = grouped.get((experiment, ref_name), [])
        refs_by_split = {}
        for ref in refs:
            refs_by_split.setdefault(ref["queryfilenum"], []).append(ref["F1"])
        deltas = [
            row["F1"] - avg(refs_by_split[row["queryfilenum"]])
            for row in rows if row["queryfilenum"] in refs_by_split
        ]
        summaries.append({
            "experiment": experiment,
            "setting": setting,
            "trees": rows[0]["num_trees"],
            "num_observations": len(rows),
            "num_subsets": len(set(row["repeat"] for row in rows)),
            "F1_mean": avg(f1), "F1_std": std(f1),
            "Jaccard_mean": avg(jac), "Jaccard_std": std(jac),
            "delta_vs_reference": avg(deltas) if deltas else None,
            "avg_context_size": avg([row["avg_context_size"] for row in rows]),
            "time_per_query": avg([row["time_one_query"] for row in rows]),
            "retrieval_time_per_query": avg([row["retrieval_time_one_query"] for row in rows]),
            "index_size_bytes_mean": avg([row["index_size_bytes"] for row in rows]),
            "index_size_MiB_mean": avg([row["index_size_MiB"] for row in rows]),
        })
    return sorted(summaries, key=lambda row: (row["experiment"], row["setting"]))


def main():
    args = parse_args()
    experiments = [item.upper() for item in csv_values(args.experiments)]
    unknown = sorted(set(experiments) - {"A", "B", "C", "D"})
    if unknown:
        raise ValueError("Unknown experiments: " + ",".join(unknown))
    queryfilenums = csv_values(args.queryfilenums, int)
    tree_sizes = csv_values(args.tree_sizes, int)
    root = Path(__file__).resolve().parent
    out_dir = root / "resultsrewrite" / args.dataset / "csqr_internal_ablation"
    manifest_path = out_dir / "subset_manifest.json"
    manifest = load_or_create_manifest(manifest_path, args)
    configs = build_configs(manifest, experiments, tree_sizes)
    print(f"Fixed pre-evaluation tree manifest: {manifest_path}")
    if args.manifest_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    raw_rows = []
    execution_cache = {}
    for config in configs:
        index_bytes = tree_index_bytes(root, args.dataset, config["trees"])
        for queryfilenum in queryfilenums:
            signature = config_signature(args, config, queryfilenum)
            log_path = out_dir / "logs" / "shared" / f"{signature}.txt"
            print(
                f"Running {config['experiment']} {config['setting']} subset={config['repeat']} "
                f"query={queryfilenum} trees={len(config['trees'])}"
            )
            if signature in execution_cache:
                output = execution_cache[signature]
            else:
                output = run_one(
                    build_command(args, config, queryfilenum), root, log_path, args.skip_existing
                )
                execution_cache[signature] = output
            metrics = parse_log(output)
            raw_rows.append({
                "dataset": args.dataset,
                "experiment": config["experiment"],
                "setting": config["setting"],
                "repeat": config["repeat"],
                "queryfilenum": queryfilenum,
                "num_trees": len(config["trees"]),
                "tree_names": ",".join(config["trees"]),
                "score_terms": config["score_terms"],
                "merge": config["merge"],
                "consensus_min": config["consensus_min"],
                "index_size_bytes": index_bytes,
                "index_size_MiB": index_bytes / (1024.0 * 1024.0),
                **metrics,
                "log_file": str(log_path),
            })

    raw_fields = [
        "dataset", "experiment", "setting", "repeat", "queryfilenum", "num_trees",
        "tree_names", "score_terms", "merge", "consensus_min", "index_size_bytes",
        "index_size_MiB", *METRICS, "avg_context_size", "retrieval_time_one_query",
        "time_one_query", "log_file",
    ]
    raw_path = write_csv(out_dir / "internal_ablation_raw.csv", raw_rows, raw_fields)
    summaries = summarize(raw_rows)
    summary_fields = [
        "experiment", "setting", "trees", "num_observations", "num_subsets",
        "F1_mean", "F1_std", "Jaccard_mean", "Jaccard_std", "delta_vs_reference",
        "avg_context_size", "time_per_query", "retrieval_time_per_query",
        "index_size_bytes_mean", "index_size_MiB_mean",
    ]
    summary_path = write_csv(out_dir / "internal_ablation_summary.csv", summaries, summary_fields)
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
