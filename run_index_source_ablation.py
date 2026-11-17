"""Controlled source ablation using the existing inference code and metric bodies.

Only inference classes/functions are imported via AST: cdlib and DGL are training/
construction dependencies and are not needed to evaluate already built indexes.
Tree objects are reused across query files; loading is outside reported query time.
"""
import argparse
import ast
import contextlib
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import run_csqr_internal_ablation as base


ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inference_namespace():
    import numpy as np
    import networkx as nx
    import torch
    import pickle
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
    env = dict(np=np, nx=nx, torch=torch, pickle=pickle, os=os,
               normalized_mutual_info_score=normalized_mutual_info_score,
               adjusted_rand_score=adjusted_rand_score, jaccard_score=jaccard_score,
               __name__='index_ablation_inference')
    for filename, names in [('graphtree_dblp_meanemb.py', {'TreeNode', 'KaryTree'}),
                            ('utils.py', {'f1_score_calculation', 'evaluation'})]:
        tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
        selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), env)
    # Old pickle files may refer to either the original module or __main__.
    module = types.ModuleType('graphtree_dblp_meanemb')
    module.__dict__.update(env)
    sys.modules['graphtree_dblp_meanemb'] = module
    sys.modules['index_ablation_inference'] = module
    globals().update({n: env[n] for n in ('TreeNode', 'KaryTree')})
    path = ROOT / 'rewriteandsearch_csqr_internal_ablation.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    main = tree.body[-1]
    assert isinstance(main, ast.If) and isinstance(main.test, ast.Compare)
    assert isinstance(main.test.left, ast.Name) and main.test.left.id == '__name__'
    definitions = [n for n in tree.body[:-1] if not (
        isinstance(n, ast.ImportFrom) and n.module in ('utils', 'graphtree_dblp_meanemb'))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), env)
    main_code = compile(ast.Module(body=main.body, type_ignores=[]), str(path), 'exec')
    return env, main_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='dblp')
    p.add_argument('--queryfilenums', default='1,2,3,4,5')
    p.add_argument('--experiments', default='B')
    p.add_argument('--settings', default='')
    p.add_argument('--output-dir', default='resultsrewrite/dblp/index_source_ablation_20260921')
    p.add_argument('--alpha', type=float, default=.8)
    p.add_argument('--belta', type=float, default=.8)
    p.add_argument('--tau', type=float, default=.1)
    p.add_argument('--topkvalue', type=int, default=1)
    p.add_argument('--merged-centrality', default='degree')
    p.add_argument('--context-provider', default='csqr', choices=('csqr','none'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--deduplicate', action='store_true', help='Keep only distinct file hashes within each source configuration')
    args = p.parse_args()
    os.chdir(ROOT)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'resultsrewrite' / args.dataset / 'csqr_internal_ablation/subset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    configs = base.build_configs(manifest, args.experiments.split(','), [1,3,10,20,30])
    if args.settings:
        configs = [c for c in configs if c['setting'] in args.settings.split(',')]
    if args.deduplicate:
        hashes = json.loads((ROOT/'resultsrewrite/dblp/index_source_ablation_20260921/index_files.json').read_text())
        by_name = {Path(x['path']).stem[:-4]: x['sha256'] for x in hashes}
        for c in configs:
            seen = set()
            c['trees'] = [t for t in c['trees'] if by_name[t] not in seen and not seen.add(by_name[t])]
            c['setting'] += '-unique'
    if args.context_provider != 'csqr':
        for c in configs:
            c['setting'] += '-' + args.context_provider
    assert configs, 'No configurations selected'
    splits = [int(q) for q in args.queryfilenums.split(',')]
    env, main_code = inference_namespace()
    # Retain every field read by the original inference code. These legacy
    # centrality/alternative-embedding fields are unused in this evaluator.
    original_tree_loader = env['load_kary_tree_safe']
    def compact_tree_loader(path):
        tree = original_tree_loader(path)
        assert tree is not None, path
        stack = [tree.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            for field in ('medoidembedding', 'centroidembedding', 'cent_dict', 'cent_max'):
                node.__dict__.pop(field, None)
        return tree
    env['load_kary_tree_safe'] = compact_tree_loader
    import torch
    assert torch.cuda.is_available(), 'Existing evaluator requires CUDA'
    # Record content hashes for code and evaluation inputs; tree hashes follow loading.
    inputs = [ROOT / name for name in ('run_index_source_ablation.py', 'run_csqr_internal_ablation.py',
        'rewriteandsearch_csqr_internal_ablation.py', 'graphtree_dblp_meanemb.py', 'utils.py')]
    inputs += [ROOT / 'pretrain_result' / (args.dataset + '.npy'), ROOT / 'dataset' / args.dataset / (args.dataset + '.edges'), manifest_path]
    inputs += sorted((ROOT / 'dataset' / args.dataset / 'transductive').glob(args.dataset + '-*'))
    provenance = dict(parameters=vars(args), python=sys.executable, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), started=time.strftime('%Y-%m-%d %H:%M:%S'),
        files={str(f): dict(bytes=f.stat().st_size, sha256=digest(f)) for f in inputs if f.is_file()},
        configs=configs, metric='F1/JAC/NMI/ARI on flattened query-by-node arrays within each split')
    if args.resume and (out/'provenance.json').exists():
        provenance['previous_run'] = json.loads((out/'provenance.json').read_text())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    loader = env['load_candidate_trees']
    rows = []
    if args.resume and (out/'raw.csv').exists():
        with (out/'raw.csv').open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for key in ('repeat','queryfilenum','num_trees','consensus_min','index_size_bytes'):
                r[key] = int(r[key])
            for key in (*base.METRICS,'index_size_MiB','avg_context_size','retrieval_time_one_query','time_one_query'):
                r[key] = float(r[key])
    cache_builder = env['ensure_csqr_caches']
    for config in configs:
        pending = [q for q in splits if not any(r['setting']==config['setting'] and r['queryfilenum']==q for r in rows)]
        if not pending:
            continue
        print('Loading', config['setting'], flush=True)
        ns = argparse.Namespace(candidate_tree_names=','.join(config['trees']))
        trees = loader(ns, args.dataset)
        assert len(trees) == len(config['trees']), 'Incomplete index loading; aborting comparison'
        env['load_candidate_trees'] = lambda *unused: trees
        prepared = [False]
        def prepare_once(*a, **kw):
            if not prepared[0]:
                cache_builder(*a, **kw)
                prepared[0] = True
            return trees
        env['ensure_csqr_caches'] = prepare_once
        index_bytes = base.tree_index_bytes(ROOT, args.dataset, config['trees'])
        for q in pending:
            command = base.build_command(argparse.Namespace(**vars(args), python=sys.executable), config, q)
            provider_pos = command.index('--context_provider') + 1
            command[provider_pos] = args.context_provider
            sys.argv = command[1:]
            log = out / ('%s_%s_q%d.log' % (config['experiment'], config['setting'], q))
            print('Evaluating', config['setting'], 'query file', q, flush=True)
            torch.cuda.synchronize()
            with log.open('w', encoding='utf-8') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                exec(main_code, env)
                torch.cuda.synchronize()
            text = log.read_text(encoding='utf-8')
            metrics = base.parse_log(text)
            assert all(k in metrics for k in base.METRICS), log
            row = dict(dataset=args.dataset, experiment=config['experiment'], setting=config['setting'],
                repeat=config['repeat'], queryfilenum=q, num_trees=len(trees),
                tree_names=','.join(config['trees']), score_terms=config['score_terms'], merge=config['merge'],
                consensus_min=config['consensus_min'], index_size_bytes=index_bytes,
                index_size_MiB=index_bytes / 1024**2, **metrics, log_file=str(log))
            rows.append(row)
            base.write_csv(out / 'raw.csv', rows, list(rows[0]))
            summaries = base.summarize(rows)
            base.write_csv(out / 'summary.csv', summaries, list(summaries[0]))
            print('F1 =', metrics['F1'], 'JAC =', metrics['JAC'], flush=True)
        # Drop all loop/main references to large tree objects before the next source.
        for key in ('candidate_trees', 'tree_cfg'):
            env.pop(key, None)
        trees.clear()
        gc.collect()
    provenance['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
