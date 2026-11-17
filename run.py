# -*- coding: utf-8 -*-
"""Portable reproduction of the five frozen GEQR main-experiment rows."""
import argparse,contextlib,csv,hashlib,json,os,pickle,sys
from pathlib import Path
import numpy as np
import torch
import run_index_source_ablation as adapter
from run_context_selection_v2_ablation import command_for,cleanup_env

ROOT=Path(__file__).resolve().parent
DATASETS=('texas','cornell','wisconsin','citeseer','dblp')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def config(dataset):return json.loads((ROOT/'configs'/f'{dataset}.json').read_text(encoding='utf-8'))
def paths(dataset,cfg):
    tree_dir=ROOT/'indexes'/dataset
    for r in cfg['trees']:yield tree_dir/r['filename'],r['sha256']
    yield ROOT/'pretrain_result'/f'{dataset}.npy',cfg['embedding_sha256']
    yield ROOT/'dataset'/dataset/f'{dataset}.edges',cfg['edges_sha256']
    yield ROOT/'dataset'/dataset/'true_communities.txt',cfg['communities_sha256']
    for split in cfg['splits']:yield ROOT/split['file'],split['sha256']

def check(dataset):
    cfg=config(dataset)
    for p,expected in paths(dataset,cfg):
        if digest(p)!=expected:raise RuntimeError(f'Changed or missing input: {p}')
    print(f'{dataset}: {len(cfg["trees"])} trees; {sum(s["rows"] for s in cfg["splits"])} query rows; input hashes OK',flush=True)

def evaluate(dataset):
    check(dataset)
    if not torch.cuda.is_available():raise RuntimeError('Main-experiment reproduction needs CUDA; data check works on CPU.')
    cfg=config(dataset);base=cfg['frozen_parameters']
    splits=[np.array([int(x) for x in (ROOT/s['file']).read_text().split()],dtype=np.int64) for s in cfg['splits']]
    nodes=np.unique(np.concatenate(splits));pos=[np.searchsorted(nodes,s) for s in splits]
    assert all(np.array_equal(nodes[p],s) for p,s in zip(pos,splits))
    out=ROOT/'results'/dataset;out.mkdir(parents=True,exist_ok=True)
    query=out/'unique_queries.txt';query.write_text('\n'.join(map(str,nodes))+'\n',encoding='utf-8')
    env,code=adapter.inference_namespace()
    trees=[]
    for record in cfg['trees']:
        p=ROOT/'indexes'/dataset/record['filename']
        with p.open('rb') as f:tree=pickle.load(f)
        tree.build_leaf_id_map()
        trees.append(dict(name=record['name'],tree_obj=tree,path=str(p),weight=1.))
    env['load_candidate_trees']=lambda *unused:trees
    env['ensure_csqr_caches']=lambda selected,*unused,**kw:selected
    manifest={'dataset':dataset}
    cmd=command_for(manifest,ROOT/'indexes'/dataset,query,1,'',base)
    for key,val in [('--tau',base['tau']),('--candidate_similarity_weight',base['similarity_weight']),('--context_score_terms',base['score_terms'])]:
        cmd[cmd.index(key)+1]=str(val)
    cmd+=['--lambda_rep',str(base['representative_relevance_weight'])]
    argv=sys.argv;sys.argv=cmd[1:]
    try:
        with (out/'inference.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):exec(code,env)
    finally:sys.argv=argv
    pred=env['y_pred'].detach().cpu().numpy().astype(bool)
    labels=env['labels'].detach().cpu().numpy().astype(bool)
    tp=(pred&labels).sum(axis=1);f1=2*tp/(pred.sum(axis=1)+labels.sum(axis=1));jac=tp/(pred|labels).sum(axis=1)
    reference=np.load(ROOT/'reference'/dataset/'full.npz')
    exact=np.array_equal(nodes,reference['nodes']) and np.array_equal(np.packbits(pred,axis=1),reference['prediction_packed'])
    rows=[];split_rows=[]
    for sid,p in enumerate(pos,1):
        split_rows.append(dict(dataset=dataset,split=sid,queries=len(p),F1=float(f1[p].mean()),JAC=float(jac[p].mean())))
        for row,i in enumerate(p,1):rows.append(dict(dataset=dataset,split=sid,query_row=row,query=int(nodes[i]),F1=float(f1[i]),JAC=float(jac[i])))
    write_csv(out/'per_query.csv',rows);write_csv(out/'split_results.csv',split_rows)
    mean_f1=float(np.mean([r['F1'] for r in split_rows]));mean_jac=float(np.mean([r['JAC'] for r in split_rows]))
    summary=dict(dataset=dataset,query_rows=len(rows),F1=mean_f1,JAC=mean_jac,split_F1_std=float(np.std([r['F1'] for r in split_rows],ddof=1)),exact_reference_predictions=exact)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(out/'predictions.npz',nodes=nodes,prediction_packed=np.packbits(pred,axis=1),graph_nodes=pred.shape[1])
    cleanup_env(env,{id(trees),*(id(x) for x in trees)})
    print(summary,flush=True)
    if not exact:raise AssertionError('Predictions differ from the published frozen result; see results and runtime versions.')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','evaluate'])
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    args=p.parse_args();os.chdir(ROOT)
    for d in args.datasets:
        if args.command=='check':check(d)
        else:evaluate(d)
if __name__=='__main__':main()
