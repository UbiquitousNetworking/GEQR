# GEQR

This package contains the GEQR query-time implementation, the five graph inputs,
shared-vector indexes, frozen selected query splits.

Due to GitHub's file size limitations, the complete code and data package is
available on Baidu Netdisk:

- File: `GEQR-code.zip`
- Download: [Baidu Netdisk](https://pan.baidu.com/s/1D8cpSeqWbzJNYTVap6Zpaw?pwd=yrqn)
- Extraction code: `yrqn`
- 
## Requirements

- Python 3.8 (the exact paper runner uses the Python 3.8 AST interface).
- NVIDIA CUDA GPU; the published inference source calls `.cuda()`.
- Install PyTorch 2.3.0 with a CUDA build compatible with your driver, then
  `python -m pip install -r requirements.txt`.
- About 1 GB free disk space after extraction and additional GPU memory for DBLP.

## Reproduce

From this package directory:

```
python run.py check
python run.py evaluate --datasets texas
python run.py evaluate --datasets texas cornell wisconsin citeseer dblp
```

`check` verifies hashes without a GPU. `evaluate` writes `results/<dataset>/`
with per-query F1/Jaccard, five split means, predictions, and a summary. 
It loads the frozen trees and runs the same query-time implementation and parameters
as the paper; `exact_reference_predictions` should be `true`. For a quick GPU smoke test, use Texas.

## Rebuild indexes

`build_index_trees.py` and `index_tree_schema.py` are included as the index
construction source. The shipped indexes are the exact frozen trees used in the
paper. Rebuilding depends on partition-library versions, seeds, and bank-specific
construction settings; a rebuild is a new experiment until structure hashes are
verified against `configs/<dataset>.json`.
