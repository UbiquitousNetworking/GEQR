"""Content-addressed, process-shared read-only leaf vectors.

Pickles contain a store descriptor, never its array. Keep the referenced .npy
alongside deployed indexes. Absolute paths follow the existing index manifests.
"""
from pathlib import Path
import hashlib
import os
import tempfile
import weakref

import numpy as np

STORAGE_VERSION = "csqr-shared-leaves-v1"
_ARRAYS = weakref.WeakValueDictionary()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SharedLeafVectors:
    def __init__(self, path, sha256, shape):
        self.path = str(Path(path).resolve())
        self.sha256 = sha256
        self.shape = tuple(shape)
        self._array = None

    def __getstate__(self):
        return dict(path=self.path, sha256=self.sha256, shape=self.shape)

    def __setstate__(self, state):
        self.__init__(**state)
        from pathlib import PureWindowsPath
        old_path = PureWindowsPath(state['path'])
        dataset = old_path.parent.parent.name
        candidate = Path(__file__).resolve().parent / 'pretrain_result' / (dataset + '.npy')
        if not candidate.is_file() or file_hash(candidate) != self.sha256:
            raise FileNotFoundError('Missing matching portable shared vectors: ' + str(candidate))
        self.path = str(candidate.resolve())
        # Verify contents and shape before making the index available.
        self.array

    @property
    def array(self):
        if self._array is None:
            p = Path(self.path)
            stat = p.stat()
            key = (self.path, self.sha256, self.shape, stat.st_size, stat.st_mtime_ns)
            array = _ARRAYS.get(key)
            if array is None:
                if file_hash(p) != self.sha256:
                    raise ValueError('Shared leaf vector checksum mismatch: ' + self.path)
                array = np.load(p, mmap_mode='r', allow_pickle=False)
                if array.dtype != np.float32 or array.shape != self.shape or array.ndim != 2:
                    raise ValueError('Invalid shared leaf vector shape or dtype')
                if not np.isfinite(array).all():
                    raise ValueError('Shared leaf vectors contain NaN or Inf')
                _ARRAYS[key] = array
            self._array = array
        return self._array

    def row(self, node_id):
        if not 0 <= node_id < self.shape[0]:
            raise ValueError('Leaf ID is outside shared embedding rows')
        return self.array[node_id]

    def manifest(self):
        return dict(path=self.path, sha256=self.sha256, shape=list(self.shape),
                    dtype='float32', bytes=Path(self.path).stat().st_size)


def create_shared_vectors(embedding, directory):
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Shared embedding must be a finite 2-D matrix')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.vectors-', suffix='.npy', dir=str(directory))
    try:
        with os.fdopen(fd, 'wb') as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        digest = file_hash(temporary)
        target = directory / (digest + '.npy')
        if target.exists():
            if file_hash(target) != digest:
                raise ValueError('Existing shared vector file is corrupt: ' + str(target))
        else:
            os.replace(temporary, target)
        store = SharedLeafVectors(target, digest, array.shape)
        store.array
        return store
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def share_leaf_vectors(tree, store):
    """Verify all leaves before replacing their inline vectors with one store."""
    stack = [tree.root]
    leaves = []
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        else:
            if len(node.node_ids) != 1:
                raise ValueError('Shared leaf must contain exactly one graph node')
            vector = store.row(int(node.node_ids[0]))
            if not np.array_equal(node.ctx_center, vector):
                raise ValueError('Leaf vector differs from shared embedding')
            leaves.append(node)
    for node in leaves:
        node.__dict__.pop('ctx_center', None)
        node.__dict__.pop('meanembedding', None)
        node._leaf_vectors = store
    tree.leaf_vectors = store
    tree.storage_version = STORAGE_VERSION
    tree.leaf_id_map = {}
    return tree
