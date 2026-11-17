import pickle
import gc
import tempfile
import unittest
from pathlib import Path
import numpy as np
from index_tree_schema import TreeNode, KaryTree, validate_tree
from shared_leaf_vectors import create_shared_vectors, share_leaf_vectors


def example(embedding):
    root=TreeNode(range(len(embedding)));root.ctx_center=embedding.mean(axis=0)
    for i,vector in enumerate(embedding):
        leaf=TreeNode([i]);leaf.ctx_center=vector.copy();root.add_child(leaf)
    return KaryTree(root)


class SharedLeafTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.emb=np.arange(4*128,dtype=np.float32).reshape(4,128)
        self.store=create_shared_vectors(self.emb,self.temp.name)

    def test_legacy_roundtrip_and_shared_roundtrip(self):
        old=example(self.emb);legacy=pickle.dumps(old)
        old=pickle.loads(legacy)
        self.assertTrue(np.array_equal(old.root.children[1].ctx_center,self.emb[1]))
        share_leaf_vectors(old,self.store)
        compact=pickle.dumps(old)
        self.assertLess(len(compact),len(legacy))
        a,b=pickle.loads(compact),pickle.loads(compact)
        self.assertIs(a.leaf_vectors.array,b.leaf_vectors.array)
        for i,leaf in enumerate(a.root.children):
            self.assertNotIn('ctx_center',leaf.__dict__)
            np.testing.assert_array_equal(leaf.ctx_center,self.emb[i])
            self.assertTrue(np.shares_memory(leaf.ctx_center,a.leaf_vectors.array))
            self.assertFalse(leaf.ctx_center.flags.writeable)
        self.assertTrue(validate_tree(a,range(4),128)['coverage_complete'])

    def test_mismatch_rejected_without_modifying_tree(self):
        tree=example(self.emb);tree.root.children[-1].ctx_center+=1
        with self.assertRaisesRegex(ValueError,'differs'):share_leaf_vectors(tree,self.store)
        self.assertNotIn('_leaf_vectors',tree.root.children[0].__dict__)

    def test_corrupt_file_rejected(self):
        tree=share_leaf_vectors(example(self.emb),self.store);payload=pickle.dumps(tree)
        # Use a different path to avoid mutating a memory-mapped file on Windows.
        bad=Path(self.temp.name)/'bad.npy';np.save(bad,self.emb+1)
        tree.leaf_vectors.path=str(bad)
        with self.assertRaisesRegex(ValueError,'checksum'):pickle.loads(pickle.dumps(tree))

    def test_missing_file_rejected(self):
        tree=share_leaf_vectors(example(self.emb),self.store)
        tree.leaf_vectors.path=str(Path(self.temp.name)/'missing.npy')
        with self.assertRaises(FileNotFoundError):pickle.loads(pickle.dumps(tree))

    def tearDown(self):
        # Release Windows mmap handles before TemporaryDirectory cleanup.
        self.store._array=None
        gc.collect()


if __name__=='__main__':unittest.main()
