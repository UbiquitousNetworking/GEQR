"""Regression tests for strict v2 partition-tree validation."""
from __future__ import annotations

import unittest

import numpy as np

from index_tree_schema import KaryTree, TreeNode, TreeValidationError, validate_tree


def cached(node: TreeNode) -> TreeNode:
    node.ctx_center = np.zeros(2, dtype=np.float32)
    node.ctx_size_score = 0.0
    node.ctx_coh_score = 0.0
    return node


def valid_tree() -> KaryTree:
    root = cached(TreeNode((0, 1)))
    root.add_child(cached(TreeNode((0,))))
    root.add_child(cached(TreeNode((1,))))
    return KaryTree(root)


class ValidateTreeTests(unittest.TestCase):
    def test_accepts_complete_unique_partition(self):
        result = validate_tree(valid_tree(), {0, 1}, embedding_dim=2)
        self.assertTrue(result["coverage_complete"])
        self.assertTrue(result["unique_leaves"])
        self.assertTrue(result["partition_valid"])

    def test_rejects_incomplete_coverage(self):
        with self.assertRaisesRegex(TreeValidationError, "Root coverage mismatch"):
            validate_tree(valid_tree(), {0, 1, 2}, embedding_dim=2)

    def test_rejects_duplicate_leaf(self):
        tree = valid_tree()
        tree.root.children[1].node_ids = (0,)
        with self.assertRaises(TreeValidationError):
            validate_tree(tree, {0, 1}, embedding_dim=2)

    def test_rejects_parent_child_set_mismatch(self):
        tree = valid_tree()
        tree.root.children.pop()
        with self.assertRaisesRegex(TreeValidationError, "Parent coverage"):
            validate_tree(tree, {0, 1}, embedding_dim=2)

    def test_rejects_broken_parent_pointer(self):
        tree = valid_tree()
        tree.root.children[0].parent = None
        with self.assertRaisesRegex(TreeValidationError, "Broken child.parent"):
            validate_tree(tree, {0, 1}, embedding_dim=2)

    def test_rejects_repeated_node_object(self):
        tree = valid_tree()
        repeated = tree.root.children[0]
        tree.root.children[1] = repeated
        with self.assertRaises(TreeValidationError):
            validate_tree(tree, {0, 1}, embedding_dim=2)


if __name__ == "__main__":
    unittest.main()
