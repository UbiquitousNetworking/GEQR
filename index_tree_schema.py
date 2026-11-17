"""Versioned, minimal schema for CSQR partition-tree indexes."""
from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Optional

import numpy as np


SCHEMA_VERSION = "csqr-partition-tree-v2"
CACHE_VERSION = "csqr-context-cache-v1"


class TreeValidationError(ValueError):
    pass


class TreeNode:
    """A partition-tree node containing only fields used by online CSQR."""

    def __init__(self, node_ids: Iterable[int]):
        self.node_ids = tuple(int(v) for v in node_ids)
        self.children = []
        self.parent = None
        self.ctx_center = None
        self.ctx_size_score = 0.0
        self.ctx_coh_score = 0.0

    @property
    def ctx_center(self):
        store = self.__dict__.get('_leaf_vectors')
        if store is not None:
            return store.row(int(self.node_ids[0]))
        # Legacy pickles keep their original field without conversion.
        return self.__dict__.get('ctx_center')

    @ctx_center.setter
    def ctx_center(self, value):
        self.__dict__.pop('_leaf_vectors', None)
        self.__dict__['ctx_center'] = value

    def add_child(self, child: "TreeNode") -> None:
        if child.parent is not None:
            raise TreeValidationError("A tree node cannot have multiple parents")
        self.children.append(child)
        child.parent = self

    def is_leaf(self) -> bool:
        return not self.children

    def is_root(self) -> bool:
        return self.parent is None


class KaryTree:
    schema_version = SCHEMA_VERSION
    cache_version = CACHE_VERSION

    def __init__(self, root: Optional[TreeNode] = None):
        self.root = root
        self.leaf_id_map = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['leaf_id_map'] = {}
        return state

    def set_root(self, root: TreeNode) -> None:
        root.parent = None
        self.root = root

    def build_leaf_id_map(self, reject_duplicates: bool = True):
        """Rebuild the map from scratch and reject ambiguous leaf paths."""
        self.leaf_id_map = {}
        if self.root is None:
            return self.leaf_id_map
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(reversed(node.children))
                continue
            if len(node.node_ids) != 1:
                raise TreeValidationError(
                    f"Leaf must contain exactly one node ID, got {len(node.node_ids)}"
                )
            node_id = int(node.node_ids[0])
            if reject_duplicates and node_id in self.leaf_id_map:
                raise TreeValidationError(f"Duplicate leaf node ID: {node_id}")
            self.leaf_id_map[node_id] = node
        return self.leaf_id_map

    def find_leaf_by_id(self, target_id: int):
        return self.leaf_id_map.get(int(target_id))

    def get_size(self) -> int:
        if self.root is None:
            return 0
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children)
        return total

    def get_height(self) -> int:
        if self.root is None:
            return 0
        height = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            height = max(height, depth)
            stack.extend((child, depth + 1) for child in node.children)
        return height


def validate_tree(
    tree: KaryTree,
    expected_node_ids: Iterable[int],
    embedding_dim: Optional[int] = None,
    require_partition: bool = True,
):
    """Validate coverage, unique leaves, parent links and partition invariants."""
    if tree.root is None:
        raise TreeValidationError("Tree has no root")
    if tree.root.parent is not None:
        raise TreeValidationError("Root parent must be None")

    expected = set(int(v) for v in expected_node_ids)
    root_ids = tuple(int(v) for v in tree.root.node_ids)
    if len(root_ids) != len(set(root_ids)):
        raise TreeValidationError("Root contains duplicate node IDs")
    if set(root_ids) != expected:
        missing = sorted(expected - set(root_ids))[:10]
        extra = sorted(set(root_ids) - expected)[:10]
        raise TreeValidationError(f"Root coverage mismatch; missing={missing}, extra={extra}")

    seen_objects = set()
    leaf_occurrences = []
    internal_nodes = 0
    max_depth = 0
    stack = [(tree.root, 0)]
    while stack:
        node, depth = stack.pop()
        identity = id(node)
        if identity in seen_objects:
            raise TreeValidationError("Cycle or shared TreeNode object detected")
        seen_objects.add(identity)
        max_depth = max(max_depth, depth)
        ids = tuple(int(v) for v in node.node_ids)
        if not ids:
            raise TreeValidationError("Empty tree node detected")
        if len(ids) != len(set(ids)):
            raise TreeValidationError("Duplicate node ID inside one tree node")
        if not set(ids).issubset(expected):
            raise TreeValidationError("Tree node contains an unknown node ID")

        center = getattr(node, "ctx_center", None)
        if center is None:
            raise TreeValidationError("Missing ctx_center")
        center = np.asarray(center)
        if center.ndim != 1 or (embedding_dim is not None and center.shape[0] != embedding_dim):
            raise TreeValidationError(f"Invalid ctx_center shape: {center.shape}")
        if not np.all(np.isfinite(center)):
            raise TreeValidationError("ctx_center contains NaN or Inf")
        for field in ("ctx_size_score", "ctx_coh_score"):
            value = float(getattr(node, field, float("nan")))
            if not math.isfinite(value):
                raise TreeValidationError(f"{field} is not finite")

        if not node.children:
            if len(ids) != 1:
                raise TreeValidationError("A leaf must contain exactly one node ID")
            leaf_occurrences.append(ids[0])
            continue

        internal_nodes += 1
        child_sets = []
        for child in node.children:
            if child.parent is not node:
                raise TreeValidationError("Broken child.parent link")
            child_set = set(int(v) for v in child.node_ids)
            child_sets.append(child_set)
            stack.append((child, depth + 1))
        union = set().union(*child_sets)
        if union != set(ids):
            raise TreeValidationError("Parent coverage differs from union of child coverage")
        if require_partition:
            total_memberships = sum(len(values) for values in child_sets)
            if total_memberships != len(union):
                raise TreeValidationError("Sibling coverage overlaps in a partition tree")

    counts = Counter(leaf_occurrences)
    duplicates = sorted(v for v, count in counts.items() if count != 1)
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    if duplicates or missing or extra:
        raise TreeValidationError(
            f"Leaf coverage invalid; missing={missing[:10]}, extra={extra[:10]}, "
            f"duplicate_ids={duplicates[:10]}"
        )
    tree.build_leaf_id_map(reject_duplicates=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "tree_nodes": len(seen_objects),
        "internal_nodes": internal_nodes,
        "leaf_nodes": len(leaf_occurrences),
        "unique_leaf_ids": len(counts),
        "max_depth": max_depth,
        "coverage_complete": True,
        "unique_leaves": True,
        "partition_valid": bool(require_partition),
    }
