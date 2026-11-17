# graphtree_dblp_with_embedding.py
# 说明：
# 1) 读入 dblp.edges 构建图
# 2) Louvain 递归构建社区树（根=全图，中间=子图社区，叶子=单节点）
# 3) fix_leaves 保证叶子一定是单节点
# 4) 从 pretrain_result/dblp.npy 加载 embedding（默认第 i 行对应 node_id=i）
# 5) 自底向上给每个 TreeNode 写入 meanembedding
# 6) 构建 leaf_id_map 便于 O(1) 查询叶子节点
# 7) 保存 pkl

import os
import pickle
import networkx as nx
import numpy as np
from cdlib import algorithms


# -----------------------------
# Tree Node
# -----------------------------
class TreeNode:
    def __init__(self, data=None, node_ids=None, meanembedding=None,medoidembedding=None,centroidembedding=None,cent_dict=None, cent_max=None):
        self.data = data
        # 重要：不要用 node_ids if node_ids else []，会把 [] 当 False
        self.node_ids = list(node_ids) if node_ids is not None else []
        self.children = []
        self.parent = None
        self.meanembedding = meanembedding
        self.medoidembedding=medoidembedding# np.ndarray or None
        self.centroidembedding=centroidembedding
        # cent_dict: {node_id: closeness}
        self.cent_dict = dict(cent_dict) if cent_dict is not None else {}

        # cent_max: max_v cent(v) within this node_ids
        # 若没传且 cent_dict 非空，则自动计算
        if cent_max is None and self.cent_dict:
            # 只对 node_ids 里的点取 max，避免 cent_dict 里混了别的键
            vals = [self.cent_dict[v] for v in self.node_ids if v in self.cent_dict]
            self.cent_max = float(max(vals)) if vals else 0.0
        else:
            self.cent_max = float(cent_max) if cent_max is not None else 0.0

    def add_child(self, *child_nodes):
        for child_node in child_nodes:
            self.children.append(child_node)
            child_node.parent = self

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_depth(self):
        if self.is_root():
            return 0
        return self.parent.get_depth() + 1

    def get_height(self):
        if self.is_leaf():
            return 0
        return 1 + max(child.get_height() for child in self.children)


# -----------------------------
# K-ary Tree
# -----------------------------
class KaryTree:
    def __init__(self):
        self.root = None
        self.leaf_id_map = {}  # node_id -> leaf TreeNode

    def set_root(self, root_node):
        self.root = root_node

    def build_leaf_id_map(self, node=None):
        """递归遍历树并构建叶子节点ID到叶子节点的映射"""
        if node is None:
            node = self.root
        if node is None:
            return

        if node.is_leaf():
            # 叶子应该只有一个 node_id，但这里写成通用
            for node_id in node.node_ids:
                self.leaf_id_map[int(node_id)] = node

        for child in node.children:
            self.build_leaf_id_map(child)

    def find_leaf_by_id(self, target_id: int):
        return self.leaf_id_map.get(int(target_id), None)

    def get_height(self):
        if self.root is None:
            return 0
        return self.root.get_height()

    def get_size(self, node=None):
        if node is None:
            node = self.root
        if node is None:
            return 0
        cnt = 1
        for child in node.children:
            cnt += self.get_size(child)
        return cnt

    def display(self, node=None, level=0, max_lines=3000):
        """
        打印树结构（加一个 max_lines 防止爆屏）
        """
        if node is None:
            node = self.root
            self._printed_lines = 0
            print("🌲 树结构显示 🌲")
            print("=" * 50)
            print(f"总高度: {self.get_height()}")
            print(f"总节点数: {self.get_size()}")
            print("-" * 50)

        if getattr(self, "_printed_lines", 0) >= max_lines:
            if getattr(self, "_printed_lines", 0) == max_lines:
                print("...（输出过多，已截断）")
                self._printed_lines += 1
            return

        indent = "    " * level
        if node == self.root:
            node_type = "🌳 根"
            connector = ""
        elif node.is_leaf():
            node_type = "🍂 叶"
            connector = "└── "
        else:
            node_type = "🌿 中"
            connector = "├── "

        depth = node.get_depth()
        children_count = len(node.children)
        node_info = f" [深度:{depth}, 子节点:{children_count}]"

        print(f"{indent}{connector}{node_type}: {node.data}{node_info}")
        self._printed_lines = getattr(self, "_printed_lines", 0) + 1

        for child in node.children:
            self.display(child, level + 1, max_lines=max_lines)

    def fix_leaves(self, node=None):
        """
        保证：叶子节点必须是单节点（len(node_ids)==1）
        如果发现叶子节点 node_ids>1，则将其拆成多个叶子节点挂到 parent 下，并移除原节点
        """
        if node is None:
            node = self.root
        if node is None:
            return

        # 先递归下去（后序更安全，避免遍历过程中修改 children 导致漏/重复）
        for child in list(node.children):
            self.fix_leaves(child)

        # 处理当前节点
        if node.is_leaf() and len(node.node_ids) > 1:
            if node.parent is None:
                # 理论上根不会是“多节点叶子”，但保险起见
                # 直接把根拆成多个叶子（不建议出现这种情况）
                old_ids = list(node.node_ids)
                node.node_ids = []
                for nid in old_ids:
                    node.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))
                return

            parent = node.parent
            old_ids = list(node.node_ids)

            # 在 parent 下添加新的叶子
            for nid in old_ids:
                parent.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))

            # 删除原始“多节点叶子”
            # parent.children.remove(node)


# -----------------------------
# Embedding utils
# -----------------------------
def precompute_closeness_for_tree(
    node: TreeNode,
    G: nx.Graph,
    global_cent: dict = None,
    exact_threshold: int = 1500,   # 子图节点数 <= 这个值才精确算 closeness
):
    """
    给整棵树每个 TreeNode 预计算:
      - node.cent_dict: dict[int, float]  (node_id -> closeness)
      - node.cent_max: float              (max closeness within node_ids)

    exact_threshold: 太大的社区用 global_cent 近似，避免爆时间

    返回:
      global_cent (dict) 便于递归过程中复用
    """

    # 先递归子节点（并接住返回的 global_cent，确保只算一次）
    for ch in node.children:
        global_cent = precompute_closeness_for_tree(ch, G, global_cent, exact_threshold)

    ids = node.node_ids
    if not ids:
        node.cent_dict = {}
        node.cent_max = 0.0
        return global_cent

    # 叶子单点：closeness 用 0.0 即可（Bias 会变成 0，不影响）
    if len(ids) == 1:
        v = int(ids[0])
        node.cent_dict = {v: 0.0}
        node.cent_max = 0.0
        return global_cent

    m = len(ids)

    # 小社区：精确计算
    if m <= exact_threshold:
        subG = G.subgraph(ids)
        cent_dict = nx.closeness_centrality(subG)  # dict: node -> cent（只包含 subG 的节点）
        # 确保 key 是 int，value 是 float（pickle/后续 torch 更稳）
        node.cent_dict = {int(v): float(cent_dict[v]) for v in ids}
        node.cent_max = max(node.cent_dict.values()) if node.cent_dict else 0.0
        return global_cent

    # 大社区：用全图 closeness 近似（只算一次）
    if global_cent is None:
        global_cent = nx.closeness_centrality(G)

    node.cent_dict = {int(v): float(global_cent[v]) for v in ids}
    node.cent_max = max(node.cent_dict.values()) if node.cent_dict else 0.0
    return global_cent



def load_embeddings(npy_path: str) -> np.ndarray:
    emb = np.load(npy_path)
    if emb.ndim != 2:
        raise ValueError(f"embedding 必须是二维 (N,D)，实际是 {emb.shape}")
    return emb


def compute_mean_embeddings(node: TreeNode, emb_matrix: np.ndarray) -> np.ndarray:
    """
    自底向上写入 meanembedding：
    - 叶子：meanembedding = emb[node_id]
    - 中间/根：meanembedding = mean(子节点 meanembedding)
    """
    if node.is_leaf():
        if len(node.node_ids) != 1:
            raise ValueError(f"叶子节点 {node.data} 的 node_ids != 1: {node.node_ids}")
        nid = int(node.node_ids[0])
        node.meanembedding = emb_matrix[nid]
        return node.meanembedding

    child_vecs = []
    for child in node.children:
        v = compute_mean_embeddings(child, emb_matrix)
        if v is not None:
            child_vecs.append(v)

    node.meanembedding = np.mean(child_vecs, axis=0) if child_vecs else None
    return node.meanembedding

def _medoid_index(X: np.ndarray, metric: str = "cosine") -> int:
    """
    返回 X 中的 medoid 索引：使得与其他点距离之和最小的点
    X: (m, d)
    metric: "cosine" 或 "l2"
    """
    m = X.shape[0]
    if m == 0:
        raise ValueError("Empty set for medoid.")
    if m == 1:
        return 0

    if metric == "cosine":
        # cosine distance = 1 - cosine similarity
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T                      # (m, m) cosine sim
        D = 1.0 - S                        # cosine dist
    elif metric == "l2":
        # pairwise squared L2
        G = X @ X.T
        sq = np.sum(X * X, axis=1, keepdims=True)
        D = sq - 2 * G + sq.T              # (m, m)
        D = np.maximum(D, 0.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # 距离和最小
    return int(np.argmin(D.sum(axis=1)))



def compute_medoid_embeddings(node, emb_matrix: np.ndarray, metric: str = "cosine"):
    """
    自底向上写入 node.meanembedding（改为 medoid）：
    - 叶子：meanembedding = emb[node_id]
    - 中间/根：meanembedding = medoid(子节点 meanembedding)
      （即从子节点代表向量中选一个最“中心”的那个）
    """
    if node.is_leaf():
        if len(node.node_ids) != 1:
            raise ValueError(f"叶子节点 {node.data} 的 node_ids != 1: {node.node_ids}")
        nid = int(node.node_ids[0])
        node.medoidembedding = emb_matrix[nid]
        return node.medoidembedding

    child_vecs = []
    for child in node.children:
        v = compute_medoid_embeddings(child, emb_matrix, metric=metric)  # ✅ 递归调用自身
        if v is not None:
            child_vecs.append(v)

    if not child_vecs:
        node.medoidembedding = None
        return None

    X = np.stack(child_vecs, axis=0)  # (num_children, dim)
    midx = _medoid_index(X, metric=metric)
    node.medoidembedding = X[midx]      # ✅ 选一个“真实子向量”作为中心
    return node.medoidembedding

def _closest_to_mean_index(X: np.ndarray, metric: str = "cosine") -> int:
    """
    返回 X 中“离均值最近”的向量索引：
    即 argmin_i d(x_i, mean(X))

    X: (m, d)
    metric: "cosine" 或 "l2"
    """
    m = X.shape[0]
    if m == 0:
        raise ValueError("Empty set for closest-to-mean.")
    if m == 1:
        return 0

    # 均值向量
    centroid = X.mean(axis=0, keepdims=True)  # (1, d)

    if metric == "cosine":
        # cosine distance = 1 - cosine similarity
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        cn = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12)
        sims = (Xn @ cn.T).reshape(-1)          # (m,)
        D = 1.0 - sims                          # (m,)
    elif metric == "l2":
        # squared L2 distance to centroid
        diff = X - centroid                     # (m, d)
        D = np.sum(diff * diff, axis=1)         # (m,)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return int(np.argmin(D))

def compute_closest_to_mean_embeddings(node, emb_matrix: np.ndarray, metric: str = "cosine"):
    """
    自底向上写入 node.centroidembedding（离均值最近的真实向量）：
    - 叶子：centroidembedding = emb[node_id]
    - 中间/根：centroidembedding = argmin d(x_i, mean(children embeddings))
      （从子节点代表向量中选一个最接近均值的真实向量）
    """
    if node.is_leaf():
        if len(node.node_ids) != 1:
            raise ValueError(f"叶子节点 {node.data} 的 node_ids != 1: {node.node_ids}")
        nid = int(node.node_ids[0])
        node.centroidembedding = emb_matrix[nid]
        return node.centroidembedding

    child_vecs = []
    for child in node.children:
        v = compute_closest_to_mean_embeddings(child, emb_matrix, metric=metric)
        if v is not None:
            child_vecs.append(v)

    if not child_vecs:
        node.centroidembedding = None
        return None

    X = np.stack(child_vecs, axis=0)  # (num_children, dim)
    cidx = _closest_to_mean_index(X, metric=metric)
    node.centroidembedding = X[cidx]  # ✅ 真实向量
    return node.centroidembedding

# -----------------------------
# Build tree by Louvain
# -----------------------------
def simple_louvain_dblp(
    edges_file: str,
    emb_path: str = "pretrain_result/dblp.npy",
    output_dir: str = "dblp_results",
    max_depth: int = 5,
    display_tree: bool = False,
):
    # 读取边文件
    G = nx.read_edgelist(edges_file, nodetype=int, create_using=nx.Graph())

    print("图加载完成:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")

    # Louvain（全图）
    communities = algorithms.louvain(G)
    print("\n社区检测结果:")
    print(f"  检测到 {len(communities.communities)} 个社区")

    # 模块度（可选）
    import community as community_louvain
    partition = {node: i for i, comm in enumerate(communities.communities) for node in comm}
    modularity = community_louvain.modularity(partition, G)
    print(f"  模块度: {modularity:.4f}")

    # 建树
    root_node = TreeNode(data="全图", node_ids=list(G.nodes))
    k_tree = KaryTree()
    k_tree.set_root(root_node)

    def build_tree_for_community(comm_nodes, parent_node: TreeNode, depth_left: int):
        # depth_left 到 0 就停止继续划分，把所有节点变成叶子挂上去
        comm_nodes = list(comm_nodes)
        community_node = TreeNode(data=f"社区 ({len(comm_nodes)} 个节点)", node_ids=comm_nodes)
        parent_node.add_child(community_node)

        if depth_left <= 0 or len(comm_nodes) <= 1:
            # 不再划分，直接把节点挂为叶子
            for nid in comm_nodes:
                community_node.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))
            return

        subg = G.subgraph(comm_nodes)
        sub_comms = algorithms.louvain(subg)

        # 如果 Louvain 只给出 1 个社区，说明分不动了：直接挂叶子
        if len(sub_comms.communities) <= 1:
            for nid in comm_nodes:
                community_node.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))
            return

        # 正常情况：递归子社区
        # 先把没被包含到 sub_comms 的节点也作为叶子（保险）
        all_nodes = set(subg.nodes)
        detected = set(n for c in sub_comms.communities for n in c)
        unassigned = all_nodes - detected
        for nid in unassigned:
            community_node.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))

        for subc in sub_comms.communities:
            if len(subc) > 1:
                build_tree_for_community(subc, community_node, depth_left - 1)
            else:
                # 单点社区直接叶子
                nid = list(subc)[0]
                community_node.add_child(TreeNode(data=f"节点 {nid}", node_ids=[int(nid)]))

    # 顶层每个社区接在根下
    for comm in communities.communities:
        build_tree_for_community(comm, root_node, max_depth)

    # 修正叶子（防止出现“社区(6个节点)”这种叶子）
    k_tree.fix_leaves()

    # 建 leaf_id_map（O(1) 查叶子）
    k_tree.build_leaf_id_map()

    # 加载 embedding 并计算 meanembedding
    emb_matrix = load_embeddings(emb_path)
    if emb_matrix.shape[0] < G.number_of_nodes():
        raise ValueError(f"embedding 行数 {emb_matrix.shape[0]} < 图节点数 {G.number_of_nodes()}")

    compute_mean_embeddings(k_tree.root, emb_matrix)
    compute_medoid_embeddings(k_tree.root, emb_matrix)
    compute_closest_to_mean_embeddings(k_tree.root, emb_matrix)
    print("\n开始预计算每个树节点的 closeness centrality（cent_dict）...")
    global_cent = None
    global_cent = precompute_closeness_for_tree(
        k_tree.root,
        G,
        global_cent=global_cent,
        exact_threshold=1500
    )
    print("closeness centrality（cent_dict / cent_max）预计算完成。")

    # 简单统计
    leaf_count = len(k_tree.leaf_id_map)
    print(f"\n叶子节点数量（按 leaf_id_map 去重）: {leaf_count}")
    print(f"树总节点数: {k_tree.get_size()}")
    print(f"根 meanembedding 维度: {k_tree.root.meanembedding.shape if k_tree.root.meanembedding is not None else None}")

    # 可选打印树
    if display_tree:
        k_tree.display(max_lines=2000)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, "k_treefix_meanemb.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(k_tree, f)

    # 保存结构文本（不保存向量，只保存结构）
    txt_path = os.path.join(output_dir, "community_tree_meanemb.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DBLP社区检测树结构\n")
        f.write("=======================\n")
        f.write(f"总节点数: {G.number_of_nodes()}\n")
        f.write(f"总边数: {G.number_of_edges()}\n")
        f.write(f"社区数: {len(communities.communities)}\n")
        f.write(f"模块度: {modularity:.4f}\n\n")

        def write_tree(n: TreeNode, level=0):
            indent = "    " * level
            f.write(f"{indent}{n.data}  ids={len(n.node_ids)}  children={len(n.children)}\n")
            for ch in n.children:
                write_tree(ch, level + 1)

        write_tree(k_tree.root)

    print(f"\n已保存：\n  pkl: {pkl_path}\n  tree txt: {txt_path}")
    return k_tree


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    edges_file = "dataset/dblp/dblp.edges"
    emb_path = "pretrain_result/dblp.npy"

    if not os.path.exists(edges_file):
        print(f"文件不存在: {edges_file}")
        print("请检查文件路径，或提供正确的文件路径")
        raise SystemExit(1)

    if not os.path.exists(emb_path):
        print(f"embedding 文件不存在: {emb_path}")
        print("请检查 pretrain_result/dblp.npy 是否存在")
        raise SystemExit(1)

    simple_louvain_dblp(
        edges_file=edges_file,
        emb_path=emb_path,
        output_dir="dblp_results",
        max_depth=5,
        display_tree=False,
    )
