from types import List

class UnionFindSet:
    def __init__(self, n):
        self.parent = [i for i in range(n+1)]
        self.rank = [1] * (n+1)

    def find(self, x):
        if x == self.parent[x]:
            return x
        
        # 路径压缩
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # 先找到根节点
        xp, yp = self.find(x), self.find(y)
        if xp == yp:
            return False

        # 按秩合并
        if self.rank[xp] <= self.rank[yp]:
            self.parent[xp] = yp
        else:
            self.parent[yp] = xp
        
        # 如果秩相同并且根节点不同，新的根节点秩加1
        if self.rank[xp] == self.rank[yp]:
            self.rank[yp] += 1
        return True


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        """#684 冗余连接
        示例 1：
        输入: [[1,2], [1,3], [2,3]]
        输出: [2,3]
        解释: 给定的无向图为:
         1
        / \
        2-3
        """
        union_set = UnionFindSet(len(edges))
        for u, v in edges:
            if not union_set.union(u, v):
                return [u, v]

    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        """685. 冗余连接 II
        有向图
        """
        nodesCount = len(edges)
        uf = UnionFindSet(nodesCount)
        parent = list(range(nodesCount + 1))
        conflict = -1
        cycle = -1
        for i, (u, v) in enumerate(edges):
            # 入度大于1
            if parent[v] != v:
                conflict = i
            else:
                parent[v] = u
                # 找环
                if not uf.union(u, v):
                    cycle = i
        if conflict < 0:
            # 入度全为1，环最后1个边
            return [edges[cycle][0], edges[cycle][1]]
        else:
            # 有入度为2的
            conflictEdge = edges[conflict]
            if cycle >= 0:
                return [parent[conflictEdge[1]], conflictEdge[1]]
            else:
                return [conflictEdge[0], conflictEdge[1]]
