import random

from data_structure.binary_tree import BTreeNode, travel_in


class BST(BTreeNode):
    """二叉搜索树
    对于树中任一节点r，其左/右子树均小于/大于r，现假设左子树小于r，右子树大于等于r"""

    def match(self, x):
        """从当前节点开始搜索，找第一个匹配的，未找到则返回None"""
        if x == self.data:
            return self
        if x < self.data and self.l_child is not None:
            return self.l_child.match(x)
        if x >= self.data and self.r_child is not None:
            return self.r_child.match(x)

    def insert(self, x, types=True):
        """插入的时候不能破坏这一结构, 时间复杂度为lgh，h为树高度"""
        if types and self.parent is not None:
            # 先找到根节点
            return self.parent.insert(x)
        if x < self.data:
            if self.l_child is None:
                self.l_child = BST(x, parent=self)
            else:
                self.l_child.insert(x, False)
        else:
            if self.r_child is None:
                self.r_child = BST(x, parent=self)
            else:
                self.r_child.insert(x, False)

    def remove(self, x):
        at = self.match(x)
        if at is None:
            return False
        self._remove_at(at)
        return True

    def _remove_at(self, node):
        """移除某个节点并不破坏结构"""
        left = node.l_child
        right = node.r_child

        # 左右子树均为空，直接移除
        if left is None and right is None:
            if node.parent is None:
                node.data = None
            else:
                p = node.parent
                if p.l_child is not None and p.l_child.data == node.data:
                    p.l_child = None
                else:
                    p.r_child = None

        # 左右子树存在一个，用子树节点代替自己
        elif left is None:
            node.data = right.data
            node.r_child = right.r_child
            node.l_child = right.l_child
        elif right is None:
            node.data = left.data
            node.r_child = left.r_child
            node.l_child = left.l_child

        # 左右子树均存在则用右子树最小值代替自己并把自己移除
        else:
            while right.l_child is not None:
                right = right.l_child

            node.data = right.data
            self._remove_at(right)

    @classmethod
    def load_from_list(cls, lst):
        if len(lst) == 0:
            return
        root = cls(lst[0])
        for v in lst[1:]:
            ret = root.insert(v)
            if ret is not None:
                root = ret
        return root


class AVL(BST):
    """高度平衡二叉搜索树，BST改进版，由于BST在给定序列基本有序时会退化成链表
    多了个限定条件，即任一节点左右子树高度差值绝对值<=1"""
    def __init__(self, data=None, l_child=None, r_child=None, parent=None):
        super().__init__(data, l_child, r_child, parent)
        self.balance_f = 0  # 平衡因子即左右子树高度差

    def _zig(self):
        """顺时针旋转， 保持中序遍历结果一样"""
        # 把左节点作为新的父节点
        cur = self.l_child

        # 根据中序遍历交换子树并更新父指针
        cur.parent, self.parent = self.parent, cur
        self.l_child = cur.r_child
        if self.l_child is not None:
            self.l_child.parent = self
        cur.r_child = self

        # 更新平衡因子
        cur_ll = 0 if cur.l_child is None else len(cur.l_child)
        cur.balance_f = cur_ll - len(cur.r_child)
        self_ll = 0 if self.l_child is None else len(self.l_child)
        self_rl = 0 if self.r_child is None else len(self.r_child)
        self.balance_f = self_ll - self_rl

        # # 如果是根节点则返回新的根节点，否则原地修改
        if cur.parent is None:
            return cur
        else:
            if cur.parent.l_child is not None and cur.parent.l_child.data == self.data:
                cur.parent.l_child = cur
            else:
                cur.parent.r_child = cur

    def _zag(self):
        """逆时针旋转， 同顺时针旋转"""
        cur = self.r_child
        cur.parent, self.parent = self.parent, cur
        self.r_child = cur.l_child
        if self.r_child is not None:
            self.r_child.parent = self
        cur.l_child = self

        # 更新平衡因子
        cur_rl = 0 if cur.r_child is None else len(cur.r_child)
        cur.balance_f = len(cur.l_child) - cur_rl
        self_ll = 0 if self.l_child is None else len(self.l_child)
        self_rl = 0 if self.r_child is None else len(self.r_child)
        self.balance_f = self_ll - self_rl

        if cur.parent is None:
            return cur
        else:
            if cur.parent.l_child is not None and cur.parent.l_child.data == self.data:
                cur.parent.l_child = cur
            else:
                cur.parent.r_child = cur

    def _insert(self, x, types=True):
        """
        插入的辅助方法，同BST，只是更新了一下平衡因子
        :param x:
        :param types:
        :return:
        """
        if types and self.parent is not None:
            # 先找到根节点
            return self.parent._insert(x)
        if x < self.data:
            if self.l_child is None:
                self.l_child = AVL(x, parent=self)
                self.balance_f += 1
                return self
            return self.l_child._insert(x, False)
        else:
            if self.r_child is None:
                self.r_child = AVL(x, parent=self)
                self.balance_f -= 1
                return self
            return self.r_child._insert(x, False)

    def insert(self, x, types=True):
        # 获取插入节点的父节点
        cur = self._insert(x, types)

        # 如果平衡因子为1或-1则有可能改变某个上层节点平衡因子不在规定范围
        # 从当前节点向上更新节点的平衡因子，直到有+-2失衡或0就不会改变父节点的平衡因子了
        while cur.balance_f != 0 and cur.parent is not None:
            tmp = cur
            cur = cur.parent

            # 左孩子节点则平衡因子加1，否则为右孩子节点
            if cur.l_child is not None and cur.l_child.data == tmp.data:
                cur.balance_f += 1
            else:
                cur.balance_f -= 1

            # 失衡需作调整，且调整后不会改变上层节点平衡因子
            if cur.balance_f == 2 or cur.balance_f == -2:
                break

        # 没失衡
        if cur.balance_f in (0, -1, 1):
            return

        # 失衡4种情况，两对称
        # 由左子树的平衡因子为1引起的当前平衡因子为2失衡，此时易知是左子树的左子树较深，经过一次右旋即可
        if cur.balance_f == 2 and cur.l_child is not None and cur.l_child.balance_f == 1:
            res = cur._zig()

        # 由左子树的平衡因子为-1引起的当前平衡因子为2失衡，左旋
        elif cur.balance_f == -2 and cur.r_child is not None and cur.r_child.balance_f == -1:
            res = cur._zag()

        # 由左子树的平衡因子为-1引起的当前平衡因子为2失衡，此时需旋转两次
        elif cur.balance_f == 2 and cur.l_child is not None and cur.l_child.balance_f == -1:
            cur.l_child._zag()
            res = cur._zig()

        # 和上一种对称
        else:
            cur.r_child._zig()
            res = cur._zag()

        return res

    # TODO
    def _remove_at(self, node):
        """删除某个节点， 也会导致失衡"""


if __name__ == '__main__':
    lst = list(range(100))
    random.shuffle(lst)
    bst = BST.load_from_list(lst)
    avl = AVL.load_from_list(lst)
    print('bst:', len(bst), repr(bst))
    print('bst-remove:', bst.remove(50), len(bst), list(travel_in(bst)))
    print('avl:', len(avl), repr(avl))
