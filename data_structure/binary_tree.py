import json


class BTreeNode:
    def __init__(self, data=None, l_child=None, r_child=None, parent=None):
        self.data = data
        self.l_child = l_child
        self.r_child = r_child
        self.parent = parent

    def __len__(self):
        """求树的高度，默认空树高度为0，只有根节点高度为1"""
        if self.data is None:
            return 0
        if self.l_child is None and self.r_child is None:
            return 1
        if self.l_child is None:
            return len(self.r_child) + 1
        if self.r_child is None:
            return len(self.l_child) + 1
        return max(len(self.l_child), len(self.r_child)) + 1

    def __str__(self):
        return self.dumps()

    def __repr__(self):
        res = str(self)
        return res if len(res) <= 100 else res[:100] + '      ...'

    def __eq__(self, other):
        """==重载"""
        return self is other
        # (self.data == other.data and self.l_child == other.l_child and self.r_child == other.r_child)

    def insert(self, x, types=0):
        """
        插入一个节点，默认不判断原先左右子树是否存在，为构建完全二叉树提供便利
        :param x: 值
        :param types: 0/1表示插入为左/右子树
        :return: 插入的BTreeNode
        """
        t = BTreeNode(x, parent=self)
        if types == 0:
            self.l_child = t
        else:
            self.r_child = t
        return t

    def travel_level(self):
        """层次遍历"""
        queue = [self]
        while queue:
            cur = queue.pop(0)
            if cur is not None:
                print(cur.data, end=' ')
                queue.append(cur.l_child)
                queue.append(cur.r_child)
        print()

    def travel_pre(self):
        """先序遍历非递归， 根据递归思想，每次先打印父节点再分别打印左右子树，
        采用栈实现，依次把左子树压入栈，同时把遍历到的左子树父节点打印，至无左子树时再顺序出栈判断右子树是否为空，不为空时继续以当前右子树开始前序遍历"""
        stack = [self]
        cur = self
        while stack:
            # 深度优先，先遍历左子树并直接打印父节点
            print(cur.data, end=' ')
            if cur.l_child is not None:
                cur = cur.l_child
                stack.append(cur)
                continue

            # 遍历到头了， 再逐个取出之前遍历过的节点判断右子树
            while stack:
                cur = stack.pop()
                if cur.r_child is None:
                    continue
                # 右子树不为空，则继续压入栈，并从右子树继续开始先序遍历
                cur = cur.r_child
                stack.append(cur)
                break
        print()

    def morris_in(self):
        """
        Morris 遍历算法整体步骤如下（假设当前遍历到的节点为 x）：
        1.如果 x 无左孩子，则访问 x 的右孩子，即 x =x.right。
        2.如果 x 有左孩子，则找到 x 左子树上最右的节点（即左子树中序遍历的最后一个节点，x 在中序遍历中的前驱节点），我们记为predecessor。根据 predecessor 的右孩子是否为空，进行如下操作。
            a.如果 predecessor 的右孩子为空，则将其右孩子指向 x，然后访问 x 的左孩子，即 x = x.left。
            b.如果 predecessor 的右孩子不为空，则此时其右孩子指向 x，说明我们已经遍历完 x 的左子树，我们将 predecessor 的右孩子置空，然后访问 xx 的右孩子，即 x = x.right。
        3.重复上述操作，直至访问完整棵树。
        Morris 遍历算法整体步骤如下（假设当前遍历到的节点为 xx）：

        作者：LeetCode-Solution
        链接：https://leetcode-cn.com/problems/recover-binary-search-tree/solution/hui-fu-er-cha-sou-suo-shu-by-leetcode-solution/
        来源：力扣（LeetCode）
        著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
        """
        cur = self
        while cur != None:
            predecessor = cur.l_child

            while predecessor != None and predecessor.r_child != None and predecessor.r_child != cur:
                predecessor = predecessor.r_child

            if predecessor == None or predecessor.r_child == cur:
                print(cur.data, end=' ')
                if predecessor != None:
                    predecessor.r_child = None
                cur = cur.r_child

            else:
                predecessor.r_child = cur
                cur = cur.l_child

        print(self)
                
    def travel_in(self):
        """中序遍历非递归， 和前序遍历类似，只是打印顺序不同
        这次也一样深度优先遍历左子树，至末尾时才开始打印"""
        stack = [self]
        cur = self
        while stack:
            if cur.l_child is not None:
                cur = cur.l_child
                stack.append(cur)
                continue
            while stack:
                # 顺序出栈并开始打印
                cur = stack.pop()
                print(cur.data, end=' ')
                if cur.r_child is None:
                    continue
                # 右子树存在时在打印完本结点作为父节点后肯定要打印右子树了，所以继续以右子树开始中序遍历
                cur = cur.r_child
                stack.append(cur)
                break
        print()

    def travel_post(self):
        """后序遍历非递归 左右父 倒过来就是父右左 和先序遍历就很像了 只是从右子树开始遍历 再把最后结果倒过来
        这里用第二种方法：思路和前面一样，遍历到尾时，先判断右子树，因为右子树肯定比父节点先打印,这时由于先打印右子树在判断根节点时要多一个标志位判断该节点是否访问过右子树来决定是否该打印根节点"""
        stack = [self]
        cur = self
        # 多一个标志位表示该节点的右子树是否访问过
        cur.is_visited = False
        while stack:
            if cur.l_child is not None:
                cur = cur.l_child
                cur.is_visited = False
                stack.append(cur)
                continue
            while stack:
                # 先不出栈，因为还不确定右子树是否访问过
                cur = stack[-1]
                # 只有当右子树为空或访问过才出栈并打印
                if cur.r_child is None or cur.is_visited:
                    stack.pop()
                    print(cur.data, end=' ')
                    continue

                # 标志右子树要开始访问了，再次访问到这个节点时右子树肯定访问完了
                cur.is_visited = True

                cur = cur.r_child
                stack.append(cur)
                cur.is_visited = False
                break
        print()

    def _dumps(self):
        """为dumps服务"""
        res = {'data': self.data}
        if self.parent is None:
            res['root'] = True
        if self.l_child is not None:
            res['left'] = self.l_child._dumps()
        if self.r_child is not None:
            res['right'] = self.r_child._dumps()
        return res

    def dumps(self):
        """
        序列化
        :return: 返回str
        """
        return json.dumps(self._dumps())

    @classmethod
    def loads(cls, data, parent=None):
        """
        从序列化后的字符串或字典恢复树
        :param data: dict or str
        :param parent: 默认父节点为空即第一个节点作为根节点
        :return: BTreeNode
        """
        try:
            if isinstance(data, str):
                data = json.loads(data)
            btree = cls(data['data'], parent=parent)
            if data.get('left'):
                btree.l_child = cls.loads(data['left'], btree)
            if data.get('right'):
                btree.r_child = cls.loads(data['right'], btree)
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise e
        else:
            return btree

    @staticmethod
    def load_from_travel(pre, tin, parent=None):
        """
        从中序遍历和任一种其它遍历中恢复树，此为前序，后序类似，默认所有节点值都不一样，通过值就能区分不同节点
        :param pre: 前序遍历结果
        :param tin: 中序遍历结果
        :param parent: 父节点
        :return: BTreeNode
        """
        if len(pre) != len(tin) or len(pre) == 0:
            return

        cur = BTreeNode(pre[0], parent=parent)

        # 中序根据父节点位置划分成两部分，左边为左子树，右边为右子树
        for i in range(len(tin)):
            if tin[i] == pre[0]:
                break
        cur.l_child = BTreeNode.load_from_travel(pre[1:i+1], tin[:i], parent=cur)
        cur.r_child = BTreeNode.load_from_travel(pre[i+1:], tin[i+1:], parent=cur)
        return cur


def all_from_list(lst: list):
    """
    从列表构建完全二叉树
    :param lst: list
    :return: BTreeNode
    """
    if not len(lst):
        return
    root = BTreeNode(lst[0])
    queue = []
    cur = root
    i = 0
    for val in lst[1:]:
        if i == 2:
            cur = queue.pop(0)
            i = 0
        queue.append(cur.insert(val, i))
        i += 1
    return root


def travel_pre(node: BTreeNode) -> None:
    """前序遍历递归
    生成器形式 惰性生值"""
    if node is not None:
        # print(node.data, end=' ')
        yield node.data
        yield from travel_pre(node.l_child)
        yield from travel_pre(node.r_child)


def travel_in(node: BTreeNode) -> None:
    """中序遍历递归"""
    if node is not None:
        yield from travel_in(node.l_child)
        yield node.data
        yield from travel_in(node.r_child)


def travel_post(node: BTreeNode) -> None:
    """后序遍历递归"""
    if node is not None:
        yield from travel_post(node.l_child)
        yield from travel_post(node.r_child)
        # print(node.data, end=' ')
        yield node.data


if __name__ == '__main__':

    # -------------------   测试    --------------------------

    # 生成完全二叉树
    t = all_from_list(list(range(10)))

    # 打印 以序列化字符串形式
    print(t)
    # 用repr限定长度
    print(repr(t))
    # 输出高度
    print(len(t))

    # loads导入一棵树 和t结构一样
    t2 = BTreeNode.loads(str(t))
    print(t2, t == t2, sep='\n')

    # 遍历
    print('层次', end=':')
    t.travel_level()
    print('先序')
    print('递归:', list(travel_pre(t)))
    print('非递归', end=':')
    t.travel_pre()
    print('中序')
    print('递归:', list(travel_in(t)))
    print('非递归', end=':')
    t.travel_in()
    print('morris', end=':')
    t.morris_in()
    print('后序')
    print('递归:', list(travel_post(t)))
    print('非递归', end=':')
    t.travel_post()

    # 从中序和前序恢复
    t3 = BTreeNode.load_from_travel(list(travel_pre(t)), list(travel_in(t)))
    print(t3, t3 == t, sep='\n')

    # -------------------   测试结束 -------------------------
