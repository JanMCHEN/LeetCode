"""
单链表实现，支持环检测，反向等，找出两个无环链表的第一个交点
"""
import copy


class Node:
    """单个节点"""
    def __init__(self, val):
        if isinstance(val, Node):
            val.val = val
        self.val = val

    def __str__(self):
        return str(self.val)


class ListNode:
    """
    单向链表,支持iter、len、str,+,索引，reverse操作
    """
    def __init__(self, val=0, node_list=None):
        """
        单个值时创建头节点，也可以传入节点列表初始化整个链表
        :param val:
        :param node_list:
        """
        if node_list is not None:
            self.head = Node(node_list[0])
            node = self
            for i in node_list[1:]:
                node.next = ListNode(i)
                node = node.next
        else:
            self.head = Node(val)
            self.next = None

    def __iter__(self):
        """
        返回一个可迭代对象, 可通过iter（）调用
        :return:
        """
        node = self
        find_one = self.find_ring()
        count = 0
        while node is not None:
            if node is find_one:
                count += 1
            if count == 2:
                break
            yield node.copy()
            node = node.next

    def __str__(self):
        """用字符串形式表示，str和print（在没有定义__repr__）会调用这个方法, 默认打印所有"""
        val_iter = (str(node.head) for node in iter(self))
        return '>'.join(val_iter)

    def __getitem__(self, item):
        """支持整数和复数索引，不支持切片"""
        assert isinstance(item, int), 'indices must be integers'
        if item < 0:
            item += len(self)
        if item >= len(self) or item < 0:
            raise IndexError('list index out of range')
        node = self
        while item:
            node = node.next
            item -= 1
        return node

    def __len__(self):
        """获取链表长度，len会调用这个方法"""
        count = 0
        for _ in iter(self):
            count += 1
        return count

    def __reversed__(self):
        """将链表反向并返回头节点，reversed调用"""
        if len(self) < 2:
            return self
        node_iter = iter(self)
        p1 = next(node_iter)
        p1.next = None
        for node in node_iter:
            node.next = p1
            p1 = node
        return p1

    def __add__(self, other):
        """重载+运算符"""
        node = self.copy()
        node[-1].next = other
        return node

    def find_ring(self):
        """判断是否存在环并找出"""
        p1 = self
        p2 = self
        find = False
        while p2 is not None:
            p1 = p1.next
            p2 = p2.next
            if not find and p2 is not None:
                p2 = p2.next
            if p1 is p2:
                if find:
                    return p1
                find = True
                p1 = self

    def append(self, node):
        """
        往链表末尾新增节点
        :param node: Node
        :return:
        """
        self[-1].next = ListNode(node)

    def copy(self):
        """
        浅拷贝一份链表
        :return:
        """
        return copy.copy(self)


def find_repeat_node(one, other):
    """
    找到两个无环链表第一个相交节点
    :type one: ListNode
    :param one: 较长链表
    :param other: 较短链表
    :return: ListNode
    """
    # 方法1：较长链表向后偏移至链表长度相同，再同时偏移，直至相等则为交点
    l1, l2 = len(one), len(other)
    if l1 < l2:
        return find_repeat_node(other, one)
    for _ in range(l1-l2):
        one = one.next
    while one is not None:
        if one is other:
            return one
        one = one.next
        other = other.next

    # # 方法2，直接把其中一个链表拼接到另一个末尾，有环则必相交，环入口则为第一个交点
    # new_one = one + other
    # return new_one.find_ring()


if __name__ == '__main__':
    # 创建链表
    list_node = ListNode(node_list=list(range(20)))
    print(list_node, len(list_node), '反向：', reversed(list_node))

    # 新增元素
    list_node.append(0)
    print(list_node)

    # 根据位置获取链表
    list_node2 = list_node[10]
    print(list_node2, len(list_node2))

    # 找两个链表交点
    print(find_repeat_node(list_node, list_node2))

