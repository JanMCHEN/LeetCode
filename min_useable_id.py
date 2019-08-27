"""
找出最小可用ID, 假设id为非负整数，现在需要找到最小可用的id，及不在给定列表中的最小整数
"""
#


def violent(lst):
    """最傻的解法，直接遍历列表找第一个不在列表里的数，复杂度O(n2)"""
    res = 0
    while True:
        if res not in lst:
            yield res
        res += 1


def beauty(lst):
    """利用数据特点，任意lst如果存在小于len(lst)的可用整数，必然存在某个值大于n，否则即为1至n-1的一个排列
    构造一个长度n+1数组用来保存[0,n]是否可用，可用的条件是<n，最后再找出第一位不可用即为最后结果，考虑到数组中每个值只有两种状态，故可用二进制0/1替代数组
    时间复杂度O(n),空间上由于用二进制保存的状态，复杂度O（n），但数组较大时二进制数值非常大，当数值过大时可以改用数组"""
    s = len(lst)
    dp = 0   # 先初始化为0，表示可用

    for v in lst:
        if v < s:
            dp |= 1 << v

    # 从低位开始找到第一个为0的位即为结果
    i = 0
    while True:
        if dp >> i & 1 == 0:
            yield i
        i += 1


def division(lst):
    """二分法，每次可以以2/n为界分成两部分，如过左边长度为2/n则在右边，否则在左边，这样就可以递归查找
    但是只能找到一个，不能写成迭代器的形式
    时间复杂度O(2/n+4/n+8/n+...)=O(n),空间复杂度O(1),原地修改数组"""
    l = 0
    r = len(lst)
    while r - l > 1:
        l_ = l
        m = (r + l - 1) // 2
        for i in range(l, r):
            if lst[i] <= m:
                lst[i], lst[l_] = lst[l_], lst[i]
                l_ += 1
        if l_ == m + 1:
            l = m + 1
        else:
            r = m + 1
    return l


def main(size=1000):
    import random

    # 构建测试数据
    test = list(range(size))
    random.shuffle(test)
    for i in range(random.randint(1, size//1000+1)):
        test[i] += size

    r1, r2, r3 = violent(test), beauty(test), division(test)
    print(next(r1), next(r2), r3)


if __name__ == '__main__':
    main(1000000)
