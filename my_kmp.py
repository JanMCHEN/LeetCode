class KMP:
    def __init__(self, p):
        # 待匹配的字符串
        self.p = p

        # 用于存储最长前缀后缀
        self.next = [-1 for _ in p]

        # 计算最长后缀
        self.compute_prefix()
        # print(self.next)

    def compute_prefix(self):
        """
        计算待匹配字符串每个位置的前缀后缀长度
        :return:
        """
        for i in range(1, len(self.p)):
            k = self.next[i-1]
            while k >= 0 and self.p[k+1] != self.p[i]:
                k = self.next[k]
            if self.p[k+1] == self.p[i]:
                self.next[i] = k + 1

    def match(self, string):
        """
        匹配字符串第一次出现的位置，未匹配返回-1
        :param string: str：被查找的字符串
        :return: int
        """
        m = len(self.p)
        q = -1
        for i, c in enumerate(string):
            while q > 0 and self.p[q+1] != c:
                q = self.next[q]
            if self.p[q+1] == c:
                q += 1
            if q == m - 1:
                return i - m + 1
        return -1


if __name__ == '__main__':
    kmp = KMP('abaabbabaab')
    ret = kmp.match('abjkhahjkhjashxbbcccabaabababbabaab')
    if ret > -1:
        print('success match at position {}'.format(ret))
    else:
        print('not matched')
