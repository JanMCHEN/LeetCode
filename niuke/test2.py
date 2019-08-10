"""
具体的规则如下：

总共有36张牌，每张牌是1~9。每个数字4张牌。
你手里有其中的14张牌，如果这14张牌满足如下条件，即算作和牌
14张牌中有2张相同数字的牌，称为雀头。
除去上述2张牌，剩下12张牌可以组成4个顺子或刻子。顺子的意思是递增的连续3个数字牌（例如234,567等），刻子的意思是相同数字的3个数字牌（例如111,777）

例如：
1 1 1 2 2 2 6 6 6 7 7 7 9 9 可以组成1,2,6,7的4个刻子和9的雀头，可以和牌
1 1 1 1 2 2 3 3 5 6 7 7 8 9 用1做雀头，组123,123,567,789的四个顺子，可以和牌
1 1 1 2 2 2 3 3 3 5 6 7 7 9 无论用1 2 3 7哪个做雀头，都无法组成和牌的条件。

现在，小包从36张牌中抽取了13张牌，他想知道在剩下的23张牌中，再取一张牌，取到哪几种数字牌可以和牌。
"""

# 碰到这种题我也没办法
# 只想到一个一个试，迭代9张牌分别加到牌面里，看能否和牌
# 关键是判断一幅牌是否为和牌，题目意思是组成顺子和刻子加起来为4就可以，刚开始理解错了以为只能是4个顺子或刻子
# 判断策略是循环1-9分别为雀头，判断剩下的牌是否符合条件

from collections import Counter


def enable():
    """
    能和牌就返回True
    :return:
    """
    for i in range(1, 10):
        # 把牌面复制一下
        count = puk_count.copy()

        # 为雀头的牌数量减2
        count[i] -= 2

        # 小于0即原来数量小于2肯定当不了雀头
        if count[i] < 0:
            continue

        # 现在开始迭代整幅牌看能否找找顺子或刻子一直找下去
        for j in range(1, 10):

            # 因为是从小到大迭代的，一张牌的数量大于3，直接减去3表示组了一个顺子
            if count[j] >= 3:
                count[j] -= 3

            # 如果牌数量大于0，且后两种牌数量都大于自己，就减去自己，表示组成了刻子
            if count[j + 1] >= count[j] > 0 and count[j + 2] >= count[j] > 0:
                count[j + 2] -= count[j]
                count[j + 1] -= count[j]

            # 能和牌的条件为迭代到最后一张牌且数量为0，因为其它情况都break出去了
            elif count[j] == 0:
                if j == 9:
                    return True
            else:
                break

    # 迭代完还没和牌则不能和
    return False


puk = list(map(int, input().split()))
puk_count = Counter(puk)
res = []
for p in range(1, 10):
    if puk_count[p] == 4:
        continue
    puk_count[p] += 1
    if enable():
        res.append(str(p))
    puk_count[p] -= 1
if not res:
    res.append('0')
print(' '.join(res))
