"""
旅行商问题
给定n个城市之间两两距离，计算只经过所有城市一遍又回到起点的最短距离
动态规划
具有最优子结构故能用动态规划

dp[i][j]其中j表示一个经过哪些城市的集合，这里可以用二进制01来表示城市的两个状态，这样就转化到一个整数上去了
dp[i][j]表示i城市经过j这些城市又回到起点的最短距离，先假设j就是一个集合，可以作差集运算，因为差集可以通过二进制运算得到

动态方程：dp[i][j] = min(city[i][k] + dp[k][j-k])其中k表示j当中的一个元素，j-k表示差集
例如：n=3，dp[i]{} = city[i][0], 即任一城市没经过任何城市到起点0的距离即为城市i直接到0距离
            dp[0]{1, 2} = min(city[0][1] + dp[1]{2}, city[0][2] + dp[2]{1})
            dp[1]{2} = city[1][2] + dp[2]{},   dp[2]{1} = city[2][1] + dp[1]{}
            …………………………依此类推
"""

# n = int(input())
#
# city = [list(map(int, input().split())) for _ in range(n)]

n = 3
city = [
    [0, 1, 3],
    [1, 0, 6],
    [3, 6, 0]
]

# 总共n个城市，需要n个二进制位来描述状态
V = 1 << (n-1)

# dp[i][j], i最大位n-1， j表示集合满集为n个1即V
dp = [[float('inf')] * V for _ in range(n)]

# 先初始化所有城市当经过一个空集时对应的距离
for i in range(n):
    dp[i][0] = city[0][i]

# j从1开始到除了起始城市全为1，如果n=4，就是从0001到0111，假设最高位就是起始城市0
for j in range(1, V):

    # 如果j包含了i则直接跳过
    for i in range(n):
        if i and (j >> (i-1)) & 1 == 1:
            continue

        for k in range(1, n):
            if (j >> (k-1)) & 1 == 0:       # 判断城市k是不是此时j的子集，j >> (k-1) & 1可以把第k位的数移到第一位和1相与，即位判断k位是否为1
                continue

            # 判断是子集就带入动态方程，相当于求所有子集的最小值， j ^ (1 << (k-1))相当于先把1移到第k位上，再和j异或，即位作差集
            dp[i][j] = min(dp[i][j], city[i][k] + dp[k][j ^ (1 << (k-1))])

# 最后把从0经过其它所有城市回到0的值输出即可
print(dp[0][V-1])