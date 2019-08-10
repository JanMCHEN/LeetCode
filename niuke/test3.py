"""
给定N（可选作为埋伏点的建筑物数）、D（相距最远的两名特工间的距离的最大值）以及可选建筑的坐标，计算在这次行动中，大锤的小队有多少种埋伏选择。
注意：
1. 两个特工不能埋伏在同一地点
2. 三个特工是等价的：即同样的位置组合(A, B, C) 只算一种埋伏方法，不能因“特工之间互换位置”而重复使用
"""

# 总算遇到个简单题了
# 就是做排列组合，数据按顺序排好了，从头开始，从后面找出在距离范围内的点，从中随便选两个共有C n 2种方法；再继续往后

N, D = list(map(int, input().split()))
pos = list(map(int, input().split()))

# 存储待求排列组合的数
tmp = []

# 从第1个开始找
r = 1

for i in range(N):
    for j in range(r, N):
        if pos[i] + D < pos[j]:
            break
        if j == N - 1:
            j = N
            break

    # 每次找到最后一个位置那么下一次从这个位置开始找
    r = j
    # 数量
    tmp.append(j - i - 1)
ans = 0
for t in tmp:
    # 对大于1的求组合才有意义
    if t > 1:
        ans += t * (t - 1) / 2
        ans %= 99997867
print(int(ans))
