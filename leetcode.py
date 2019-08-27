from typing import List
from random import randint
import time
from my_kmp import KMP


class TreeNode:
    """树节点"""
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode:
    """
    单向链表
    """
    def __init__(self, x=0, node_list=[]):
        if node_list:
            self.val = node_list[0]
            node = self
            for i in node_list[1:]:
                node.next = ListNode(i)
                node = node.next
        else:
            self.val = x
            self.next = None

    def show(self):
        node = self
        node_list = []
        while node:
            node_list.append(node.val)
            node = node.next
        print(node_list)


class LFUCache:
    """
    设计并实现最不经常使用（LFU）缓存的数据结构。它应该支持以下操作：get 和 put。
    get(key) - 如果键存在于缓存中，则获取键的值（总是正数），否则返回 -1。
    put(key, value) - 如果键不存在，请设置或插入值。当缓存达到其容量时，它应该在插入新项目之前，使最不经常使用的项目无效。
    在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，最近最少使用的键将被去除。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq_key = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        # 找出当前key的频率并加1
        for freq in self.freq_key:
            if key in self.freq_key[freq]:
                self.freq_key[freq].remove(key)
                if not self.freq_key[freq]:
                    self.freq_key.pop(freq)
                self.freq_key.setdefault(freq + 1, []).append(key)
                break

        return self.cache.get(key)

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key not in self.cache:
            if len(self.cache) == self.capacity:
                # 弹出频率最小的第一个元素，并移除
                pop_key = self.freq_key[min(self.freq_key)].pop(0)
                self.cache.pop(pop_key)

            self.cache[key] = value
            self.freq_key.setdefault(1, []).append(key)
        else:
            self.cache[key] = value
            self.get(key)


class Solution(object):
    """
    各种算法实现
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """#1 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
        你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
        思路：利用字典保存值和索引，遍历数组，每次发现目标值不在字典里则继续寻找后面的并添加进字典里"""
        nums_dict = {}
        ret = []
        for i in range(len(nums)):
            if target - nums[i] not in nums_dict:
                nums_dict[nums[i]] = i
            else:
                ret = [nums_dict.get(target - nums[i]), i]
                break
        return ret

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """#15 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
        注意：答案中不可以包含重复的三元组。
        思路: 从两数之和得到启发，把-c看成是target，遍历排序之后的数组，每次弹出一个c，再遍历剩余的数组找到所有符合条件的，复杂度为O(n2)"""
        if len(nums) < 3:
            return []
        ret = []
        nums.sort()
        tmp = 0.1
        while len(nums) >= 3:
            target = -nums.pop()

            # 如果和上一个target一样则直接跳过
            if target == tmp:
                continue

            # 由于target从最大值开始弹出，取负值只要其大于0则结束循环
            if target > 0:
                break

            # 保存target
            tmp = target
            nums_set = set()

            # 用以记录上一次添加到ret时的位置，防止重复添加
            last_append = -1

            for i in range(len(nums)):
                if last_append > 0 and nums[i] == nums[last_append]:
                    continue
                if target - nums[i] not in nums_set:
                    nums_set.add(nums[i])
                else:
                    ret.append([target - nums[i], nums[i], -target])
                    last_append = i
        return ret

    def findWords(self, board, words):
        """
        #212 给定一个二维网格 board 和一个字典中的单词列表 words，找出所有同时在二维网格和字典中出现的单词。
        单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        if words is None or board is None:
            return []

        step = len(board[0])
        board_list = []
        board_set = {}
        ret = []
        for item in board:
            board_list += item

        # 将board存入字典，减小复杂度
        for i in range(len(board_list)):
            board_set.setdefault(i, board_list[i])

        for word in words:
            # 首先判断一些特殊情况
            if (not word) or (word in ret) or (len(word) >= step + len(board)):
                continue
            stack = [[]]  # 用以保存每个字母在bord_set中匹配的位置
            all_exist = True

            for i in range(len(word)):
                if word[i] in board_set.values():
                    continue
                else:
                    all_exist = False
                    break

            # 任意一个字母找不到时直接跳过这个单词
            if not all_exist:
                continue

            # 长度为1时由于已经匹配到了故一定能找到
            if len(word) == 1:
                ret.append(word)
                continue

            # 先从第一个字母开始
            for i in board_set:
                if board_set[i] == word[0]:
                    stack[0].append(i)

            # st表示word下标
            st = 0
            # 首先向前搜索
            forward = True

            while st >= 0:

                # 判断当前字母在board对应的位置上还有没有没搜索过的，没有则把当前字母之后的搜索记录清空，并回溯到之前字母
                if not stack[st]:
                    for i in range(st + 1, len(stack)):
                        stack[i] = []
                    st -= 1
                    continue

                cur = stack[st].pop()
                location = []

                # 找相邻位置元素
                if forward:
                    if (cur + 1) % step != 0:
                        location.append(cur + 1)
                    if (cur + step) < len(board_set):
                        location.append(cur + step)
                else:
                    if cur % step != 0:
                        location.append(cur - 1)

                    if cur - step >= 0:
                        location.append(cur - step)

                if st + 1 < len(word):
                    if word[st + 1] not in [board_set[i] for i in location]:

                        # 没有匹配到则切换搜索模式或继续以当前字母下一个匹配位置开始搜索，只有第一个字母能选择向前还是向后搜索
                        if st == 0 and forward:
                            forward = False
                            stack[0].append(cur)
                        elif st == 0 and not forward:
                            forward = True
                        continue

                    # 匹配到了则切换到下一个字母，
                    st += 1
                    # 并且将匹配到的位置压入对应字母的栈中
                    if st >= len(stack):
                        stack.append([])
                    stack[st] += location

                else:
                    # 代表成功匹配到最后
                    ret.append(word)
                    break
        ret.sort()
        return ret

    def longestMountain(self, A: List[int]) -> int:
        """
        #845 给出一个整数数组 A，返回最长 “山脉” 的长度。
        :param A:
        :return:
        """
        if len(A) < 3:
            return 0
        ret = 0

        # 从第一个数开始判断
        stack = [0]

        # 能找到的最长山脉，当只有01012之类的出现时可以尽早结束循环
        max_num = 2 * (max(A) - min(A)) + 1

        # 开始循环查找
        while stack:
            cur = stack.pop()

            # 当已经找到最长山脉时直接结束循环
            if ret >= len(A) - cur or ret >= max_num:
                break

            # 山脉左右两边元素个数
            l_tmp, r_tmp = 0, 0

            # 从cur位置开始查找
            while cur < len(A) - 1:
                if A[cur] < A[cur + 1] and r_tmp == 0:
                    l_tmp += 1
                    cur += 1
                elif A[cur] > A[cur + 1] and l_tmp > 0:
                    r_tmp += 1
                    cur += 1
                else:
                    break

            # 查找完成后判断是不是找到更长的
            if r_tmp > 0 and l_tmp + r_tmp >= ret:
                ret = l_tmp + r_tmp + 1

            # 当没找到时应该从下一个位置继续寻找
            if r_tmp == 0:
                cur += 1

            # 这里主要是把下一个查找位置压入栈中，注意当找到山脉时只需把当前没构成山脉的位置添加过去即可
            if cur < len(A):
                stack.append(cur)
        return ret

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        """
        #617 合并两个二叉树
        给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
        你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
        否则不为 NULL 的节点将直接作为新二叉树的节点。
        """
        if not t1:
            return t2
        if not t2:
            return t1
        # t3 = TreeNode(t1.val+t2.val)
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """#14 最长公共前缀"""
        if not strs:
            return ''
        if len(strs) == 1:
            return strs[0]
        # strs = set(strs)
        for i in range(len(strs[0])):
            for s in strs[1:]:
                if strs[0][:i + 1] == s[:i + 1]:
                    continue
                return s[:i]
        return strs[0]

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        """#448 给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
            找到所有在 [1, n] 范围之间没有出现在数组中的数字。"""
        # 利用集合去重
        return list({i for i in range(1, len(nums) + 1)} - set(nums))

        # 由热评想到的另一种方法，对于数组中出现过的元素作为下标对数组作减法，
        # 保证出现过的元素都为负，那么为正的自然就是没出现过的
        # length = len(nums)
        # for i in nums:
        #     if nums[i-1]>0:
        #         nums[i-1] -= length
        # return [index+1 for index, i in enumerate(nums) if i > 0]

    def removeDuplicates(self, nums: List[int], n=2) -> int:
        """#80 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现n次，返回移除后数组的新长度。
            不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。"""
        # 方法1简单粗暴， 只要当前元素和它之前2个元素相等，就认为元素是重复了2次以上，则不交换
        i = 0
        for num in nums:
            if i < n or num != nums[i - n]:
                nums[i] = num
                i += 1
        return i

        # 方法2，用两个指针i，k分别记录交换位置和当前作比较的元素，times记录元素出现次数，遍历一遍数组，把符合条件的元素交换到前面
        # if not nums:
        #     return 0
        # i, k = 1, 0
        # times = 1
        # for j in range(1, len(nums)):
        #     if nums[k] == nums[j]:
        #         times += 1
        #         if j + 1 == len(nums) and i < j and times == 2:           # 如果已经到了最后一个元素且为两个相等的元素则交换
        #             nums[i] = nums[j]
        #         if times <= 2:
        #             if i < j:
        #                 nums[i] = nums[j]
        #             i += 1
        #     else:
        #         k = j
        #         times = 1
        #         if i < j:
        #             nums[i] = nums[j]
        #         i += 1
        # return i

    def searchMatrix(self, matrix, target):
        """
        #240 搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
        1.每行的元素从左到右升序排列。
        2.每列的元素从上到下升序排列。
        思路：每次查询中间位置，比较之后可以较小约1/4，直到矩阵元素内容为空
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        column_mid = (len(matrix[0]) - 1) // 2
        row_mid = (len(matrix) - 1) // 2
        if target > matrix[row_mid][column_mid]:
            return self.searchMatrix(matrix[row_mid + 1:], target) or self.searchMatrix(
                [i[column_mid + 1:] for i in matrix[:row_mid + 1]], target)
        elif target < matrix[row_mid][column_mid]:
            return self.searchMatrix(matrix[:row_mid], target) or self.searchMatrix(
                [i[:column_mid] for i in matrix[row_mid:]], target)
        else:
            return True

    def nthUglyNumber(self, n: int) -> int:
        """
        #313 找出第 n 个丑数。
        丑数就是只包含质因数 2, 3, 5 的正整数。1也为丑数"""
        # 定义3个指针，分别指代当前位置乘 2，3，5操作
        # 每个位置都应该乘一遍2，3，5
        # 每次取最小的加到列表末尾
        ret = [1]
        i1, i2, i3 = 0, 0, 0
        for i in range(n - 1):
            ret.append(min(ret[i1] * 2, ret[i2] * 3, ret[i3] * 5))
            if ret[-1] == ret[i1] * 2:
                i1 += 1
            if ret[-1] == ret[i2] * 3:
                i2 += 1
            if ret[-1] == ret[i3] * 5:
                i3 += 1
        return ret[-1]

    def swimInWater(self, grid: List[List[int]]) -> int:
        """#778 在一个 N x N 的坐标方格 grid 中，每一个方格的值 grid[i][j] 表示在位置 (i,j) 的平台高度。
        现在开始下雨了。当时间为 t 时，此时雨水导致水池中任意位置的水位为 t 。你可以从一个平台游向四周相邻的任意一个平台，但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。当然，在你游泳的时候你必须待在坐标方格里面。
        你从坐标方格的左上平台 (0，0) 出发。最少耗时多久你才能到达坐标方格的右下平台 (N-1, N-1)？
        思路：t逐渐增大，直到能走到末尾，才返回这个t；判断能否走到末尾时每次把周围小于等于t的位置压入栈中，直到栈中的元素在末尾则表示走到尾了"""
        N = len(grid)

        # 第一次t直接从首尾元素的最大值开始，因为这两个元素必须要经过
        t = max(grid[0][0], grid[N - 1][N - 1])

        while True:
            # 每次从顶点开始
            stack = [[0, 0]]
            # 定义一个集合存放走过的元素防止重复
            walk = set()

            while stack:
                i, j = stack.pop()

                # 走到末尾直接返回t
                if i == j == N - 1:
                    return t

                walk.add((i, j))

                # 判断4个相邻元素
                if j + 1 < N and grid[i][j + 1] <= t and (i, j + 1) not in walk:
                    stack.append([i, j + 1])
                if i + 1 < N and grid[i + 1][j] <= t and (i + 1, j) not in walk:
                    stack.append([i + 1, j])
                if j - 1 >= 0 and grid[i][j - 1] <= t and (i, j - 1) not in walk:
                    stack.append([i, j - 1])
                if i - 1 >= 0 and grid[i - 1][j] <= t and (i - 1, j) not in walk:
                    stack.append([i - 1, j])
            t += 1

    def canCross(self, stones: List[int]) -> bool:
        """403. 青蛙过河
        思路：保存相邻石子间的距离为一个列表，方便判断
                再维护一个每个位置跳跃步长列表的字典"""
        if stones[1] > 1:
            return False
        # 相邻石子间隔的距离
        steps = [stones[i] - stones[i - 1] for i in range(1, len(stones))]
        # 如果全是1、2，直接返回true
        if max(steps) < 3:
            return True
        # 定义一个字典维护每个位置跳跃过的步长的列表
        jump = {0: [1]}
        # i为当前位置
        i = 0
        while i < len(steps):
            # 指向下一个位置
            i += 1
            if not jump.get(i - 1):
                # 如果当前位置没有发生跳跃则表示是跳过的石头直接下个位置
                continue
            # 弹出当前位置步长列表，也可以直接get，每次pop出来应该能减小空间占用
            step = jump.pop(i - 1)
            # 存放当前位置添加过的步长，防止重复添加
            used_step = set()
            while step:
                # 如果i到了末尾则表示成功过河
                if i == len(steps):
                    return True
                # 每次从步长列表弹出一个步长
                cur_step = step.pop()
                if cur_step in used_step:
                    continue
                # 步长加入集合防止重复
                used_step.add(cur_step)
                # 当前位置能跳哪些步长
                able_step = [cur_step - 1, cur_step, cur_step + 1]
                for step_ in able_step:
                    start = 0
                    for j in range(i, len(steps)):
                        # 累加跳的步长
                        start += steps[j]
                        # 有匹配的直接添加到字典里匹配位置上的步长列表里
                        if start == step_:
                            jump.setdefault(j, []).append(step_)
                        # 一旦超出则直接退出循环进入下一个步长匹配
                        if start > step_:
                            break
        # 没有新的位置匹配则表示没能过河
        return False

    def pancakeSort(self, A: List[int]) -> List[int]:
        """969. 煎饼排序
        思路：每次找到最大值翻转换到最前面再整个翻转把最大值放到最后，如此循环"""
        ret = []
        while A:
            max_one = max(A)
            for i, v in enumerate(A):
                if v == max_one:
                    break
            # 如果最大值已经在最后面了则不翻转
            if i == len(A) - 1:
                A.pop()
                continue
            elif i != 0:
                # 如果最大值不在最前面则要进行这次翻转
                ret.append(i + 1)
                for j in range((i+1) // 2):
                    A[j], A[i - j] = A[i - j], A[j]
            ret.append(len(A))
            A.reverse()
            A.pop()
        return ret

    def insertionSortList(self, head: ListNode) -> ListNode:
        """147. 对链表进行插入排序"""
        if not head:
            return head

        sort_list = [head]
        node = head.next
        while node:
            if sort_list[-1].val > node.val:
                for i in range(len(sort_list)):
                    if node.val < sort_list[i].val:
                        sort_list.insert(i, node)
                        break
            else:
                sort_list.append(node)
            node = node.next
        head = sort_list[0]

        # 交换节点时要注意把最后一个节点的next置为空，否则节点可能成为局部循环链表
        sort_list[-1].next = None

        for i in range(len(sort_list) - 1):
            sort_list[i].next = sort_list[i + 1]
        return head

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """452用最少数量的箭引爆气球
        思路：贪婪算法，先根据所有点的左值进行从大到小排序，然后每次从最后一个开始，每射一箭，尽量多引爆些气球，
        直到没有满足条件的则箭数量加一，进行下一轮循环"""
        if not points or len(points[0]) == 0:
            return 0
        points.sort(key=lambda x: -x[0])
        ret = 0
        while points:
            point = points.pop()
            ret += 1
            while points:
                if points[-1][0] <= point[1]:
                    # 每找到一个气球在区间内则更新点的左右值
                    point = [max(point[0], points[-1][0]), min(point[1], points[-1][1])]
                    # 并删除这个点表示引爆了
                    points.pop()
                else:
                    # 没找到则退出当前循环
                    break
        return ret

    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        """
        #218 天际线问题
        思路：按顺序扫描建筑点坐标，并以左端点进队右端点出队维持一个建筑对整个天际线的影响，每次新增的均为建筑轮廓顶点坐标
        :param buildings:
        :return:
        """
        import heapq

        if not buildings:
            return []
        points = []
        heap = []
        ret = []
        max_high = - float('inf')

        # 将建筑以坐标（x，h）的形式存起来并按x排序，其中为了区分左端点与右端点高度为正负值
        for build in buildings:
            points.append([build[0], build[2]])
            points.append([build[1], - build[2]])
        points.sort(key=lambda x: x[0])

        # 遍历点坐标
        for i, point in enumerate(points):
            # 为左端点时
            if point[1] > 0:

                # 构建堆，每次能获得负最小值，即最大值
                heapq.heappush(heap, -point[1])

                if point[1] > max_high:
                    max_high = point[1]

                    # 与前一个保存的天际线左端点相同时更新高度值
                    if i > 0 and ret[-1][0] == point[0] and points[i - 1][1] > 0:
                        ret[-1][-1] = max_high

                    # 否则当高度不同时新增天际线点
                    elif not ret or ret[-1][-1] != max_high:
                        ret.append([point[0], max_high])

            # 为右端点时
            else:
                # 取出该点
                heap.remove(point[1])
                # 重新调整堆
                heapq.heapify(heap)

                # 高度较小则直接跳过
                if -point[1] < max_high:
                    continue

                # 判断堆是否为空
                if heap:
                    max_high = -heap[0]
                else:
                    max_high = 0
                # 当与下一个坐标不重合并且高度不相等时新增
                if i == len(points) - 1 or (points[i + 1][0] != point[0] and -point[1] != max_high):
                    ret.append([point[0], max_high])
        return ret

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        #120 三角形最短路径之和
        动态规划
        :param triangle: 三角形数组
        :return: 最短路径
        """
        rows = len(triangle)
        if rows == 0:
            return 0

        '''
          方法2：自底向上
          需开辟O(rows)空间，动态方程：dp[j] = min(dp[j+1], dp[j]) + triangle[i][j];i为层数，j为该层第几个元素
        '''

        # 需开辟空间
        # dp = [i for i in triangle[-1]]

        # 原地修改
        dp = triangle[-1]

        # 自底向上
        for i in range(rows - 2, -1, -1):
            for j in range(i + 1):
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
        return dp[0]

        #
        # # 方法1：需开辟O(n2)空间，自顶向下
        # # dp[i][j] 表示到从上到下走到i,j位置最小路径的值.
        # # 动态方程: dp[i][j] = min(dp[i-1][j], dp[i-1][j+1]) + triangle[i][j]
        #
        # dp = [[0] * col for col in range(1, rows+1)]
        # dp[0][0] = triangle[0][0]
        #
        # for i in range(1, rows):
        #     dp[i][0] = dp[i-1][0] + triangle[i][0]
        #
        #     for j in range(1, i):
        #         dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j]
        #
        #     dp[i][i] = dp[i-1][i-1] + triangle[i][i]
        # return min(dp[-1])
        #

    def convert(self, s: str, numRows: int) -> str:
        """
        #6 Z字形变换
        开辟numRows个字符串分别保存每一行的字符，最后再后并
        :param s:
        :param numRows:
        :return:
        """
        if numRows <= 1:
            return s
        ret_list = ['' for _ in range(numRows)]

        # Z字形移动方向
        up = False
        # 当前移动到哪个字符串
        cur = 0
        for i in range(len(s)):
            ret_list[cur] += s[i]
            if not up:
                cur += 1
                if cur == numRows - 1:
                    up = True
            else:
                cur -= 1
                if cur == 0:
                    up = False
        return ''.join(ret_list)

    def shortestPalindrome(self, s: str) -> str:
        """
        #214 最短回文串
        :param s:
        :return:
        """
        # 方法1：先把原字符串翻转，再拼接直至成回文串，时间复杂度O(n2)
        # s_reve = s[::-1]
        # ret = ''
        # for i in range(len(s)):
        #     ret = s_reve[:i] + s
        #     if ret == ret[::-1]:
        #         break
        # return ret

        # 方法2：先找到从头开始的最长回文子串，再把剩下的翻转拼接到原串,时间复杂度O(n2)
        # size = len(s)
        # if size < 2:
        #     return s
        # for i in range(size, 0, -1):
        #     if s[:i] == s[:i][::-1]:
        #         break
        # return s[i:][::-1] + s

        # 方法3 利用KMP算法中的next数组， 时间复杂度O（n）
        p = s + '#' + s[::-1]
        max_len = KMP(p).next[-1] + 1
        return p[-len(s):-max_len] + s

    def longestPalindrome(self, s: str) -> str:
        """
        #5 最长回文子串 动态规划
        :param s:
        :return:
        """
        size = len(s)
        if size < 2:
            return s

        # 二维 dp 问题
        # 状态：dp[l,r]: s[l:r] 包括 l，r ，表示的字符串是不是回文串
        dp = [[False for _ in range(size)] for _ in range(size)]

        longest_l = 1
        res = s[0]

        # 因为只有 1 个字符的情况在最开始做了判断
        # 左边界一定要比右边界小，因此右边界从 1 开始
        for r in range(1, size):
            for l in range(r):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 在头尾字符相等的前提下，如果收缩以后不构成区间（最多只有 1 个元素），直接返回 True 即可
                # 否则要继续看收缩以后的区间的回文性
                # 重点理解 or 的短路性质在这里的作用
                if s[l] == s[r] and (r - l <= 2 or dp[l + 1][r - 1]):
                    dp[l][r] = True
                    cur_len = r - l + 1
                    if cur_len > longest_l:
                        longest_l = cur_len
                        res = s[l:r + 1]
        return res

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        #2 两数相加
        :param l1:
        :param l2:
        :return:
        """

        ret = ListNode(0)
        tmp = ret

        while True:
            # 3个相加，tmp.val可以理解为进位
            val = l1.val + l2.val + tmp.val
            if val >= 10:
                val -= 10
                # 向前进位
                tmp.next = ListNode(1)

            tmp.val = val
            l1 = l1.next
            l2 = l2.next

            if not (l1 or l2):
                # 两个加数都算完了直接返回结果
                return ret

            # 否者加数补0
            if not l1:
                l1 = ListNode(0)
            elif not l2:
                l2 = ListNode(0)
            if not tmp.next:
                tmp.next = ListNode(0)
            tmp = tmp.next

    def trap(self, height: List[int]) -> int:
        """
        #42 接雨水 方法多多 各显神通
        :param height:
        :return:
        """
        if not height:
            return 0
        water = 0
        length = len(height)
        i = 0
        while i + 1 < length:
            val = height[i]
            if val == 0:
                i += 1
                continue
            tmp = []
            for j in range(i + 1, length):
                # 遍历之后的柱高，如果大于之前的高度，则从此刻开始继续寻找
                if height[j] >= val:
                    i = j
                    break
                # 保存寻找过的位置
                tmp.append(j)
                # 到尾了还没找到更高的
                if j + 1 == length:
                    # 找到最大值的及其位置
                    max_one = max([height[i] for i in tmp])
                    for k in tmp[::-1]:
                        if height[k] == max_one:
                            i = k
                            break
                        tmp.pop()
                    val = max_one
            # 此时最多能容纳这么多水
            water += val * len(tmp)
            # 再减去其中方块占的体积
            for j in tmp:
                water -= height[j]

        return water

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        #452 用最少数量的箭引爆气球 贪心算法 一次尽可能引爆多气球
        :param points:
        :return:
        """
        if not points or len(points[0]) == 0:
            return 0
        points.sort(key=lambda x: -x[0])
        ret = 0
        while points:
            point = points.pop()
            ret += 1
            while points:
                if points[-1][0] <= point[1]:
                    # 每找到一个气球在区间内则更新点的左右值
                    point = [max(point[0], points[-1][0]), min(point[1], points[-1][1])]
                    # 并删除这个点表示引爆了
                    points.pop()
                else:
                    # 没找到则退出当前循环
                    break
        return ret

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        # 37 解数独    回溯
        """
        from collections import defaultdict

        def put_able(i, j, num):
            # 判断在（i，j）位置能不能放num

            # 旁边3*3个方块的位置
            box_i = i // 3 * 3 + j // 3

            # 没有重复的则放置并返回true
            if rows_dict[i][num] == 0 and cols_dict[j][num] == 0 and box_dict[box_i][num] == 0:
                # 把每一个区域该num个数加1
                rows_dict[i][num] += 1
                cols_dict[j][num] += 1
                box_dict[box_i][num] += 1
                board[i][j] = str(num)
                return True

            # 否则无法放置返回false
            board[i][j] = '.'
            return False

        def back_do():
            # 回溯
            # print(stack)

            # 弹出上一个确定好的位置并将该位置所在区域对应数值个数减1
            i, j, num = stack.pop()
            box_i = i // 3 * 3 + j // 3
            rows_dict[i][num] -= 1
            cols_dict[j][num] -= 1
            box_dict[box_i][num] -= 1
            board[i][j] = '.'

            # 继续迭代至9
            for n in range(num + 1, 10):
                if put_able(i, j, n):
                    stack.append((i, j, n))
                    return i, j

            # 每找到则继续递归
            return back_do()

        stack = []
        row = col = 0

        # 将各个区域对应数字个数维护成一个字典，用列表存储
        rows_dict = [defaultdict(int) for _ in range(9)]
        cols_dict = [defaultdict(int) for _ in range(9)]
        box_dict = [defaultdict(int) for _ in range(9)]

        # 初始化
        for r in range(9):
            for c in range(9):
                if board[r][c] != '.':
                    d = int(board[r][c])
                    rows_dict[r][d] += 1
                    cols_dict[c][d] += 1
                    box_dict[r//3*3+c//3][d] += 1

        # 循环直到最后一个
        while True:
            # print(board)
            if board[row][col] == '.':
                stack.append((row, col, 0))
                row, col = back_do()
            if row == col == 8:
                break
            if col == 8:
                col = 0
                row += 1
            else:
                col += 1

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        找中位数
        :param nums1:
        :param nums2:
        :return:
        """
        m, n = len(nums1), len(nums2)
        if m > n:
            return self.findMedianSortedArrays(nums2, nums1)

        lo, ro = 0, m
        if not m:
            return nums2[n // 2] if n % 2 else (nums2[n // 2 - 1] + nums2[n // 2]) / 2

        while lo < ro:
            c1 = (ro + lo) / 2
            c2 = (m + n) / 2 - c1

            if c2 != int(c2):
                c2 = int(c2)
                l_max2, r_min2 = nums2[c2], nums2[c2]
            else:
                c2 = int(c2)
                if c2 == n:
                    l_max2 = nums2[c2 - 1]
                    r_min2 = float('inf')
                elif c2 == 0:
                    l_max2 = float('-inf')
                    r_min2 = nums2[c2]
                else:
                    l_max2, r_min2 = nums2[c2 - 1], nums2[c2]

            if c1 != int(c1):
                c1 = int(c1)
                l_max1, r_min1 = nums1[c1], nums1[c1]
            else:
                c1 = int(c1)
                if c1 == m:
                    l_max1, r_min1 = nums1[c1 - 1], float('inf')
                elif c1 == 0:
                    r_min1, l_max1 = nums1[c1], float('-inf')
                else:
                    l_max1, r_min1 = nums1[c1 - 1], nums1[c1]

            if l_max1 <= r_min2 and l_max2 <= r_min1:
                return (max(l_max1, l_max2) + min(r_min1, r_min2)) / 2

            if l_max1 > r_min2:
                ro = c1 - 1
            elif l_max2 > r_min1:
                lo = c1 + 1
            print()
        return (max(l_max1, l_max2) + min(r_min1, r_min2)) / 2

    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        #322 零钱兑换 动态规划dp[a] = min(dp[a-ci]) + 1,
        dp[a]为组成a所需要的最少硬币个数，会等于组成a-其中一个硬币所需要的最少硬币+1
        """

        dp = [float('inf') for _ in range(amount + 1)]
        dp[0] = 0

        for i in coins:
            if i <= amount:
                dp[i] = 1

        for i in range(min(coins) + 1, amount + 1):
            dp[i] = min([dp[i - c] for c in coins if i - c >= 0]) + 1

        return dp[-1]

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        """
        #983 最低票价 动态规划
        # dp[i]表示从第i天到第i天所花费的最小钱
        #               前一天买了一天的票   前7天买了7天的票    前30天买了30天的票
        # dp[i] = min([dp[i-1]+costs[0],   dp[i-7]+costs[1],   dp[i-30]+costs[2] ] )
        """
        n = days[-1]
        dp = [0 for i in range(n + 1)]

        for i in days:
            dp[i] = -1

        for i in range(1, n + 1):
            if dp[i] != -1:
                dp[i] = dp[i - 1]
            else:
                cost7 = costs[1] if i <= 7 else dp[i - 7] + costs[1]
                cost30 = costs[2] if i <= 30 else dp[i - 30] + costs[2]
                dp[i] = min(dp[i - 1] + costs[0], cost7, cost30)
        return dp[-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        #64 最小路径和 动态规划
        两种策略：
        1、构建一个和grid同样大小dp，dp[i][j]表示到i，j位置所需要的最小路径，动态方程：
                    dpi][j] = grid[i][j] + min(dp[i-1][j],dp[i][j-1])，考虑到每次计算dp[i][j]时都只用到grid[i][j],可以直接把grid当dp用
        2、仔细想这个计算过程，其实每一个dp[i][j]都只用到同一行前一列和同一列前一行的dp值，故可以构建一个长度为n的数组，每循环一行更新到达每一列所需最短路径，即更新一遍dp值，当循环下一行时
            此时这一行每一列可以由上一行对应列上和当前行前一列最小值获得，而此时前一列已经更新了，dp也可以dp=grid[0]直接初始化
        两种策略都要注意下边界情况，第二种其实也可以列循环套行，此时dp长度应该为m用来保存到达每一列时对应的最小路径，复杂度均为O(mn),空间复杂度由于修改gird都可以降至O(1)
        至于从头开始还是从尾开始，肯定都可以的，因为两个点肯定都是要经过的，从头到尾和从尾到头最短路径肯定是相同的
        :param grid:
        :return:
        """
        m = len(grid)
        if not m:
            return 0
        n = len(grid[0])
        if not n:
            return 0

        # ######## 策略2 ##########
        # dp = grid[0]
        # for i in range(m):
        #     for j in range(n):
        #         if i == j == 0:
        #             continue
        #         if j and i:
        #             dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
        #         elif j:
        #             dp[j] = grid[i][j] + dp[j - 1]
        #         else:
        #             dp[j] = grid[i][j] + dp[j]
        #######################################

        # #######策略1########
        dp = grid
        for i in range(m):
            for j in range(n):
                if i == j == 0:
                    continue
                if j and i:
                    dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
                elif j:
                    dp[i][j] = grid[i][j] + dp[i][j-1]
                elif i:
                    dp[i][j] = grid[i][j] + dp[i-1][j]

        return dp[m-1][n-1]

    def numSquarefulPerms(self, A: List[int]) -> int:
        """
        # 996 正方形数组的数目，重新排列数组使其任意相邻位置上的和为完全平方数，只要有一个位置不同即视为排列方式不同，输出排列方式数目
        动态规划 + dfs
        先简单理一下，首先要判断任意两个元素相加是否为完全平方数，以此为依据来判断是否能作为相邻元素，这里可以将两元素满足条件则视为一条边
        那就转化成不重复任一边将所有节点连接起来，注意会存在相同元素，可以在最后作处理：
        对于一个顶点来说，有n个相同元素和它构成一条边则只需将结果除于n！，因为这三条边根据顺序不同会被重复计算n！次
        :param A:
        :return:
        """

        import math
        from collections import Counter
        from functools import lru_cache

        def is_sqrt(a):
            b = math.sqrt(a)
            return True if b == int(b) else False

        n = len(A)

        # n个顶点根据是否访问过可以用n位二进制表示，类试TSP
        count = 1 << n

        # 保存计算过的边，这很重要，可以大大降低算法复杂度
        dp = [[-1] * count for _ in range(n)]

        # edge[i]表示能和i顶点构成边的顶点列表
        edge = [[] for _ in range(n)]

        # 初始化edge
        for i in range(n - 1):
            for j in range(i + 1, n):
                if is_sqrt(A[i] + A[j]):
                    edge[i].append(j)
                    edge[j].append(i)
            if not edge[i]:
                return 0

        # @lru_cache(None)   # 装饰器，可以实现类试dp的功能，保存计算过的值
        def dfs(i, search):
            """
            # DFS实现
            :param i: 当前顶点
            :param search: 搜索过的顶点，二进制表示
            :return: ans路径
            """

            # 减少递归次数，遇到计算过的值直接返回
            if dp[i][search] > -1:
                return dp[i][search]

            # 搜索完了，返回1
            if search == count - 1:
                return 1

            ans = 0

            for j in edge[i]:
                if (search >> j) & 1:
                    # j在search里边，即搜索过了
                    continue
                ans += dfs(j, (1 << j) | search)

            # 保存计算值
            dp[i][search] = ans

            return ans

        # 将所有顶点作为起始点搜索一遍结果累加
        ans = sum(dfs(i, 1 << i)for i in range(n))

        # 排除重复边
        for v in Counter(A).values():
            ans //= math.factorial(v)
        return ans


def remove_duplicates(nums: List[int]) -> int:
    """找出无序序列中不同元素个数"""
    count = len(nums)
    step = count - 2
    while step > 0:
        i = 0
        for j in range(count):
            if i >= step and nums[i-step] != nums[j]:
                nums[i] = nums[j]
                i += 1
            elif i < step:
                i += 1
        step -= 1
        if count > i:
            count = i
        # print(nums[:count])
    return count


def remove_duplicates2(nums: List[int]) -> int:
    """先排序再查找"""
    nums.sort()
    s = Solution()
    return s.removeDuplicates(nums, 1)


def remove_duplicates3(nums: List[int]) -> int:
    """利用集合去重"""
    return len(set(nums))


def complex_test(range_num=100, count_num=1000, funcs=None):
    """
    多次运行测试
    :param range_num:
    :param count_num:
    :param funcs:
    :return:
    """
    if not funcs:
        return
    array = [randint(0, range_num) for _ in range(count_num)]
    for func in funcs:
        start = time.time()
        print('{:<25}{:^10d}{:^12d}{:>10}'.format(func.__name__, count_num, func(array), time.time()-start))


if __name__ == '__main__':
    s = Solution()
    print(s.numSquarefulPerms([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
