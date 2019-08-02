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


class Trie:
    """
    前缀树，用于单词查找
    """
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tree = {}
        self.is_a_word = '$'

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.tree
        for char in word:
            node = node.setdefault(char, {})
        node[self.is_a_word] = self.is_a_word

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if self.tree == {}:
            return False
        search_node = self.tree
        for w in word:
            if w in search_node:
                search_node = search_node[w]
            else:
                return False
        if self.is_a_word in search_node:
            return True
        return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        if self.tree == {}:
            return False
        start_node = self.tree
        for pre in prefix:
            if pre in start_node:
                start_node = start_node[pre]
            else:
                return False
        return True


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
    # a = [
    #     ["o", "a", "a", "n"],
    #     ["e", "t", "a", "e"],
    #     ["i", "h", "k", "r"],
    #     ["i", "f", "l", "v"],
    #      ]
    # b = ["at", "oa", 'oathkrvnerv', 'vvrkatafhtao']
    # word = 'hello'
    # prefix = 'h'
    s = Solution()
    # print(s.searchMatrix([[1, 2, 3, 4, 5],
    #                       [6, 7, 8, 9, 10],
    #                       [11,12,13,14,15],
    #                       [16,17,18,19,20],
    #                       [21,22,23,24,25]],
    #                      5))
    grid = [[652,2041,1010,1052,2125,667,128,1025,983,1815,1380,1311,2297,1715,1148,2315,351,1196,29,285,1990,250,655,456,149,1285,236,582,514,2032,773,1638,826,563,613,2259,142,679,2053,2192,1643,851,2448,401,963,1467,1185,616,1497,1851],[2161,1351,1846,1008,855,1079,932,2381,1835,451,2035,1637,2283,1705,1889,548,1216,1406,1103,2401,538,346,1809,376,170,1508,1560,2197,2290,639,1609,1344,1031,2222,2130,173,2081,2047,783,1608,2414,1861,2124,791,1012,1640,410,1760,1944,121],[752,464,1562,1076,1602,2450,1662,1695,576,164,506,825,372,1000,1898,2241,1972,1896,2438,2492,2004,1840,1928,1410,1630,663,287,262,430,1226,1173,1952,2429,937,690,275,899,632,878,405,889,113,388,2107,13,891,15,653,1576,193],[1681,2013,897,143,1584,1780,1991,2265,1909,2024,832,1477,465,1125,1864,317,1448,1121,1829,2079,2261,1577,409,43,338,2046,1332,1923,584,658,1466,715,1041,2282,2343,2069,445,977,1238,1549,712,145,1205,425,2445,283,1194,206,1211,1480],[213,1558,2,1281,1425,1074,294,1039,2090,724,1999,650,1473,1603,129,168,1158,657,914,36,524,926,1392,568,1244,843,2341,1316,1874,807,1689,2387,550,1776,1779,2227,1162,2190,1612,34,1974,1813,999,1383,1495,671,1995,178,2147,1834],[2109,895,2301,1922,1054,1661,1970,169,863,39,1598,842,726,2451,1133,1215,1236,116,466,1987,205,371,1596,904,633,1210,1976,2085,1375,735,1897,1535,357,2256,302,515,1854,1384,1138,1140,2238,2029,1257,2328,2219,738,959,472,1159,435],[839,2042,1814,874,355,1169,111,1868,2435,482,108,698,546,1688,2439,1759,1155,1755,2003,1129,1575,2038,509,2368,1624,1941,137,1498,72,216,2464,1706,60,2353,771,1363,0,1468,1278,2006,172,1161,158,1739,188,549,2270,1786,950,1595],[1174,644,869,941,1545,2018,115,1674,1369,1297,292,749,1149,1414,377,1912,2419,1442,1144,1199,1268,1552,441,2415,1663,2360,5,75,226,1994,1292,490,704,2202,1563,1583,1081,2163,711,886,315,387,1789,1836,258,912,483,1707,162,2193],[461,1127,1325,2071,271,2423,365,943,532,1015,117,2059,670,877,868,1930,1870,2346,1510,1718,934,2285,2092,1504,1853,641,929,1600,1546,2141,898,1304,14,709,976,2011,1128,517,308,1290,1933,890,1934,1992,2383,1696,416,1444,880,1261],[984,2390,1386,504,2064,1625,403,2398,765,1458,2442,1501,454,459,1537,2225,1291,1918,1412,1634,566,1094,554,722,2048,1142,813,1524,2000,1812,95,896,918,1550,51,1004,247,2436,923,1722,2488,2176,1570,190,2258,2086,945,468,2204,1105],[789,1475,780,1723,802,1367,2293,1123,2062,2329,2236,1636,2300,1117,1220,1925,2262,59,993,1810,859,772,962,2044,2382,572,1690,2009,561,693,385,2087,237,2424,1432,1029,1175,1749,1505,1633,1282,449,1208,1802,894,565,1597,2039,1631,184],[2310,1832,539,2203,54,1822,1252,480,2057,1623,1604,2156,323,659,433,1439,2316,1019,1881,1591,935,2490,1632,12,2269,228,1873,1193,17,2123,786,2494,2273,230,257,1153,1773,1028,1852,1526,1366,1605,614,2272,2138,2457,1017,2207,215,710],[1481,471,992,541,862,884,1592,1801,1106,1704,1587,333,1218,1083,2371,718,240,1233,703,1515,2497,1385,478,559,1090,1582,1197,1047,44,2132,1959,686,265,814,1916,806,2148,646,2140,1428,69,270,2233,56,705,1830,1313,852,505,1528],[1393,1306,359,2496,165,1003,2286,2364,1408,20,743,1512,496,1837,78,998,112,277,1411,2373,1894,553,1269,422,846,2422,746,457,2313,838,118,1454,1985,1686,2292,1176,589,1567,696,1474,1180,1459,672,2392,2402,1658,1275,2420,2244,126],[1402,166,776,2002,1559,1772,2217,2022,1362,609,2493,2471,1554,719,1503,185,1124,183,7,685,1626,1006,1423,1965,1816,2421,22,2215,320,1349,2466,822,1714,1232,612,413,18,643,1580,1484,19,2055,721,1096,1733,1708,90,175,1790,1997],[1417,1059,1879,2367,1237,1330,364,89,298,2385,2037,2463,2404,1915,543,1551,1296,2060,269,274,757,665,1478,1659,598,1648,2433,350,1461,570,525,1091,2359,2021,1005,1645,1204,716,607,23,1069,680,1241,627,1164,500,1151,988,1319,907],[1725,782,477,1743,1993,251,2384,1235,748,1073,2102,1553,1255,266,2247,1112,2409,2456,1310,2271,2139,481,474,1871,1982,1487,1748,1082,358,1747,1299,2425,1803,2274,1303,1701,527,1676,182,2449,1954,1132,2121,2388,1716,1781,154,53,1188,1058],[1867,2131,1371,1377,2082,124,1720,2146,253,1231,1649,1968,2198,1787,856,2242,1880,2252,1183,1808,1167,2101,1669,2347,871,1945,2254,1267,1035,1566,389,594,91,1341,1673,1514,513,2459,676,1165,144,362,695,2031,552,1593,623,1262,2186,762],[526,1828,2455,2119,195,883,304,1345,529,2411,199,2417,587,2354,382,2418,264,1518,1326,761,2050,2007,1920,245,1806,558,1018,900,2028,956,1198,181,1302,913,2394,300,2288,1046,648,820,420,1684,625,2372,583,1447,1839,2495,2074,991],[2481,2370,339,1427,1250,452,1799,601,241,218,725,989,723,1877,1260,1788,1606,1800,881,1390,1719,222,1568,31,1209,580,1191,45,1858,1273,796,1359,1670,1905,378,1893,46,1246,979,1908,516,1251,1798,787,1887,11,470,2187,1731,1940],[2461,742,606,282,1581,917,244,2379,458,1077,1685,2489,821,1130,231,1141,1811,1339,1573,1462,1521,1655,1068,1450,2026,1619,1346,2218,1490,1948,978,2440,668,1774,58,1492,1118,1664,2327,411,1756,2133,9,927,1400,1967,800,808,595,1294],[986,2229,1195,2226,581,2230,593,2162,439,1556,1456,1571,1884,2255,122,412,360,418,1259,343,1348,2078,2128,691,303,1338,2289,1962,599,854,1726,379,2486,427,1635,1404,301,687,1009,1050,1134,2407,485,1202,2393,2174,207,2149,2250,1437],[1950,619,948,1978,2413,2458,759,1644,1885,2358,2208,1513,990,1712,1095,1523,1340,467,1223,1642,2122,801,1401,1413,1919,2462,2332,2447,1067,866,2304,879,1692,1728,677,100,2295,793,537,1131,1734,1263,1793,972,329,799,1342,915,1460,2108],[331,1471,694,139,155,2049,1247,864,1002,758,1906,1831,2223,2231,2020,197,1152,531,2127,1289,1479,1352,1071,2279,1166,1671,508,1113,1627,1650,830,2239,1206,1866,2075,2325,2375,2221,428,1741,2019,2251,1457,887,1399,1775,1702,1738,701,2320],[2179,353,702,1396,2416,2484,147,1883,971,1368,1727,312,1588,395,2305,1154,2105,928,392,974,853,982,299,369,1666,755,2454,750,1288,2056,1765,238,1917,2395,1665,1329,24,1960,1641,1350,1337,645,313,341,1744,1936,1264,955,1937,2118],[2070,688,2333,1051,1443,1110,2083,1119,870,1863,102,1754,651,885,865,925,841,1361,1942,2235,1758,16,911,673,191,229,1277,1961,131,424,1395,1063,1761,1710,1859,84,849,756,1651,1324,1947,634,209,163,2377,21,2120,523,153,2076],[1519,1098,460,1683,1243,1027,2476,2005,731,1407,494,528,573,1087,1135,753,174,1007,902,2363,32,324,1157,872,475,1065,1184,1365,1357,334,2352,1653,2016,605,2134,2280,141,2245,1314,2017,2389,2058,751,2336,1062,407,794,469,49,1888],[1139,2164,2068,805,1494,567,1020,322,1791,1913,2483,1055,1343,386,436,2340,1240,2326,777,503,882,1318,1848,94,1882,729,272,2470,2030,1080,1440,2308,1358,1616,1946,290,1730,1958,4,1820,2206,1409,2160,176,426,995,774,63,1872,2309],[2296,2437,1639,1986,2177,1248,321,2199,1086,6,547,1276,621,2178,1057,221,1703,2209,1293,764,1308,1421,996,2051,2479,1539,2196,1543,1272,52,2410,1522,2072,1729,631,1372,1732,1614,2183,336,1529,1844,123,2067,1172,1821,1900,908,2263,1980],[1177,327,2211,1516,1740,1860,600,2378,511,391,1449,1827,1957,326,577,1207,2474,1397,1465,2232,1186,2287,352,2312,235,1033,419,785,97,2482,366,1353,1066,2088,1469,610,1472,319,2452,68,1287,818,77,620,160,1939,924,562,700,374],[1534,1201,893,1507,493,1601,2248,987,596,1677,1334,730,1682,71,1168,2142,227,159,848,961,66,2094,67,307,1034,622,1426,2444,2171,518,130,1618,933,1675,2339,1819,1435,1347,1213,1038,404,2443,571,2144,2386,958,2100,1907,1452,967],[214,136,400,858,1,255,1485,1667,845,223,161,1266,875,2430,1499,861,828,2348,1724,1145,2426,1796,348,104,1841,484,636,2499,823,938,1219,268,256,73,2478,2480,2431,1969,1620,1271,2080,210,1104,2376,1694,2129,1951,1200,2234,2012],[30,109,1574,2061,669,502,394,969,293,1274,1391,1711,233,597,1843,2168,1589,1926,2182,2284,1470,2281,2073,1561,578,79,381,1943,2111,1795,2191,1418,947,1971,1328,217,373,2491,396,922,408,909,692,1594,2173,1687,148,2253,345,1323],[1049,328,2299,910,788,867,1735,1745,1977,438,1179,1932,1483,1963,2349,1621,1823,8,1107,741,1101,804,1768,575,295,202,203,1093,1579,2095,951,101,1307,76,501,64,2025,186,637,103,1354,1108,1382,1476,763,520,399,1493,443,1228],[1256,1300,545,1419,779,2342,1855,177,1797,1869,1548,1360,278,1509,1607,920,551,2473,1979,615,1825,1610,1856,2369,2266,1064,2432,82,2298,626,499,1381,498,267,1680,384,417,1622,2355,1927,1764,901,1817,2104,2195,942,1178,536,642,1613],[367,380,1849,2323,940,1463,1312,363,296,74,936,1910,585,107,1488,1023,2126,574,1657,1931,1709,135,1335,1891,61,1092,2014,1875,1270,361,344,476,432,965,1016,1373,1911,402,1181,114,1784,519,617,980,1678,1217,198,2427,1242,2321],[586,2240,592,2380,2366,699,1752,276,2331,1043,1045,261,2345,1783,2264,713,921,2172,2001,179,1699,140,437,1672,473,1317,949,535,2184,1182,491,1482,1234,42,325,1547,1356,770,1253,1914,674,1376,1737,939,2089,857,48,618,1697,40],[540,649,2311,1525,1061,1500,2322,1013,2165,1192,2027,2097,906,1014,2267,1301,604,1100,1818,1115,892,635,2066,309,462,440,450,1713,683,1988,87,1441,2260,1084,2008,2040,810,970,1892,2216,248,2249,1721,2135,2469,521,784,1451,204,1146],[1203,291,2155,2093,1766,591,1032,1929,180,960,732,2084,447,2106,968,28,1224,2467,1955,342,25,905,1502,260,486,654,263,953,836,38,1617,92,1496,2175,2214,778,495,744,1327,1239,707,1102,740,1679,678,1753,2335,681,660,697],[588,1422,354,1895,10,1750,2307,463,157,225,981,335,1212,569,1865,608,2276,2113,93,1975,975,2152,530,444,152,2344,2166,1364,2324,70,1983,533,1309,1431,2201,1446,62,1533,1656,1693,187,316,1122,488,2205,1379,1387,714,557,564],[1569,297,2472,252,542,356,33,797,1762,2112,2103,769,2278,2330,1902,2498,249,1030,1389,2054,2169,888,171,1429,1660,1403,1053,1085,1378,2468,728,790,2453,2405,640,944,414,1769,997,1464,2052,1430,1147,151,489,119,310,2277,507,2465],[1742,370,590,2010,1565,2275,1416,556,2143,1538,2116,1557,1586,840,2362,2034,1284,196,555,132,133,2356,1691,2159,455,2158,1953,479,873,98,876,1037,2403,1654,2477,1305,35,146,446,952,829,167,340,1056,80,2188,1315,2257,393,1717],[1981,1137,1001,127,1585,844,232,2151,397,973,429,816,390,2337,1160,1564,2220,1388,1530,349,1771,781,1899,110,1826,1989,280,1824,661,1116,375,1355,803,2294,760,2157,1850,254,954,50,243,2446,2077,1331,1230,2189,1531,831,2396,1847],[2043,809,2338,2117,2098,675,1394,689,827,1984,666,1163,2036,2213,1021,706,1629,2228,1532,2150,1370,431,1280,273,624,647,754,1136,2306,2033,2397,739,1170,534,2441,134,1807,2361,65,2302,1921,1097,1114,1438,1445,1424,330,2428,106,2374],[2399,286,2485,1541,2314,220,1520,1757,212,1805,1322,3,1320,211,2224,1258,1434,85,2015,708,2212,1022,1698,156,2145,1578,1901,1903,434,579,811,1878,1024,2136,522,1890,2460,1590,2303,194,88,766,2243,2318,246,512,81,1374,603,684],[1060,1420,1949,1171,1415,2023,96,2210,1767,1405,736,2268,1453,1187,497,239,189,305,423,2246,279,1336,817,2291,1120,1956,1599,288,1833,1075,1150,1089,2170,745,105,2065,1647,1026,1436,1143,767,1996,1286,1527,1938,150,662,1838,1777,83],[919,985,1973,1555,1221,835,1517,1070,337,442,47,1156,2412,957,1298,208,2167,1886,812,99,2185,1088,1099,487,1229,994,2357,720,1109,544,2408,120,1078,628,1646,916,1572,347,1433,775,1876,2099,383,717,1746,1668,259,57,1486,1225],[1321,2194,1862,289,2319,656,1126,1042,837,281,1536,2045,200,630,664,1511,1044,2334,1072,1254,2317,1763,2091,284,1227,2115,638,2487,2110,1214,1190,795,2237,2153,629,1785,2096,2200,86,510,727,55,2114,2365,125,224,2154,1792,306,398],[734,2406,1398,201,611,368,1751,2181,318,1736,930,847,1966,1924,1265,1111,850,138,1011,798,1540,332,2350,1782,931,1845,1544,492,242,966,1036,860,192,1279,1935,824,1249,747,768,1770,946,1842,1700,421,311,1804,1652,1489,1542,682],[2063,2180,1778,1506,1295,1964,2391,448,27,406,1333,1189,1615,314,733,833,1794,1628,2400,819,1040,2434,834,234,2137,453,2351,219,737,1455,1998,1857,2475,792,41,415,815,1222,964,26,1245,602,37,1491,1048,1611,1904,903,560,1283]]
    # [[7, 34, 16, 12, 15, 0],
    #  [10, 26, 4, 30, 1, 20],
    #  [28, 27, 33, 35, 3, 8],
    #  [29, 9, 13, 14, 11, 32],
    #  [31, 21, 23, 24, 19, 18],
    #  [22, 6, 17, 5, 2, 25]]
    # # obj = Trie()
    # # obj.insert(word)
    # # param_2 = obj.search(word)
    # # param_3 = obj.startsWith(prefix)
    # # print(Solution().findWords(a, b))
    # # print(s.longestMountain([0,1,2,0,1,0,2,0,0,2,1,2,2,1,0,0,1,0,2,2,1,0,1,2,1,0]))
    # li = [1, 2, 2, 3, 4, 5, 6, 8, 9, 8, 9, 2, 4, 10, 11, 12, 7, 8, 1, 9, 2, 0, 1, 1, 2, 2]
    # print(li[:remove_duplicates(li)])
    # print('{0:{4}<21}{1:{4}^10}{2:{4}^5}{3:{4}>10}'.format('函数名', '数据规模', '返回值', '时间', chr(32)))
    # for i in range(10, 30):
    #     complex_test(range_num=100*i, count_num=2 ** i, funcs=(remove_duplicates3, remove_duplicates3))

    # print(s.swimInWater(grid))
    # print(s.pancakeSort([93,19,91,20,82,12,18,5,57,14,37,36,32,99,100,33,22,58,83,75,49,70,60,63,15,31,88,21,35,66,89,64,69,95,50,41,52,30,56,47,1,17,77,13,26,39,53,98,81,48,8,46,45,3,55,84,51,24,42,34,25,38,96,71,27,80,85,40,28,6,59,86,65,73,29,10,94,61,2,4,7,90,43,54,87,23,97,9,62,44,68,78,72,11,74,79,67,76,92,16]))
    # node = ListNode(node_list=[3, 4, 2, 1, 9, 8, 6, 9])
    # node.show()
    # s.insertionSortList(node).show()
    # print(s.shortestPalindrome(''))

    board = [['.' for _ in range(9)] for _ in range(9)]
    s.solveSudoku(board)
    print(board)
