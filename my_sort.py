# python3排序算法实现


def select_sort(array):
    """
    选择排序算法  遍历数组每次把最小值交换到前面
    :param array:
    :return:None
    """
    for index in range(len(array)-1):
        min_index = index
        for i in range(index+1, len(array)):
            if array[i] < array[min_index]:
                min_index = i
        if index != min_index:
            array[index], array[min_index] = array[min_index], array[index]
    # return None


def bubble_sort(array):
    """
    冒泡排序算法  类似冒泡现象 每次把较大值往后移一位
    :param array:
    :return:None
    """
    for i in range(len(array)-1):
        for j in range(len(array)-1-i):
            if array[j] > array[j+1]:
                array[j+1], array[j] = array[j], array[j+1]
    # return None


def insert(array, gap=1):
    """
    插入算法    将一个数据插入到已经排好序的有序数据中，从而得到一个新的、个数加一的有序数据，
    :param array: 序列
    :param gap: 步长
    :return:None
    """

    # 相当于原数组分成gap组
    # 对所有组进行插入排序
    # 最终gap=1时才能将数组完全排序
    for i in range(gap):
        for j in range(i + gap, len(array), gap):
            # 已排好序的数据
            pre_index = j - gap
            current = array[j]
            # 把current插入到前面已排好续的数据
            while pre_index >= 0 and current < array[pre_index]:
                array[pre_index + gap] = array[pre_index]
                pre_index -= gap
            array[pre_index + gap] = current


def insert_sort(array):
    """
    一般插入算法实现 每次只插一个
    :param array:
    :return:
    """
    insert(array)
    # return None


def shell_sort(array):
    """
    希尔插入算法 缩减增量排序
    希尔排序是按照不同步长对元素进行插入排序，当刚开始元素很无序的时候，步长最大，所以插入排序的元素个数很少，速度很快；
    当元素基本有序了，步长很小，插入排序对于有序的序列效率很高。所以，希尔排序的时间复杂度会比o(n^2)好一些。
    相当于较少了直接插入排序数据的交换次数
    :param array:
    :return:
    """
    # 步长
    gap = len(array)//2
    while gap > 0:
        insert(array, gap)
        gap //= 2
    return None


def merge(ll, rl=[]):
    """
    两个有序序列合并为一个
    时间复杂度O(n)
    :param ll:
    :param rl:
    :return:
    """
    l_point, r_point = (0, 0)
    result = []
    while l_point < len(ll) and r_point < len(rl):
        if ll[l_point] < rl[r_point]:
            result.append(ll[l_point])
            l_point += 1
        else:
            result.append(rl[r_point])
            r_point += 1
    result += ll[l_point:]
    result += rl[r_point:]
    return result


def merge_sort(array):
    """
    归并算法 分治法 将每个子序列有序后合并成整个有序序列 总共归并logn次，故时间复杂度为O(nlogn)
    :param array:
    :return:None
    """
    if len(array) < 2:
        return array

    # # 递归
    # mid = len(array) // 2
    # res1 = merge_sort(array[:mid])
    # res2 = merge_sort(array[mid:])
    # return merge(res1, res2)

    # 非递归
    st = 1
    res = array

    while st < len(array):
        for i in range(0, len(array), 2*st):
            res[i:i+2*st] = merge(res[i:i+st], res[i+st:i+2*st])
        st *= 2
    return None


def quick_sort(array):
    """
    快排递归实现     二分法      每次选一个分界点把数组分成两部分
    :param array:
    :return:array
    """
    if len(array) <= 1:
        return array

    basic = array[0]
    left = 0
    right = len(array) - 1
    while left < right:
        if array[right] < basic:
            array[left] = array[right]
            left += 1
            while left < right:
                if array[left] > basic:
                    array[right] = array[left]
                    right -= 1
                    break
                else:
                    left += 1
        else:
            right -= 1
    array[left] = basic
    return quick_sort(array[0: left]) + [basic] + quick_sort(array[left+1:])
    #     # 一句话快排
    # return quick_sort([item for item in array if item < array[0]]) + array[0:1] +
    # quick_sort([item for item in array if item>array[0]])


def quick_sort2(array):
    """
    快排非递归实现
    :param array:
    :return:None
    """

    # 先传入整个序列
    left = 0
    right = len(array) - 1
    stack = [left, right]

    # 当栈不为空说明还有子序列未比较完，以此实现类似递归操作
    while stack:
        # print(stack, array)
        # 每次获得子序列首尾端
        low = stack.pop(0)
        high = stack.pop(0)

        # 子序列为空时结束本次循环
        if high <= low:
            continue

        # 每次比较序列末尾元素
        pivot = array[high]

        # i为比较的元素的最终位置
        i = low - 1

        # 遍历子序列获得作比较的元素的位置
        for j in range(low, high + 1):
            # ##如果小于pivot， 则交换，交换的目的是保证i位置之前的元素都比pivot小或等
            if array[j] <= pivot:
                i += 1
                if i != j:
                    array[i], array[j] = array[j], array[i]
        # 获得位置i之后，继续以i为界构造子序列首尾位置
        stack.extend([low, i - 1, i + 1, high])
    # return None


def heap_adjust(array, i, end):
    """
    调整大堆
    :param array:
    :param i:初始节点位置
    :param end:需调整序列长度
    :return:None
    """
    # 保存需要调整的节点值
    tmp = array[i]
    # 获取左节点
    j = 2 * i + 1
    while j < end:
        if j+1 < end and array[j+1] > array[j]:
            # 如果存在右节点并且右节点大于左节点，则切换到右节点上
            j += 1
        if array[j] > tmp:
            # 如果 节点值比调整对象大，则把子节点赋给父节点
            array[i] = array[j]
            # 并把待调整对象切换到子节点上
            i = j
        # 继续比较下一级左节点
        j = 2 * j + 1
    # 把调整对象赋给当前定位到的节点，这样可以保证堆中i层节点以下成堆时从i层调整能还原成堆
    array[i] = tmp


def heap_sort(array):
    """
    堆排序算法  先把序列调整成大堆，然后每次把堆顶即最大值交换到末尾，再重新调整出去末尾序列成大堆
    :param array:
    :return: None
    """
    end = len(array)
    # 获取第一个非叶子节点位置
    start = end // 2 - 1

    # 将序列调整成大堆
    for i in range(start, -1, -1):
        # 从下至上，从右至左循环使序列调整成大堆
        heap_adjust(array, i, end)

    # 每次把堆顶元素即最大值交换到当前序列尾端，待调整序列长度减1
    for end in range(len(array)-1, 0, -1):
        # print(array)
        array[0], array[end] = array[end], array[0]
        heap_adjust(array, 0, end)

    return None
