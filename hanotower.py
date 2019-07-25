class HanoiMove:
    """
    汉诺塔问题
    把n块铁饼从杆1挪到杆2上，先把n-1块从1挪到3，然后把最后一块从1挪到2，再把n-1块从3挪到2，于是把原来n块铁饼减小成n-1块，
    最后化简成挪1块的问题。
    时间复杂度为2^n
    """
    def __init__(self, nums=5):
        """
        :param nums: 铁饼数量
        """
        if not isinstance(nums, int) or nums <= 0:
            raise RuntimeError('invalid nums')
        self.nums = nums
        # 存放每一步步骤，假设铁饼从上到下编号为1.。。。
        self.steps = []
        # 总共3根杆
        self.staff = {1, 2, 3}
        # 移动铁饼
        self.move(1, 2, nums)

    def move(self, mov_from, mov_to, mov_nums, top=1):
        """
        把铁饼从一根杆移动到另一杆
        :param mov_from: 移动前位置
        :param mov_to:  移动后位置
        :param mov_nums:    铁饼数量
        :param top: 最上边铁饼编号
        :return: None
        """
        if mov_nums == 1:
            # 移动1块铁饼时直接打印步骤
            self.steps.append((top, mov_from, mov_to))
            return

        # 找到中转杆，即第三根杆
        at = (self.staff - {mov_from, mov_to}).pop()

        # 移动n-1块到中转杆
        if mov_nums > 1:
            self.move(mov_from, at, mov_nums-1)

        # 移动最后一块到目标杆
        self.move(mov_from, mov_to, 1, mov_nums)

        # 把n-1块从中转杆移到目标杆
        if mov_nums > 1:
            self.move(at, mov_to, mov_nums-1)

    def print_step(self):
        """
        打印移动步骤
        :return:
        """
        print('total %d steps!' % len(self.steps))
        for i, step in enumerate(self.steps):
            print('%d:moving ring %d from %d to %d' % (i+1, *step))


if __name__ == '__main__':
    hm = HanoiMove(8)
    hm.print_step()
