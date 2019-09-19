import threading
import time


def proper(cls):
    return object.__new__(cls)


@proper
class SingletonHun:
    """饿汉单例模式, 用之前就已经生成类实例了，利用装饰器装饰即加载特性，预先生成类实例，通过实现__call__方法使得实例成为可调用对象
    需要注意和以往不同的是Singleton直接返回的是唯一实例，而Singleton()是调用实例对象，这里执行的是初始化"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)


class SingletonLazy:
    """懒汉单例模式，第一次用才生成实例，直接用__new__方法实现
    值得注意的是这种实现会存在线程安全问题，即可能存在多个线程同时访问_obj时由于还没来得及赋值导致产生多个实例，可以加锁"""
    _obj = None
    _instance_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._obj is None:
                time.sleep(0.1)
                cls._obj = super().__new__(cls)
        return cls._obj


if __name__ == '__main__':
    a, b = SingletonHun, SingletonHun
    assert a is b, 'SingletonHun has not only a instance'

    # ------------懒汉式多线程安全测试-----------------
    instance_set = set()

    def task():
        instance_set.add(SingletonLazy())

    lazy_pool = [threading.Thread(target=task) for _ in range(100)]
    for th in lazy_pool:
        th.start()
    for th in lazy_pool:
        th.join()
    assert len(instance_set) == 1, f'thread error, have {len(instance_set)} instance'

