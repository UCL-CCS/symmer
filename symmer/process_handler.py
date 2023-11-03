import multiprocessing as mp
import ray

class ProcessHandler:

    method = 'mp'

    def __init__(self):
        self.n_logical_cores = mp.cpu_count()

    def _process_ray(self, func):
        def worker(iter, shared=None):
            return func(iter, shared)
        return worker
            
    def _process_mp(self, func):
        def worker(iter, shared=None):
            return func(iter, shared)
        return worker

    def parallelize(self, func):
        if self.method == 'mp':
            return self._process_mp(func)
        elif self.method == 'ray':
            return self._process_ray(func)
        
if __name__ == '__main__':
    PH = ProcessHandler()

    @PH.parallelize
    def add_n(l, n):
        return [i+n for i in l]
    
    print(add_n([1,2,3,4,5,6,7], 3))

