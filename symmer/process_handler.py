import os
import sys
import numpy as np
import quimb
from ray import remote, put, get
from multiprocessing import Process, Queue, set_start_method

if sys.platform.lower() in ['linux', 'darwin', 'linux2']:
    set_start_method('fork', force = True)
else:
    set_start_method('spawn', force = True)

class ProcessHandler:

    if sys.platform.lower() in ['linux', 'darwin', 'linux2']:
        method  = 'mp'
    else:
        method  = 'ray'

    verbose = False

    def __init__(self):
        self.n_logical_cores = os.cpu_count()

    def prepare_chunks(self, iter):
        """ split a list into smaller sized chunks
        """
        iter = list(iter)
        self.n_chunks = min(len(iter), self.n_logical_cores)
        chunk_size = int(np.ceil(len(iter)/self.n_chunks))
        indices = np.append(np.arange(self.n_chunks)*chunk_size, None)
        for i,j in zip(indices[:-1], indices[1:]):
            yield iter[i:j]

    def _process_ray(self, func, iter, shared):
        """ Helper function for ray processing
        """
        if self.verbose:
            print(f'*** executing in ray mode ***')
        # duplicate func with ray.remote wrapper :
        @remote(num_cpus=self.n_logical_cores,
            runtime_env={
                "env_vars": {
                    "NUMBA_NUM_THREADS":   os.getenv("NUMBA_NUM_THREADS"),
                    "OMP_NUM_THREADS":     os.getenv("NUMBA_NUM_THREADS"),
                    "NUMEXPR_MAX_THREADS": str(self.n_logical_cores)
                }
            }
        )
        def _func(iter, shared):
            return func(iter, shared)
        # place into shared memory:
        shared_obj = put(shared)
        # split iterable into smaller chunks and parallelize remote instances:
        results = get(
            [
                _func.remote(chunk, shared_obj) 
                for chunk in self.prepare_chunks(iter)
            ]
        )
        # flatten the list and return:
        return [a for b in results for a in b]
        
    def _process_mp(self, func, iter, shared):
        """ Helper function for multiprocessing
        """
        if self.verbose:
            print(f'*** executing in multiprocessing mode ***')
        # wrapper function for putting results into queue
        def _func(iter, shared, _order=None, _queue=None):
            data_out = func(iter, shared)
            _queue.put((_order, data_out))

        chunks = list(self.prepare_chunks(iter))
        procs = [] # for storing processes
        queue = Queue(self.n_chunks) # storage of data from processes
        for index,chunk in enumerate(chunks): # index to ensure procs returned in correct order
            proc = Process(target=_func, args=(chunk, shared, index, queue))
            procs.append(proc)
            proc.start()
        # retrieve data from the queue
        data = []
        for _ in range(self.n_chunks):
            data.append(queue.get())
        # complete the processes
        for proc in procs:
            proc.join()
        # sort by correct chunk ordering and flatten data
        _,data = zip(*sorted(data))
        data = [a for b in data for a in b]
        return data
    
    def _process_single(self, func, iter, shared):
        """ Helper function for single threading
        """
        if self.verbose:
            print(f'*** executing in single-threaded mode ***')
        return func(iter, shared)
    
    def parallelize(self, func):

        def wrapper(iter, shared):

            _func = lambda iter,shared: [func(i, shared) for i in iter]

            if self.method   == 'mp':
                return self._process_mp(_func, iter, shared)
            elif self.method == 'ray':
                return self._process_ray(_func, iter, shared)
            elif self.method == 'single_thread':
                return self._process_single(_func, iter, shared)
            else:
                raise ValueError(f'Invalid processing method {self.method}, must be ray, mp or single_thread.')
            
        return wrapper
    
process = ProcessHandler()
    
if __name__ == '__main__':
            
    @process.parallelize
    def multiply_list(iter, shared):
        return [i*shared for i in iter]
    
    l = list(range(100))

    process.method = 'single_thread'
    print(multiply_list(l,2))
    process.method = 'mp'
    print(multiply_list(l,2))
    process.method = 'ray'
    print(multiply_list(l,2))

