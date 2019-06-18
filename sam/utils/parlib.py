#!/usr/bin/python

import multiprocessing as mp
from multiprocessing import Manager
from time import sleep
import os
from joblib import Parallel, delayed


def worker_subprocess(function, devices, lockd, results, lockr,
                      pids, lockp, args, kwargs, *others):
        device = None
        while device is None:
            with lockd:
                try:
                    device = devices.pop()
                except IndexError:
                    pass
            sleep(1)
        output = function(*args, **kwargs, device=device)
        with lockd:
            devices.append(device)
        with lockr:
            results.append(output)
        with lockp:
            pids.append(os.getpid())


def parallel_run(function, *args, nruns=1, njobs=1, gpus=1, **kwargs):
    """ Mutiprocessed execution of a function with parameters, with GPU management.
    This function is useful when the used wants to execute a bootstrap on a
    function on GPU devices, as joblib does not include such feature.
    Args:
        function (function): Function to execute.
        \*args: arguments going to be fed to the function.
        nruns (int): Total number of executions of the function.
        njobs (int): Number of parallel executions (defaults to ``cdt.SETTINGS.NJOBS``).
        gpus (int): Number of GPU devices allocated to the job (defaults to ``cdt.SETTINGS.GPU``)
        \**kwargs: Keyword arguments going to be fed to the function.
    Returns:
        list: concatenated list of outputs of executions. The order of elements
        does not correspond to the initial order.
    """
    if gpus == 0 and njobs > 1:
        return Parallel(n_jobs=njobs)(delayed(function)(*args, **kwargs) for i in range(nruns))
    manager = Manager()
    devices = manager.list([f'cuda:{i%gpus}' if gpus !=0
                            else 'cpu' for i in range(njobs)])
    results = manager.list()
    pids = manager.list()
    lockd = manager.Lock()
    lockr = manager.Lock()
    lockp = manager.Lock()
    poll = [mp.Process(target=worker_subprocess,
                       args=(function, devices,
                             lockd, results, lockr,
                             pids, lockp, args,
                             kwargs))
            for i in range(nruns)]
    for p in poll:
        p.start()
    for p in poll:
        p.join()

    return list(results)
