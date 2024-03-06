import multiprocessing
import time

import dask.bag as db

# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as PathosProcessingPool


# A very simple function (so my computer doesn't explode when trying to benchmark these)
def weird_function(args):
    x, y = args
    return x + y


def process_with_pool(pool_method, the_size, func):
    global result
    start_time = time.time()

    if pool_method == "map":
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
            data = [(x, y) for x in range(the_size) for y in range(the_size)]
            result = pool.map(func, data)
    elif pool_method == "starmap":
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
            data = [((x, y),) for x in range(the_size) for y in range(the_size)]
            result = pool.starmap(func, data)
    elif pool_method == "apply":
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
            data = [(x, y) for x in range(the_size) for y in range(the_size)]
            result = [pool.apply(func, args=(args,)) for args in data]
    elif pool_method == "imap":
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
            data = [(x, y) for x in range(the_size) for y in range(the_size)]
            result = list(pool.imap(func, data))
    elif pool_method == "imap_unordered":
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
            data = [(x, y) for x in range(the_size) for y in range(the_size)]
            result = list(pool.imap_unordered(func, data))
    elif pool_method == "pathos":
        pool = PathosProcessingPool(nodes=multiprocessing.cpu_count() * 2)
        data = [(x, y) for x in range(the_size) for y in range(the_size)]
        result = pool.map(func, data)
        pool.close()
        pool.join()
    elif pool_method == "dask_1":
        data = [(x, y) for x in range(the_size) for y in range(the_size)]
        bag = db.from_sequence(data)
        result = list(bag.map(func).compute())

    end_time = time.time()
    return result, end_time - start_time


if __name__ == "__main__":
    size = 400
    pool_methods = [
        "map",
        "starmap",
        "apply",
        "imap",
        "imap_unordered",
        "pathos",
        "dask_1",
    ]

    for method in pool_methods:
        result, elapsed_time = process_with_pool(method, size, weird_function)
        print(f"Time taken for pool.{method}: {elapsed_time} seconds")
