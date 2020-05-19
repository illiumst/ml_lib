import multiprocessing as mp
import time


def run_n_in_parallel(f, n, processes=0, **kwargs):
    processes = processes if processes else mp.cpu_count()
    output = mp.Queue()
    kwargs.update(output=output)
    # Setup a list of processes that we want to run

    processes = [mp.Process(target=f, kwargs=kwargs) for _ in range(n)]
    # Run processes
    results = []
    for p in processes:
        p.start()
    while len(results) != n:
        time.sleep(1)
        # Get process results from the output queue
        results.extend([output.get() for _ in processes])

    # Exit the completed processes
    for p in processes:
        p.join()

    return results

raise NotImplementedError()
