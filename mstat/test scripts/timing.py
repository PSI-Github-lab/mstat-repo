from struct import unpack
import time
import random
import multiprocessing
from multiprocessing import Pool
from scipy import rand
import tqdm

random.seed(42)


def list_append(args):
    (count, id, out_list) = args
    """
    Creates an empty list and then appends a 
    random number to the list 'count' number
    of times. A CPU-heavy operation!
    """
    for i in range(count):
        out_list.append(random.random())

    return out_list

def multiproc1(size, procs):
    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list 
    jobs = []
    out_list = list()
    for i in range(0, procs):
        arg = ((size, i, out_list),)
        process = multiprocessing.Process(target=list_append, 
                                          args=arg)
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)      
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in tqdm.tqdm(jobs):
        j.join()

    return out_list

def multiproc2(size, procs):
    out_list = list()
    pool = multiprocessing.Pool(processes=procs)
    args = [(size, 0, out_list)] * procs
    return list(pool.imap(list_append, args)) #, total=len(args)))
    return out_list

def main(size, procs):
    multiproc1(size, procs)

    print("List processing complete.")

if __name__ == "__main__":
    size = 10000000   # Number of random numbers to add
    procs = 10   # Number of processes to create

    start_time = time.time()
    l = multiproc1(size, procs)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(len(l))

    start_time = time.time()
    for i in range(0, procs):
        out_list = list()
        list_append((size, i, out_list),)
    print("--- %s seconds ---" % (time.time() - start_time))