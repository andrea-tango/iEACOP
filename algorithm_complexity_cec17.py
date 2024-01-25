from time import time
from pathlib import Path
import os

from CEC.functions import functions as cec_functions
from run_cec17 import run_CEC
from iEACOP.iEACOP import iEACOP


def compute_t_1():
    dimensions = 30
    evals = 10000
    overall = []
    x = [0 for _ in range(dimensions)]
    for i in range(1, 30):
        if i > 1:
            i += 1  # remember that F2 has been deleted
        problem_computing_time = 0
        function = cec_functions[f'CEC17-F{i}']
        for j in range(evals):
            tic = time()
            function(x)
            toc = time() - tic
            problem_computing_time += toc
        overall.append(problem_computing_time)
    indicator = sum(overall) / len(overall)
    return indicator


def compute_t_2():
    boundaries = [-100, 100]
    dimensions = 30
    ieacop = iEACOP(boundaries=boundaries,
                    dimensions=dimensions,
                    write=False)
    overall = []
    for i in range(1, 30):
        tic = time()
        iEACOP.solve(max_fitness_evaluations=10000,
                     n_individuals=None,
                     fitness_function=run_CEC,
                     fitness_args=i,
                     optimization_method=None,
                     seed=None,
                     rep=None)
        toc = time() - tic
        overall.append(toc)
    indicator = sum(overall) / len(overall)
    return indicator


def main():
    out_path = 'Complexity_CEC17'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    t_1 = compute_t_1()
    print('t_1', t_1)
    with open(f'{out_path}{os.sep}T1.txt', 'w') as stream:
        stream.write(str(t_1))

    overall = []
    for i in range(5):
        t_2 = compute_t_2()
        print(f't_2_R{i}', t_2)
        overall.append(t_2)
        with open(f'{out_path}{os.sep}T2_R{i}.txt', 'w') as stream:
            stream.write(str(t_2))
    t_2_mean = sum(overall) / len(overall)
    print('t_2_mean', t_2_mean)
    with open(f'{out_path}{os.sep}T2_mean.txt', 'w') as stream:
        stream.write(str(t_2_mean))

    algorithm_complexity = (t_2_mean - t_1) / t_1
    print('algorithm_complexity', algorithm_complexity)
    with open(f'{out_path}{os.sep}algorithm_complexity.txt', 'w') as stream:
        stream.write(str(algorithm_complexity))


if __name__ == '__main__':
    main()
