import os
from pathlib import Path
import numpy as np
from scipy.io import savemat
from io import StringIO
import multiprocessing as mp
from shutil import make_archive

from CEC.functions import functions as cec_functions
from iEACOP.iEACOP import iEACOP


def run_CEC(x, fx_n):
    if fx_n > 1:
        fx_n += 1
    return cec_functions[f'CEC17-F{fx_n}'](x)


def run(outdir, function, boundaries, dimensions, fiteval, optimization_method):
    path_out = outdir + os.sep + "Function%d" % function + os.sep + "%dD" % dimensions

    ieacop = iEACOP(boundaries,
                    dimensions,
                    path_out=path_out,
                    creation_method={"name": "uniform"},
                    verbose=False)

    for opt in range(25):
        _ = ieacop.solve(fiteval,
                         n_individuals=None,
                         fitness_function=run_CEC,
                         fitness_args=function,
                         optimization_method=optimization_method,
                         seed=opt,
                         rep=opt)


def main():
    outdir = "Results_CEC17_apr26"
    dimensions = 30
    evals_per_dimension = 1000  # set the number of function evaluations per dimension
    fitevals = {dimensions: evals_per_dimension * dimensions}

    functions = tuple(range(1, 30))
    values = tuple((k + 1) * 100 if k > 1 else k * 100 for k in functions)  # use the actual best fitness values
    optimum = dict(zip(functions, values))

    boundaries = [-100, 100]

    print("* Test on %dD" % dimensions)
    jobs = []
    for function in functions:
        print(" * Running function %s" % function)
        p = mp.Process(target=run,
                       args=(outdir, function, boundaries, dimensions, fitevals[dimensions], None))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # Saving the results as mat files
    sample_size = int(evals_per_dimension / 10)
    sample_resolution = 10 * dimensions
    algorithm_name = "iEACOP"  # PaperID?
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_dir = f"{algorithm_name}_results_{dimensions}D"
    Path(f'{outdir}{os.sep}{save_dir}').mkdir(parents=True, exist_ok=True)
    for function in functions:
        a_reps = np.full((sample_size, 25), np.nan)
        label_reps = []
        path_out = outdir + os.sep + "Function%d" % function + os.sep + "%dD" % dimensions
        for rep in range(25):
            fitness_file = path_out + os.sep + f"{algorithm_name}_fitness_rep%d" % rep
            with open(fitness_file, "r") as stream:
                fitness = np.genfromtxt(StringIO(stream.read()), delimiter="\t")
            error_value = fitness[:, 1] - optimum[function]
            for i in range(sample_size):
                sampling_point = sample_resolution * (i + 1)
                j = 0
                while j < len(fitness) and fitness[j, 0] <= sampling_point:
                    j += 1
                a_reps[i, rep] = error_value[j - 1]
            label_reps.append(f"Run {rep + 1}")
        mdic = {"Min_EV": a_reps, "Run": label_reps}
        file_name = f"{outdir}{os.sep}{save_dir}{os.sep}{algorithm_name}_F{function}_Min_EV.mat"
        if os.path.exists(file_name):
            os.remove(file_name)
        savemat(file_name, mdic)
        txt_file_name = f"{outdir}{os.sep}{algorithm_name}_F{function}_Min_EV.txt"
        if os.path.exists(txt_file_name):
            os.remove(txt_file_name)
        np.savetxt(txt_file_name, a_reps)
    os.chdir(outdir)
    make_archive(f"{algorithm_name}_results_{dimensions}D", "zip", save_dir)


if __name__ == "__main__":
    main()
