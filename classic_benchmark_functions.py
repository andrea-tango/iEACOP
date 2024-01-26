import math

from iEACOP.bEACOP import bEACOP
from iEACOP.iEACOP import iEACOP


def Mishra01(individual):
    D = len(individual)
    x_n = D
    accum = 0
    for i in range(D - 1):
        accum += individual[i]
    x_n -= accum
    return (1 + x_n) ** x_n


def Quintic(individual):
    D = len(individual)
    accum = 0
    for i in range(D):
        accum += math.fabs(
            individual[i] ** 5 + 3 * individual[i] ** 4 + 4 * individual[i] ** 3 + 2 * individual[i] ** 2 - 10 *
            individual[i] - 4)
    return accum


def Michalewicz(individual):
    D = len(individual)
    M = 10
    accum = 0
    for i in range(D):
        accum += math.sin(individual[i]) * (math.sin((i * individual[i] ** 2.) / math.pi)) ** (2. * M)
    return -accum


def Schubert(individual):
    D = len(individual)
    totale = 1
    for i in range(D):
        accum = 0.0
        for j in range(5):
            accum += j * math.cos((j + 1) * individual[i] + j)
        totale *= accum
    return totale


def Alpine(individual):
    D = len(individual)
    accum = 0.0
    for i in range(D):
        accum += math.fabs(individual[i] * math.sin(individual[i]) + 0.1 * individual[i])
    return accum


def Bohachevsky(individual):
    D = len(individual)
    accum = 0.0
    for i in range(D - 1):
        accum += (individual[i] ** 2 + 2 * individual[i + 1] ** 2 - 0.3 * math.cos(
            3 * math.pi * individual[i]) - 0.4 * math.cos(4 * math.pi * individual[i + 1]) + 0.7)
    return accum


def Ferretti(individual):
    D = len(individual)
    accum = 30.0
    for i in range(D):
        accum += math.fabs(individual[i])
    return accum


def Plateau(individual):
    D = len(individual)
    accum = 30.0
    for i in range(D):
        accum += math.floor(individual[i])
    return accum


def XinSheYang(individual):
    D = len(individual)
    numer = 0.0
    denom = 0.0
    expo = 0.0
    for i in range(D):
        numer += math.fabs(individual[i])
    for i in range(D):
        expo += math.sin(individual[i] ** 2.)
    denom = math.exp(expo)
    return numer / denom


def Vincent(individual):
    D = len(individual)
    accum = 0.0
    for i in range(D):
        accum += math.sin(10 * math.log(individual[i]))
    return (1.0 / D) * accum


def Vincent2(individual):
    D = len(individual)
    accum = 0.0
    for i in range(D):
        accum += math.sin(10 * math.log(individual[i]))
    return accum


def Griewank(individual):
    accum = 1.0
    n = len(individual)
    sub1 = 0.0
    for i in range(n):
        sub1 = sub1 + math.pow(individual[i], 2) / 4000
    sub2 = 1.0
    for i in range(n):
        sub2 = sub2 * math.cos(individual[i] / (1 + math.sqrt(i)))
    accum = accum + sub1 - sub2
    return accum


def Ackley(individual):
    n = len(individual)
    accum = 20.0 + math.e
    sub1 = 0.0
    for i in range(n):
        sub1 = sub1 + math.pow(individual[i], 2)
    sub1 = sub1 / n
    accum = accum - 20.0 * math.exp(-0.2 * math.sqrt(sub1))
    sub2 = 0.0
    for i in range(n):
        sub2 = sub2 + math.cos(2 * individual[i] * math.pi)
    sub2 = sub2 / n
    accum = accum - math.exp(sub2)
    return accum


def Rastrigin(individual):
    A = 10.0
    n = len(individual)
    accum = A * n
    for c in range(n):
        accum = accum + (individual[c] * individual[c] - A * math.cos(2 * individual[c] * math.pi))
    return accum


def Schwefel(individual):
    n = len(individual)
    accum = 418.9829 * n
    subacc = 0.0
    for i in range(n):
        subacc = subacc + individual[i] * math.sin(math.sqrt(math.fabs(individual[i])))
    accum = accum - subacc
    return accum


def Rosenbrock(individual):
    accum = 0.0
    n = len(individual)
    for c in range(n - 1):
        # accum += ( 100.0*pow(individual[c+1]-math.pow(individual[c],2), 2) math.pow(1.0-individual[c],2) +  )
        accum += 100.0 * (individual[c] ** 2 - individual[c + 1]) ** 2 + (individual[c] - 1) ** 2
    return accum


def Nobile1(individual):
    first = 0.0
    n = len(individual)
    for c in range(n):
        first += 1.0 / (0.001 + math.exp(individual[c] - 10. ** (-n)))

    innerloop = 0.0
    for c in range(n):
        innerloop += (individual[c] - 10. ** (-n)) ** 2.

    second = 1.0 / (0.001 + math.sqrt(innerloop))
    return math.sin(first) - second


def Nobile3(individual):
    n = len(individual)

    first = 0.0
    for c in range(n):
        first += 1.0 / (0.001 + math.exp(individual[c] - 10 ** (-c)))

    innerloop = 0.0
    for c in range(n):
        innerloop += (individual[c] - 10 ** (-c)) ** 2

    second = 1.0 / (0.001 + math.sqrt(innerloop))
    return math.sin(first) - (second)


def Nobile2(individual):
    accum = 0.0
    n = len(individual)
    for c in range(n):
        accum += individual[c] ** ((1.0 * (c + 1)) / (2 * n))
    return accum


def Alpine_shifted(individual):
    K = 1e6

    D = len(individual)
    accum = 0.0
    for i in range(D):
        accum += math.fabs((individual[i] - K) * math.sin((individual[i] - K)) + 0.1 * (individual[i] - K))
    return accum


def my_fitness(individual):
    accum = 1.0
    n = len(individual)
    sub1 = 0.0
    for i in range(n):
        sub1 = sub1 + math.pow(individual[i], 2) / 4000
    sub2 = 1.0
    for i in range(n):
        sub2 = sub2 * math.cos(individual[i] / (1 + math.sqrt(i)))
    accum = accum + sub1 - sub2
    return accum


if __name__ == '__main__':
    K = 1e6

    limits = {}

    limits["Ackley"] = [-30, 30]
    limits["Alpine"] = [-10, 10]
    limits["Bohachevsky"] = [-15, 15]
    limits["Griewank"] = [-600, 600]
    limits["Quintic"] = [-10, 10]
    limits["Plateau"] = [-5.12, 5.12]
    limits["Rastrigin"] = [-5, 5]
    limits["Rosenbrock"] = [-5, 10]
    limits["Schwefel"] = [-500, 500]
    limits["Schubert"] = [-10, 10]
    limits["Vincent"] = [0.25, 10]
    limits["Vincent2"] = [0.25, 10]
    limits["XinSheYang"] = [-2 * math.pi, 2 * math.pi]
    limits["Michalewicz"] = [0, math.pi]
    limits["Mishra01"] = [0, 1]
    limits["Nobile1"] = [1e-10, 10]
    limits["Nobile2"] = [1e-10, 10]
    limits["Nobile3"] = [1e-10, 10]
    limits["my_fitness"] = [-600, 600]

    function = "Quintic"

    dimensioni = 100
    boundaries = [limits[function]] * dimensioni
    individuals = 30

    fiteval = 10000 * dimensioni

    beacop = bEACOP(boundaries, creation_method={'name': "uniform"}, verbose=False)
    ieacop = iEACOP(boundaries, creation_method={'name': "uniform"}, verbose=False)

    print("@" * 100)
    print("* Testing bEACOP'")
    solution = beacop.solve(fiteval,
                            n_individuals=individuals,
                            fitness_function=eval(function),
                            coeff=2 * individuals,
                            seed=42)

    print("Best fitness is:", solution.calculated_fitness)
    print()
    print("@" * 100)

    print("* Testing iEACOP'")
    solution = ieacop.solve(fiteval,
                            n_individuals=individuals,
                            fitness_function=eval(function),
                            coeff=2 * individuals,
                            seed=42)

    print("Best fitness is:", solution.calculated_fitness)
    print()
    print("@" * 100)
