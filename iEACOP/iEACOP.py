import os
import sys
import math
import random as rnd
from pathlib import Path
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.stats import qmc
from scipy.special import gamma


class Individual(object):


	def __init__(self):
		self.x = []
		self.calculated_fitness = sys.float_info.max
		self.old_fitness = sys.float_info.max
		self.n_stuck = 0



class iEACOP(object):  # evolutionary algorithm for complex-process optimization


	def __repr__(self):
		return str(self.id)


	def __init__(self, boundaries, dimensions=None, path_out=".", write=True, creation_method={'name': "uniform"}, n_change=20, epsilon=1e-3, verbose=False, id=None):

		self.best = None  # best
		self.seed = None

		# if id is not None:
		self.id = id if id is not None else None

		self.init_solutions = []
		self.solutions = []
		self.offspring = []
		self.fitness_evaluations = 0
		self.local_fit_evals = 0
		self.creation_method = creation_method
		self.n_change = n_change
		self.max_fitness_evaluations = 0
		self.iterations = 0
		self.path_out = path_out
		self.verbose = verbose
		self.write = write

		self.fitness = None
		self.fitness_args = None
		self.opt_method = None
		self.changeable = None

		self.local_solutions = []
		self.balance = 0.5
		self.epsilon = epsilon

		self.apply_local_search = False
		self.last_best_local = 0
		self.last_restart_local = 0

		self.n1 = 1
		self.n2 = 10

		self.boundaries = self.check_boundaries(boundaries, dimensions)

		Path(self.path_out).mkdir(parents=True, exist_ok=True)  # python >= 3.5

		self.fitness_file = None
		self.positions_file = None


	def check_boundaries(self, boundaries, dimensions):

		is_nested = all(isinstance(bound, list) for bound in boundaries)

		if is_nested:
			is_lb_ub = all(len(bound) == 2 for bound in boundaries)

			if is_lb_ub:
				if dimensions is not None:
					if len(boundaries) == 1 and dimensions != 0:
						return boundaries * dimensions

					elif len(boundaries) == dimensions:
						return boundaries

					else:
						print("Error! The number of dimensions and the number of provided boundaries must be equal.")
						exit(-2)
				else:
					return boundaries

			else:
				print(
					"Error! The boundaries must be either a list [l_bound, u_bound] or a list of lists [[l_bound, u_bound]], with a list for each dimension.")
				exit(-1)
		else:
			if dimensions is not None:
				if len(boundaries) == 2:
					return [boundaries] * dimensions
				else:
					print(
						"Error! The boundaries must be either a list [l_bound, u_bound] or a list of lists [[l_bound, u_bound]], with a list for each dimension.")
					exit(-1)
			else:
				return [boundaries]


	def set_fitness(self, fitness, fitness_args):

		self.fitness = fitness
		self.fitness_args = fitness_args


	def update_calculated_fitness(self, kind=0):

		if kind == 0:
			for ind in self.init_solutions:
				ind.calculated_fitness = self.evaluate_individual(ind.x)

			self.init_solutions.sort(key=lambda x: x.calculated_fitness)

		elif kind == 1:
			n = len(self.offspring)
			for i in range(n):
				for ind in self.offspring[i]:
					ind.calculated_fitness = self.evaluate_individual(ind.x)

				self.offspring[i].sort(key=lambda x: x.calculated_fitness)


	def evaluate_individual(self, x):

		self.fitness_evaluations += 1
		self.local_fit_evals += 1

		if self.fitness is None: raise Exception("Error: Fitness function not valid")

		if self.fitness_args is None:
			return self.fitness(x)

		else:
			return self.fitness(x, self.fitness_args)


	def heuristic(self, dim):

		d = 10 * dim
		n = int(math.ceil((1 + math.sqrt(1 + 4 * d)) / 2.0))

		if n % 2 == 0:
			return n
		else:
			return n + 1


	def generate(self, n_individuals):

		positions = []
		dim = len(self.boundaries)

		if self.creation_method['name'] == "uniform":

			for m in range(n_individuals):
				position = []
				for i in range(dim):
					position.append(
						self.boundaries[i][0] + (self.boundaries[i][1] - self.boundaries[i][0]) * rnd.random())

				positions.append(np.array(position))

		elif self.creation_method['name'] == "logarithmic":

			# se il range e' negativo, allora generiamo con un ordine di grandezza inferiore arbitrario (todo custom)
			# e poi applichiamo il segno

			for m in range(n_individuals):
				position = []
				for i in range(dim):
					if self.boundaries[i][0] < 0:
						minimo = -5
						massimo = math.log(self.boundaries[i][1])
						res = math.exp(minimo + (massimo - minimo) * rnd.random())
						if rnd.random() > .5:
							res *= -1
						position.append(res)
					else:
						minimo = math.log(self.boundaries[i][0])
						massimo = math.log(self.boundaries[i][1])
						position.append(math.exp(minimo + (massimo - minimo) * rnd.random()))

				positions.append(np.array(position))

		elif self.creation_method['name'] == "latin":

			l_bounds = [self.boundaries[i][0] for i in range(dim)]
			u_bounds = [self.boundaries[i][1] for i in range(dim)]

			sampler = qmc.LatinHypercube(d=dim, seed=self.seed + rnd.randint(1, 100))  # , optimization="random-cd")
			sample = sampler.random(n=n_individuals)

			sample = qmc.scale(sample, l_bounds, u_bounds)

			positions = list(sample)

		else:
			print("unknown individual initialization mode")

		return positions


	def create_individuals(self, n_individuals, dim, coeff):

		self.solutions = []
		self.init_solutions = []

		if n_individuals is None or n_individuals == 0:
			n_individuals = self.heuristic(dim)

		if self.verbose:
			print(" * The problem has", dim, "dimensions")

		if coeff == 0:
			m = 10 * dim
		else:
			m = coeff

		if m < n_individuals:
			print("Error! Too many individuals wrt the problem dimension")
			exit(-3)

		positions = self.generate(m)

		for i in range(m):
			ind = Individual()
			ind.x = positions[i]
			self.init_solutions.append(ind)

		self.update_calculated_fitness(kind=0)

		n = int(round(n_individuals / 2.0))
		for i in range(n):
			self.solutions.append(self.init_solutions[i])

		for i in range(n_individuals - n):
			idx = rnd.randint(n, m - 1)
			self.solutions.append(self.init_solutions[idx])

		self.solutions.sort(key=lambda x: x.calculated_fitness)

		self.best = deepcopy(self.solutions[0])

		if self.verbose:
			print(" * %d individuals created\n" % len(self.solutions))

		if self.write:
			self.write_results()
			# print('* wrote created individuals')


	def check_diversity(self):

		dim = len(self.boundaries)

		for i in range(len(self.solutions) - 1):
			xi = self.solutions[i].x
			for j in range(i + 1, len(self.solutions)):
				xj = self.solutions[j].x

				values = np.absolute((xi - xj) / xj)
				maxval = np.max(values)

				if maxval < self.epsilon:
					ind = Individual()
					ind.x = self.generate(1)[0]
					ind.calculated_fitness = self.evaluate_individual(ind.x)

					self.solutions[j] = deepcopy(ind)


	def combination_method(self):

		n = len(self.solutions)
		dim = len(self.boundaries)

		self.offspring = []

		c1 = np.zeros(dim)
		c2 = np.zeros(dim)
		values = np.zeros(dim)
		alpha = 0
		beta = 0

		for i in range(n):
			x_new = []
			for j in range(n):

				if i != j:
					alpha = -1

					if i < j:
						alpha = 1

					beta = float((abs(j - i) - 1)) / float(n - 2)

					ind = Individual()

					for d in range(dim):
						delta = (self.solutions[j].x[d] - self.solutions[i].x[d]) / 2.0

						c1[d] = self.solutions[i].x[d] - delta * (1 + alpha * beta)
						c2[d] = self.solutions[i].x[d] + delta * (1 - alpha * beta)

						if c1[d] < self.boundaries[d][0]:
							c1[d] = self.boundaries[d][0]

						if c1[d] > self.boundaries[d][1]:
							c1[d] = self.boundaries[d][1]

						if c2[d] < self.boundaries[d][0]:
							c2[d] = self.boundaries[d][0]

						if c2[d] > self.boundaries[d][1]:
							c2[d] = self.boundaries[d][1]

						value = c1[d] + (c2[d] - c1[d]) * rnd.random()

						if value < self.boundaries[d][0]:
							value = self.boundaries[d][0]

						if value > self.boundaries[d][1]:
							value = self.boundaries[d][1]

						values[d] = value

					ind.x = deepcopy(values)
					x_new.append(ind)

			self.offspring.append(x_new)

		self.update_calculated_fitness(kind=1)


	def go_beyond(self, idx):

		dim = len(self.boundaries)

		xpr = deepcopy(self.solutions[idx])
		xch = deepcopy(self.offspring[idx][0])

		improvement = 1
		lambd = 1.0

		# c1 = np.zeros(dim)
		# c2 = np.zeros(dim)
		# values = np.zeros(dim)

		def sample_points_inside_hypersphere(n_dim, r, c, num, rng):
			lows = np.zeros(num)
			highs = np.ones(num)
			u = rng.uniform(low=lows, high=highs)
			x = rng.normal(loc=lows, scale=highs, size=(n_dim, num))
			s = np.sqrt(np.sum(x ** 2, axis=0))
			x = x / s
			return x * u ** (1 / n_dim) * r + np.expand_dims(c, axis=1)

		while xch.calculated_fitness < xpr.calculated_fitness:

			ind = Individual()

			c1 = xch.x - (xpr.x - xch.x) / lambd
			c2 = xch.x

			c12 = np.linalg.norm(c1 - c2, ord=2)
			# first attempt
			radius = c12 / 2  # maybe multiply by np.sqrt(dim)
			# second attempt
			# c12_hyperrectangle_volume = np.prod(np.abs(c2 - c1))  # desired hypersphere volume
			# radius = (c12_hyperrectangle_volume * gamma(1 + .5 * dim) / np.pi ** (dim / 2.)) ** (1. / dim)
			unit_vector = (c1 - c2) / c12
			center = unit_vector * radius + c2
			n_dim = len(self.boundaries)
			rng = np.random.default_rng(seed=self.id)
			xs = sample_points_inside_hypersphere(n_dim, r=radius, c=center, num=1, rng=rng)
			# samples may lie outside the search space
			lwb, upb = list(zip(*self.boundaries))
			xs = np.clip(xs, np.expand_dims(lwb, axis=1), np.expand_dims(upb, axis=1))
			if xs.ndim == 2:
				xs = np.squeeze(xs)

			values = xs
			# print(values.shape, values.dtype, values.ndim)
			# exit()

			# for d in range(dim):
			#
			# 	c1[d] = xch.x[d] - (xpr.x[d] - xch.x[d]) / lambd
			# 	c2[d] = xch.x[d]
			#
			# 	if c1[d] < self.boundaries[d][0]:
			# 		c1[d] = self.boundaries[d][0]
			#
			# 	if c1[d] > self.boundaries[d][1]:
			# 		c1[d] = self.boundaries[d][1]
			#
			# 	if c2[d] < self.boundaries[d][0]:
			# 		c2[d] = self.boundaries[d][0]
			#
			# 	if c2[d] > self.boundaries[d][1]:
			# 		c2[d] = self.boundaries[d][1]
			#
			# 	value = c1[d] + (c2[d] - c1[d]) * rnd.random()
			#
			# 	if value < self.boundaries[d][0]:
			# 		value = self.boundaries[d][0]
			#
			# 	if value > self.boundaries[d][1]:
			# 		value = self.boundaries[d][1]
			#
			# 	values[d] = value

			ind.x = deepcopy(values)

			ind.calculated_fitness = self.evaluate_individual(ind.x)

			xpr = deepcopy(xch)
			xch = deepcopy(ind)
			improvement += 1

			if improvement == 2:
				lambd /= 2.0
				improvement = 0

		return xpr


	def update_population(self):

		n = len(self.solutions)
		for i in range(n):
			if self.offspring[i][0].calculated_fitness < self.solutions[i].calculated_fitness:
				ind = self.go_beyond(i)
				self.solutions[i].calculated_fitness = ind.calculated_fitness
				self.solutions[i].x = deepcopy(ind.x)

				self.solutions[i].n_stuck = 0

			else:
				self.solutions[i].n_stuck += 1

				if self.solutions[i].n_stuck > self.n_change:
					# if self.solutions[i].calculated_fitness != self.best.calculated_fitness:
					# print "* individual", str(i), "re-initizalized"

					dim = len(self.solutions[i].x)
					ind = Individual()
					ind.x = self.generate(1)[0]
					ind.calculated_fitness = self.evaluate_individual(ind.x)

					self.solutions[i] = deepcopy(ind)

		self.solutions.sort(key=lambda x: x.calculated_fitness)

		if  self.solutions[0].calculated_fitness < self.best.calculated_fitness:
			self.best = deepcopy(self.solutions[0])
			self.last_best_local = 0
			self.last_restart_local = 0

			if self.iterations >= self.n_change*2:
				self.apply_local_search = True

		else:
			self.last_restart_local += 1


	def local_search(self, z):

		z_local = deepcopy(z)

		if self.opt_method in ("L-BFGS-B", "Powell"):  # "SLSQP", "Nelder-Mead", "TNC", "trust-constr"
			if self.fitness_args is None:
				res = minimize(self.fitness, z_local.x, method=self.opt_method, bounds=tuple(self.boundaries))
			else:
				res = minimize(self.fitness, z_local.x, self.fitness_args, method=self.opt_method, bounds=tuple(self.boundaries))
		else:
			if self.fitness_args is None:
				res = minimize(self.fitness, z_local.x, self.fitness_args, method=self.opt_method)
			else:
				res = minimize(self.fitness, z_local.x, method=self.opt_method)

		z_local.x = res.x
		z_local.calculated_fitness = res.fun

		self.fitness_evaluations += res.nfev

		return z_local


	def check_local_search(self, z):

		if z.calculated_fitness < self.best.calculated_fitness:
			self.last_best_local    = 0
			self.last_restart_local = 0
			self.apply_local_search = True

			if self.verbose:
				print(" * New global best found")
				print(" * Old best fitness: %.3e" % self.best.calculated_fitness)

			self.solutions[-1] = deepcopy(z)
			self.solutions.sort(key=lambda x: x.calculated_fitness)

			self.best = deepcopy(self.solutions[0])

			if self.verbose:
				print(" * New best fitness: %.3e" % self.best.calculated_fitness)

			return True

		else:

			self.last_best_local += 1
			self.last_restart_local += 1
			self.apply_local_search = False

			return False


	def evaluate_local_search(self, z, z1):

		evaluation = self.check_local_search(z)

		# new idea to improve
		if evaluation == False:

			self.old_opt_method = self.opt_method

			if self.old_opt_method == "Powell" and self.changeable:
				self.opt_method = "L-BFGS-B"

			elif self.opt_method == "L-BFGS-B" and self.changeable:
				self.opt_method = "Powell"

			if self.changeable:

				if self.verbose:
					print(" * No improvement detected! Trying '%s' instead of '%s' ..."%(self.opt_method, self.old_opt_method))

				z_new = self.local_search(z1)

				evaluation = self.check_local_search(z_new)

				if z_new.calculated_fitness < z.calculated_fitness:
					if self.verbose:
						print(" * Improvement detected! Setting '%s' as default ..."%self.opt_method)

					to_add = True
					for i,old in enumerate(self.local_solutions):
						if np.array_equal(z_new.x, old.x):
							to_add = False
							break

					if to_add:
						self.local_solutions.append(z_new)

				else:
					if self.verbose:
						print(" * No improvement detected! Resetting '%s' as default ..."%self.old_opt_method)
					self.opt_method = self.old_opt_method

					to_add = True
					for i,old in enumerate(self.local_solutions):
						if np.array_equal(z.x, old.x):
							to_add = False
							break

					if to_add:
						self.local_solutions.append(z)

		else:
			to_add = True
			for i,old in enumerate(self.local_solutions):
				if np.array_equal(z.x, old.x):
					to_add = False
					break

			if to_add:
				self.local_solutions.append(z)


	def apply_local1(self):

		if self.verbose:
			print("\n * Applying local search 1")

		z_best = deepcopy(self.best)

		z = self.local_search(z_best)

		self.evaluate_local_search(z, z_best)

		if self.verbose:
			print()


	def apply_local2(self):

		if self.verbose:
			print("\n * Applying local search 2")

		y = np.array(deepcopy(self.offspring)).flatten()

		yq = {}
		yd = {}
		for i in range(len(y)):
			yq[i] = y[i].calculated_fitness
			distances = np.array([distance.euclidean(y[i].x, old.x) for old in self.local_solutions])
			yd[i] = np.min(distances)

		yq = dict(sorted(yq.items(), key=lambda x:x[1]))
		yd = dict(sorted(yd.items(), key=lambda x:x[1], reverse=True))

		score = np.zeros(len(y))
		for k in range(len(score)):

			i = list(yq.keys()).index(k)
			j = list(yd.keys()).index(k)

			score[k] = (1 - self.balance) * i + self.balance * j

		idx = np.argmin(score)

		y_min = deepcopy(y[idx])

		z = self.local_search(y_min)

		self.evaluate_local_search(z, y_min)

		if self.verbose:
			print()


	def iterate(self):

		self.check_diversity()
		self.combination_method()
		self.update_population()

		if self.apply_local_search and self.fitness_evaluations > self.max_fitness_evaluations * .5:

			if self.last_best_local == 0:
				self.apply_local1()

				if len(self.local_solutions) > 1:
					self.apply_local2()

			else:

				if len(self.local_solutions) == 0:
					if self.local_fit_evals >= self.n1:
						self.apply_local1()

				elif self.local_fit_evals >= self.n2 and self.iterations%self.n2==0:
						self.apply_local2()

			self.local_fit_evals = 0

		if self.last_restart_local >= self.n_change:

			self.apply_local_search = True
			self.last_restart_local = 0

		self.iterations += 1


	def termination_criterion(self):

		if self.verbose:
			print("\t* Iteration: %5d - Fitness evaluations: %6d - Best fitness: %.3e" % (
				self.iterations, self.fitness_evaluations, self.best.calculated_fitness))

		if self.fitness_evaluations >= self.max_fitness_evaluations:
			if self.verbose:
				print("\n * Maximum number of fitness evaluations reached")
				print(" * Best fitness detected:", self.best.calculated_fitness)
				print()
			return True
		else:
			return False


	def write_results(self):

		with open(self.fitness_file, "a") as fo:
			fo.write(str(self.fitness_evaluations) + "\t")
			fo.write(str(self.best.calculated_fitness) + "\n")

		with open(self.positions_file, "a") as fo1:
			for i in range(0, len(self.best.x)):
				fo1.write(str(self.best.x[i]) + "\t")
			fo1.write("\n")


	def solve(self, max_fitness_evaluations=10000, n_individuals=None, fitness_function=None, fitness_args=None, optimization_method=None, coeff=0, seed=None, rep=None):

		if self.verbose:
			print(" * Optimization process started ...\n")

		self.fitness_evaluations = 0
		self.iterations = 0
		self.apply_local_search = False

		if seed is not None:
			self.seed = seed
			np.random.seed(self.seed)
			rnd.seed(self.seed)

		if rep is None:
			self.fitness_file = self.path_out + os.sep + "iEACOP_fitness"
			self.positions_file = self.path_out + os.sep + "iEACOP_positions"

		else:
			self.fitness_file = self.path_out + os.sep + "iEACOP_fitness_rep%d" % rep
			self.positions_file = self.path_out + os.sep + "iEACOP_positions_rep%d" % rep

		if os.path.exists(self.fitness_file):
			os.remove(self.fitness_file)

		if os.path.exists(self.positions_file):
			os.remove(self.positions_file)

		self.set_fitness(fitness_function, fitness_args)

		self.create_individuals(n_individuals, len(self.boundaries), coeff)

		if optimization_method is None:
			self.opt_method = "Powell"
			self.changeable = True

		else:
			self.opt_method = optimization_method
			self.changeable = False

		if self.verbose:
			print(" * Using %s for the local search step\n" % self.opt_method)

		self.max_fitness_evaluations = max_fitness_evaluations

		while not self.termination_criterion():

			self.iterate()

			if self.write:
				self.write_results()

		return self.best
