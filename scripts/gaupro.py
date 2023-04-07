#!/usr/bin/env python3

# from https://thuijskens.github.io/2016/12/29/bayesian-optimisation/


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
### --- Figure config
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
LEGEND_SIZE = 15
TITLE_SIZE = 25
AXIS_SIZE = 15

from emukit.test_functions import forrester_function
from emukit.core.loop import UserFunctionWrapper
from emukit.core.initial_designs import RandomDesign

target_function, space = forrester_function()
design = RandomDesign(space) 
X = design.get_samples(30)
Y = target_function(X)

from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.core.loop import UserFunctionResult

num_iterations = 10

bo = GPBayesianOptimization(variables_list=space.parameters, X=X, Y=Y)
results = None

for _ in range(num_iterations):
    X_new = bo.get_next_points(results)
    Y_new = target_function(X_new)
    results = [UserFunctionResult(X_new[0], Y_new[0])]

X = bo.loop_state.X
Y = bo.loop_state.Y

x = np.arange(0.0, 1.0, 0.01)
y = target_function(x)

plt.figure()
plt.plot(x, y)
print(f'size {X.size}')
for i, (xs, ys) in enumerate(zip(X, Y)):
    plt.plot(xs, ys, 'ro', markersize=10 + 10 * (i+1)/len(X))

print(X,Y)
plt.show()