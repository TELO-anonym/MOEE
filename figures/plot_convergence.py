from moee.util.plotting import *

# settings define all the results we wish to process
results_dir = r'..\results'

problem_names = [
    'WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice',
    'Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel',
    'Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10',
    'ACKLEY_20', 'GRIEWANK_20',
    'push4', 'push8'
]

problem_names_for_paper = [
    'WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice',
    'Cosines', 'ACKLEY', 'GRIEWANK', 'SixHumpCamel',
    'Hartmann6', 'GSobol', 'ACKLEY', 'GRIEWANK',
    'ACKLEY', 'GRIEWANK',
    'push4', 'push8'
]


# boolean indicating whether the problem should be plotted with a log axis
# number of problems equals numbers of booleans
problem_logplot = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

# Comparison of our proposal and the state-of-the-art acquisition functions in terms of convergence
# method_names = ['MOEE_UCB_topsis', 'eFront_eps0.1', 'eRandom_eps0.1', 'PI_nsga2', 'EI', 'UCB']
# method_names_for_paper = ['MOEE', '$\epsilon$-PF', '$\epsilon$-RS', 'PI', 'EI', 'LCB']

# Ablation study within proposed acquisition functions in terms of convergence
# method_names = ['MOEE_UCB_topsis', 'MOEE_UCB_PF', 'MOEE_UCB_exploit', 'MOEE_UCB_random', 'MOEE_UCB_switch']
# method_names_for_paper = ['MOEE', 'MOEE$_{PF}$', 'MOEE$_{exploit}$', 'MOEE$_{e\_PF}$', 'MOEE$_{switch}$']

# Comparison of nsga2 with lbfgs
# method_names = ['MOEE_UCB_topsis', 'PI', 'EI_LBFGS', 'UCB_LBFGS', 'AWEI', 'MGFI']
# method_names_for_paper = ['MOEE', 'PI$_{LBFGSB}$', 'EI$_{LBFGSB}$', 'LCB$_{LBFGSB}$', 'SAWEI', 'MGFI']

# MOEE_{EI}
method_names = ['MOEE_EI_topsis', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'PI_nsga2', 'EI', 'UCB', 'PI', 'EI_LBFGS', 'UCB_LBFGS']
method_names_for_paper = ['MOEE$_{EI}$', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'PI', 'EI', 'LCB', 'PI$_{LBFGSB}$', 'EI$_{LBFGSB}$', 'LCB$_{LBFGSB}$']


save_images = True
# load in all the optimisation results
results = process_results(results_dir, problem_names, method_names, budget=250, exp_no_start=1, exp_no_end=30)

# load the best push8 results found by uniformly sampling 100000 decision vectors
# and locally optimising the best 100 of these with L-BFGS-B.
# push8_estimated_optima[i] contains the (i+1)'th problem instance's
with np.load(r'../training_data/push8_best_solutions.npz') as data:
    push8_estimated_optima = data['results']

# calculate the distance from each push8 run to the corresponding estimated optima
for method_name in method_names:
    dist = results['push8'][method_name] - push8_estimated_optima[:, None][:30]

    # simple sanity checking - check the distance between them is >= 0, meaning that
    # the estimated optima are better or equal to the evaluated function values

    assert np.all(dist >= 0)

    results['push8'][method_name] = dist

plot_convergence(results,
                 problem_names,
                 problem_names_for_paper,
                 problem_logplot,
                 method_names,
                 method_names_for_paper,
                 LABEL_FONTSIZE=15,
                 TITLE_FONTSIZE=20,
                 TICK_FONTSIZE=15,
                 LEGEND_FONTSIZE=20,
                 save=True)

# plot_convergence_combined(results,
#                           ['WangFreitas', 'BraninForrester', 'Branin', 'GoldsteinPrice'],
#                           ['WangFreitas', 'BraninForrester', 'Branin', 'GoldsteinPrice'],
#                           [True, True, True, True],
#                           method_names,
#                           method_names_for_paper,
#                           LABEL_FONTSIZE=22,
#                           TITLE_FONTSIZE=25,
#                           TICK_FONTSIZE=20,
#                           save=True)
#
# plot_convergence_combined(results,
#                           ['Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel'],
#                           ['Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel'],
#                           [True, True, True, True],
#                           method_names,
#                           method_names_for_paper,
#                           LABEL_FONTSIZE=22,
#                           TITLE_FONTSIZE=25,
#                           TICK_FONTSIZE=20,
#                           save=True)
#
# plot_convergence_combined(results,
#                           ['Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10'],
#                           ['Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10'],
#                           [True, True, True, True],
#                           method_names,
#                           method_names_for_paper,
#                           LABEL_FONTSIZE=22,
#                           TITLE_FONTSIZE=25,
#                           TICK_FONTSIZE=20,
#                           save=True)
#
# plot_convergence_combined(results,
#                           ['ACKLEY_20',  'GRIEWANK_20'],
#                           ['ACKLEY_20',  'GRIEWANK_20'],
#                           [True, True, True, True],
#                           method_names,
#                           method_names_for_paper,
#                           LABEL_FONTSIZE=22,
#                           TITLE_FONTSIZE=25,
#                           TICK_FONTSIZE=20,
#                           save=True)
