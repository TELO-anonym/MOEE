from moee.util.plotting import *

# settings define all the results we wish to process
results_dir = r'..\results'

problem_names = ['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice',
                 'Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel',
                 'Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10',
                 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']

problem_names_for_paper = ['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice',
                           'Cosines', 'ACKLEY', 'GRIEWANK', 'SixHumpCamel',
                           'Hartmann6', 'GSobol', 'ACKLEY', 'GRIEWANK',
                           'ACKLEY', 'GRIEWANK', 'push4', 'push8']

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
method_names = ['MOEE_EI_topsis', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'PI_nsga2', 'EI', 'UCB', 'PI',
                'EI_LBFGS', 'UCB_LBFGS']
method_names_for_paper = ['MOEE$_{EI}$', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'PI', 'EI', 'LCB',
                          'PI$_{LBFGSB}$', 'EI$_{LBFGSB}$', 'LCB$_{LBFGSB}$']

save_images = True
# load in all the optimisation results
results = process_results(results_dir, problem_names, method_names, budget=250, exp_no_start=1, exp_no_end=30)

plot_boxplots(results,
              # [50, 150, 250],
              [250],
              problem_names,
              problem_names_for_paper,
              problem_logplot,
              method_names,
              method_names_for_paper,
              LABEL_FONTSIZE=22,
              TITLE_FONTSIZE=25,
              TICK_FONTSIZE=20,
              save=save_images)

# plot_boxplots_combined(results,
#               [50, 150, 250],
#               ['Cosines', 'logSixHumpCamel', 'logGSobol'],
#               ['Cosines', 'logSixHumpCamel', 'logGSobol'],
#               [True, True, False],
#               method_names,
#               method_names_for_paper,
#               LABEL_FONTSIZE=22,
#               TITLE_FONTSIZE=25,
#               TICK_FONTSIZE=20,
#               save=save_images)
