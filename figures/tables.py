# Main results table
from moee.util.plotting import *

results_dir = r'..\results'

# problem_rows = [['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel'],
#                 ['Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']]

# problem_names = ['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2','SixHumpCamel',
#                  'Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']

problem_rows = [['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice'],
                ['Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel'],
                ['Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10'],
                ['ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']]

problem_names = ['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2',
                 'SixHumpCamel',
                 'Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']

problem_paper_rows = problem_rows

problem_dim_rows = [[1, 2, 2, 2],
                    [2, 2, 2, 2],
                    [6, 10, 10, 10],
                    [20, 20, 4, 8]]

# method_names = ['MOEE_UCB_topsis', 'eFront_eps0.1', 'eRandom_eps0.1', 'PI_nsga2', 'EI', 'UCB']
# method_names_for_table = ['MOEE', '$\epsilon$-PF', '$\epsilon$-RS', 'PI', 'EI', 'LCB']

# method_names = ['MOEE_UCB_topsis', 'MOEE_UCB_PF', 'MOEE_UCB_exploit', 'MOEE_UCB_random', 'MOEE_UCB_switch']
# method_names_for_table = ['MOEE', 'MOEE$_{PF}$', 'MOEE$_{exploit}$', 'MOEE$_{e\_PF}$', 'MOEE$_{switch}$']


method_names = ['MOEE_UCB_topsis', 'PI', 'EI_LBFGS', 'UCB_LBFGS', 'AWEI', 'MGFI']
method_names_for_table = ['MOEE', 'PI$_{LBFGS}$', 'EI$_{LBFGS}$', 'UCB$_{LBFGS}$', 'SAWEI', 'MGFI']

# method_names = ['MOEE_EI_topsis', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'PI_nsga2', 'EI', 'UCB', 'PI', 'EI_LBFGS', 'UCB_LBFGS']
# method_names_for_table = ['MOEE$_{EI}$', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'PI', 'EI', 'UCB', 'PI$_{LBFGS}$', 'EI$_{LBFGS}$', 'UCB$_{LBFGS}$']


results = process_results(results_dir, problem_names, method_names, budget=250, exp_no_start=1, exp_no_end=30)

with np.load(r'../training_data/push8_best_solutions.npz') as data:
    push8_estimated_optima = data['results']

for method_name in method_names:
    dist = results['push8'][method_name] - push8_estimated_optima[:, None][:30]

    # simple sanity checking - check the distance between them is >= 0, meaning that
    # the estimated optima are better or equal to the evaluated function values

    assert np.all(dist >= 0)

    results['push8'][method_name] = dist

# process = "median" or "mean"
table_data = create_table_data_MOEE(results, problem_names, method_names, 30, process="median")

create_table_MOEE(table_data, problem_rows, problem_paper_rows,
                  problem_dim_rows, method_names, method_names_for_table, process="median")
