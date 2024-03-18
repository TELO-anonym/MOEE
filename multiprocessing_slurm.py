from moee.optimizer import perform_experiment
from multiprocessing import Pool, cpu_count
import sys

if len(sys.argv) != 3:
    print("Usage ", sys.argv[0], " <p> <N>")
    sys.exit()
else:
    p = int(sys.argv[1])
    N = int(sys.argv[2])


def MOEE(problem_name, acf, type2, run_no):
    # 'topsis', 'random', 'exploit', 'PF', 'switch'
    perform_experiment(
        problem_name,  # problem name
        run_no,  # problem instance (LHS samples and optional args)
        "MOEE",  # method name
        acquisition_args={"type": acf, "type2": type2, "explore": 0},  # acq func args
        budget=250,
        continue_runs=True,  # resume runs
        verbose=True,  # print status
        save=True,  # whether to save the run
    )


def basic(problem_name, acf, run_no):
    # "EI",
    # "PI",
    # "UCB",
    # "EI_LBFGS",
    # "PI_nsga2",
    # "UCB_LBFGS"
    perform_experiment(
        problem_name,  # problem name
        run_no,  # problem instance (LHS samples and optional args)
        acf,  # method name
        verbose=True,  # print status
        continue_runs=True,  # resume runs
        save=True  # whether to save the run
    )


def greed(problem_name, acf, run_no):
    #   "eRandom",
    #   "eFront",
    perform_experiment(
        problem_name,  # problem name
        run_no,  # problem instance (LHS samples and optional args)
        acf,  # method name
        verbose=True,  # print status
        continue_runs=True,  # resume runs
        save=True,  # whether to save the run
        acquisition_args={"epsilon": 0.1},  # acq func args
    )


def AWEI(problem_name, acf, run_no):
    perform_experiment(
        problem_name,  # problem name
        run_no,  # problem instance (LHS samples and optional args)
        acf,  # method name
        verbose=True,  # print status
        continue_runs=True,  # resume runs
        save=True,  # whether to save the run
        acquisition_args={"ubr": [], "alpha": 0.5, "wei_pi_pure_term": 0, "wei_ei_term": 0},  # acq func args
    )


def MGFI(problem_name, acf, run_no):
    perform_experiment(
        problem_name,  # problem name
        run_no,  # problem instance (LHS samples and optional args)
        acf,  # method name
        verbose=True,  # print status
        continue_runs=True,  # resume runs
        save=True,  # whether to save the run
        acquisition_args={'t': 2},  # acq func args
    )


if __name__ == '__main__':
    problem_names = ['WangFreitas', 'Branin', 'BraninForrester', 'GoldsteinPrice',
                     'Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel',
                     'Hartmann6', 'GSobol', 'ACKLEY_10', 'GRIEWANK_10',
                     'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']

    method = "MOEE"
    type = 'UCB'  # 'UCB', 'EI'
    type2 = "topsis"  # 'topsis', 'PF', 'exploit', 'random', 'switch'
    inputs = [('WangFreitas', type, type2, int(i)) for i in range(1, N + 1)]
    for problem in problem_names[1:]:
        inputs += [(problem, type, type2, int(i)) for i in range(1, N + 1)]

    # 'eFront', 'eRandom'
    # method = 'eFront'
    # inputs = [('WangFreitas', method, int(i)) for i in range(1, N + 1)]
    # for problem in problem_names[1:]:
    #     inputs += [(problem, method, int(i)) for i in range(1, N + 1)]

    # 'PI_nsga2', 'EI', 'LCB', 'PI', 'EI_LBFGS', 'UCB_LBFGS'
    # method = 'PI_nsga2'
    # inputs = [('WangFreitas', method, int(i)) for i in range(1, N + 1)]
    # for problem in problem_names[1:]:
    #     inputs += [(problem, method, int(i)) for i in range(1, N + 1)]

    # 'AWEI'
    # method = 'AWEI'
    # inputs = [('WangFreitas', method, int(i)) for i in range(1, N + 1)]
    # for problem in problem_names[1:]:
    #     inputs += [(problem, method, int(i)) for i in range(1, N + 1)]

    # 'MGFI'
    # method = 'MGFI'
    # inputs = [('WangFreitas', method, int(i)) for i in range(1, N + 1)]
    # for problem in problem_names[1:]:
    #     inputs += [(problem, method, int(i)) for i in range(1, N + 1)]

    # Evaluate f for all inputs using a pool of processes
    with Pool(p) as my_pool:
        if method == 'MOEE':
            print(my_pool.starmap(MOEE, inputs))
        elif method in ['eFront', 'eRandom']:
            print(my_pool.starmap(greed, inputs))
        elif method in ['PI_nsga2', 'EI', 'LCB', 'PI', 'EI_LBFGS', 'UCB_LBFGS']:
            print(my_pool.starmap(basic, inputs))
        elif method == 'MGFI':
            print(my_pool.starmap(MGFI, inputs))
        elif method == 'AWEI':
            print(my_pool.starmap(AWEI, inputs))
