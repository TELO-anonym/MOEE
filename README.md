# Multi-Objectivising Acquisition Functions in Bayesian Optimisation

The repository contains all training data used for the initialisation of
each of the 30 optimisation runs carried to evaluate each method and
the code to generate new training data and also to perform the optimisation
runs themselves. 

## Training data

The initial training locations for each of the 30 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in
the `training_data` directory in this repository with the filename structure
`ProblemName_number`, e.g. the first set of training locations for the Branin
problem is stored in `Branin_1.npz`. Each of these files is a compressed numpy
file created with [numpy.savez](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html).
It has two [numpy.ndarrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
containing the 2*D initial locations and their corresponding fitness values.
To load and inspect these values use the following instructions:

```python
> cd /egreedy
> python
>>> import numpy as np
>>> with np.load('training_data/Branin_1.npz') as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']
>>> Xtr.shape, Ytr.shape
((4, 2), (4, 1))
```

The robot pushing test problems (push4 and push8) have a third array
`'arr_2'`  that contains their instance-specific parameters:

```python
> cd /egreedy
> python
>>> import numpy as np
>>> with np.load('training_data/push4_1.npz', allow_pickle=True) as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']
        instance_params = data['arr_2']
>>> instance_params
array({'t1_x': -4.268447250704135, 't1_y': -0.6937799887556437}, dtype=object)
```

these are automatically passed to the problem function when it is instantiated
to create a specific problem instance.

## Reproduction of experiments

The python file `optimizer.py` provides a convenient way to reproduce all 
experimental evaluations carried out the paper. 

```bash
> python optimizer.py
```

If you prefer to run experiments in parallel, the file `job_bo.sh` provides a convenient way to conduct that.

```
>sbatch job_bo.sh
```

## Reproduction of figures and tables in the paper

The python files `plot_convergence.py`, `plot_boxplots.py`, and `tables.py` contain the code to load and process the optimisation results (stored in the
`results` directory) as well as the code to produce all results figures and tables used in the paper.

## Acknowledgement

This repository has been developed based on the work from [egreedy](https://github.com/georgedeath/egreedy) [De Ath et al., 2021]. I would like to express my gratitude to the original authors and contributors for their pioneering efforts and for making their code available. Their work has been instrumental in the development of this project.

[De Ath et al., 2021] George De Ath, Richard M. Everson, Alma A. M. Rahat, and Jonathan E. Fieldsend. 2021. Greed Is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation. ACM Trans. Evol. Learn. Optim. 1, 1, Article 1 (May 2021), 22 pages.
