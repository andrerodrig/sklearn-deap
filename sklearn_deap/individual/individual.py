import numpy as np

from sklearn_deap.types import param_types


def init_individual(pcls, maxints):
    part = pcls(np.random.randint(low=0, high=maxint + 1) for maxint in maxints)
    return part


def mut_individual(individual, up, indpb, gene_type=None):
    for i, up, rn in zip(range(len(up)), up, (np.random.rand() for _ in range(len(up)))):
        if rn < indpb:
            individual[i] = np.random.randint(low=0, high=up + 1)
    return (individual,)


def cx_individuals(ind1, ind2, indpb, gene_type):
    for i, gt, rn in zip(
        range(len(ind1)), gene_type, (np.random.rand() for _ in range(len(ind1)))
    ):
        if rn > indpb:
            continue
        if gt is param_types.Categorical:
            ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            # Case when parameters are numerical
            if ind1[i] <= ind2[i]:
                ind1[i] = np.random.randint(low=ind1[i], high=ind2[i] + 1)
                ind2[i] = np.random.randint(low=ind1[i], high=ind2[i] + 1)
            else:
                ind1[i] = np.random.randint(low=ind2[i], high=ind1[i] + 1)
                ind2[i] = np.random.randint(low=ind2[i], high=ind1[i] + 1)

    return ind1, ind2


def individual_to_params(individual, name_values):
    return dict(
        (name, values[gene]) for gene, (name, values) in zip(individual, name_values)
    )