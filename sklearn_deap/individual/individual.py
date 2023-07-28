import numpy as np

from typing import List, Type, Dict, Tuple
from sklearn_deap.types import param_types


def init_individual(pcls: Type, maxints: List[int]) -> List[int]:
    part = pcls(np.random.randint(low=0, high=maxint + 1) for maxint in maxints)
    return part


def mut_individual(individual, up: List[int], indpb: float) -> Tuple[List[int]]:
    individual[:] = iterate_mut(np.array(individual), up, indpb)
    return (individual,)


def iterate_mut(individual: np.ndarray, up: List[int], indpb: float) -> np.ndarray:
    for i, u, rn in zip(range(len(up)), up, (np.random.rand() for _ in range(len(up)))):
        if rn < indpb:
            individual[i] = np.random.randint(low=0, high=u + 1)
    return individual


def cx_individuals(ind1: List[int], ind2: List[int], indpb: float, gene_type: List[int]) -> Tuple[List[int]]:
    ind1[:], ind2[:] = iterate_cx(list(ind1), list(ind2), indpb, gene_type)
    return ind1, ind2


def iterate_cx(ind1_array: np.ndarray, ind2_array: np.ndarray, indpb: float, gene_type: List[int]) -> Tuple[np.ndarray]:
    for i, gt, rn in zip(
        range(len(ind1_array)), gene_type, (np.random.rand() for _ in range(len(ind1_array)))
    ):
        if rn > indpb:
            continue
        if gt is param_types.Categorical:
            ind1_array[i], ind2_array[i] = ind2_array[i], ind1_array[i]
        else:
            # Case when parameters are numerical
            if ind1_array[i] <= ind2_array[i]:
                ind1_array[i] = np.random.randint(low=ind1_array[i], high=ind2_array[i] + 1)
                ind2_array[i] = np.random.randint(low=ind1_array[i], high=ind2_array[i] + 1)
            else:
                ind1_array[i] = np.random.randint(low=ind2_array[i], high=ind1_array[i] + 1)
                ind2_array[i] = np.random.randint(low=ind2_array[i], high=ind1_array[i] + 1)

    return ind1_array, ind2_array


def individual_to_params(individual, name_values: List[Tuple]) -> Dict[str, float]:    
    params_dict = dict(
        (name, values[gene]) for gene, (name, values) in zip(individual, name_values)
    )
    return params_dict