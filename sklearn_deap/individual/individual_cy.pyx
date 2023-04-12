import numpy as np

from typing import List, Type, Dict, Tuple
from sklearn_deap.individual.lib_individual cimport get_iterate_cx, get_iterate_mut


def init_individual(pcls: Type, maxints: List[int]) -> List[int]:
    part = pcls(np.random.randint(low=0, high=maxint + 1) for maxint in maxints)
    return part


def mut_individual(individual, up: List[int], indpb: float) -> Tuple[List[int]]:
    individual[:] = get_iterate_mut(list(individual), up, indpb)
    return (individual,)


def cx_individuals(ind1: List[int], ind2: List[int], indpb: float, gene_type: List[int]) -> Tuple[list]:
    ind1[:], ind2[:] = get_iterate_cx(list(ind1), list(ind2), indpb, gene_type)
    return ind1, ind2


def individual_to_params(individual, name_values: List[Tuple]) -> Dict[str, float]:    
    params_dict = dict(
        (name, values[gene]) for gene, (name, values) in zip(individual, name_values)
    )
    return params_dict