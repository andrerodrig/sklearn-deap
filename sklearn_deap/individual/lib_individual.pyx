import numpy as np
cimport numpy as np

from sklearn_deap.types import param_types


cdef list get_iterate_mut(list individual, list up, double indpb):
    cdef int i, u
    cdef double rn

    for i, u, rn in zip(range(len(up)), up, (np.random.rand() for _ in range(len(up)))):
        if rn < indpb:
            individual[i] = np.random.randint(low=0, high=u + 1)
    return individual


cdef tuple get_iterate_cx(list ind1, list ind2, double indpb, list gene_type):
    cdef int i, gt
    cdef double rn

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
