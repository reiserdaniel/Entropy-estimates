import random
import numpy as np


def new_time_data(attractor_type, num_dimension):
    if attractor_type is "uniform":
        return [random.random() for r in range(num_dimension)]

    elif attractor_type is "gaussian":
        return [random.gauss(0, 1) for r in range(num_dimension)]

    elif attractor_type is "correlated_gaussian":
        base = random.gauss(0, 1)
        covariance = 0.7
        return [covariance * base + (1-covariance) * random.gauss(0, 1) for r in range(num_dimension)]

    elif attractor_type is "simple_cube":
        return generate_cube(num_dimension)

    elif attractor_type is "gaussian_cube":
        return generate_gaussian_cube(num_dimension, 0.05)

    else:
        return [r for r in range(num_dimension)]

def true_entopy_value(attractor_type, num_dimension):
    if attractor_type is "uniform":
        return np.log(1.)

    elif attractor_type is "gaussian":
        return np.log(np.sqrt(2 * np.e * np.pi)) * num_dimension

    elif attractor_type is "correlated_gaussian":
        # I'm really not sure about this result, needs more mathematical confirmation
        return num_dimension \
               * np.log(0.3 * np.sqrt(2. * np.pi * np.e)) \
               + np.log(0.7 * np.sqrt(2. * np.pi * np.e))




def generate_cube(num_dimension):
    value = [1-2*random.randrange(2) for r in range(num_dimension)]
    normal_vector_direction = int(random.random()*num_dimension)

    for i in range(num_dimension):
        if i is not normal_vector_direction:
            value[i] = random.random()*2-1

    return value, normal_vector_direction


def generate_gaussian_cube(num_dimension, sigma):
    vector, normal_vector_direction = generate_cube(num_dimension)
    vector = [vector[r] + random.gauss(0, sigma) for r in range(num_dimension)]
    return vector, normal_vector_direction
