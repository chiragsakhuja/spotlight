import numpy as np

import space

CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.05   # mutation probability
PARENT_P = 0.2

def crossover_and_mutate(father, pop, pop_f):
    offspring = dict()
    dna_length = len(father.param_values)

    # Make direct copy of father
    for i in range(dna_length):
        offspring[father.param_labels[i]] = father.param_values[i]

    # With random probability, crossover with a promising mother
    if np.random.rand() < CROSS_RATE:
        pop_f_sort_idx = np.argsort(pop_f)
        p = len(pop)
        while p >= len(pop):
            p = np.random.geometric(PARENT_P) - 1
        mother = pop[pop_f_sort_idx[p]]
        cross_points = np.random.choice([True, False], size=dna_length)
        for i, cross_point in enumerate(cross_points):
            label = father.param_labels[i]
            offspring[label] = mother.param_values[i] if cross_point else father.param_values[i]

    # With random probability, mutate each gene with a random mother
    for i in range(dna_length):
        if np.random.rand() < MUTATION_RATE:
            mother = pop[np.random.randint(len(pop))]
            offspring[mother.param_labels[i]] = mother.param_values[i]

    return space.Point(offspring)

def generate_batch(space, batch_size, last_gen, last_gen_f):
    points = list()
    if last_gen:
        member_idx = len(last_gen)
        while len(points) < batch_size:
            if member_idx >= len(last_gen):
                member_idx = 0
                member_order = np.arange(len(last_gen))
                np.random.shuffle(member_order)
            member = last_gen[member_order[member_idx]]
            points.append(crossover_and_mutate(member, last_gen, last_gen_f))
    else:
        for _ in range(batch_size):
            space_idx = np.random.randint(space.size)
            sw_point = space.build_point(space_idx)
            points.append(sw_point)
    return points