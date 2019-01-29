import numpy as np
import random
import logging


# Load item list
FILENAME = 'items.txt'
LOGFILE = 'knapsack_logs.txt'
LOGGING_ENABLED = True
ITEMS = np.loadtxt(FILENAME)
N_ITEMS = len(ITEMS)
PENALTY_COEF = 3

# Genetic Algorithm parameters
MAX_WEIGHT = 15
POPULATION_SIZE = 20
# ratio of selection from one generation to the next
SELECTED_RATIO = 0.2
MAX_GENERATION = 20
MUTATION_FREQUENCY = 0.6


def output(message):
    if LOGGING_ENABLED:
        logging.info('\n' + message)
    
    print(message)


class Gene(object):
    """Defines the Gene object"""
    def __init__(self, value=-1):
        if value > -1:
            self.value = value
        else:
            if random.uniform(0, 1) > 0.5:
                self.value = 1
            else:
                self.value = 0

    def flip(self):
        self.value = not self.value


class Chromosome(object):
    """Defines the Chromosome object"""
    total_n_chromosomes = 0

    def __init__(self, generation, genome=None, origin="Genesis"):
        """Creates a chromosome by randomly removing values from the item list"""
        type(self).total_n_chromosomes += 1
        if genome is None:
            self.genome = [Gene() for _ in range(0, N_ITEMS)]
        else:
            self.genome = genome
        self.number = type(self).total_n_chromosomes
        self.generation = generation
        self.origin = origin
        self.n_mutation = 0

    def as_np_array(self):
        """Returns the genome as a numpy array (useful for calculations)"""
        return np.array([gene.value for gene in self.genome])

    def knapsack(self):
        """Returns the content of the knapsack as defined by the genome"""
        return np.resize(self.as_np_array(), (ITEMS.shape[1], N_ITEMS)).T * ITEMS

    def total_value(self):
        """Returns the value of the chromosome"""
        return int(np.sum(self.knapsack()[:, 0]))

    def total_weight(self):
        """Returns the total weight of the chromosome"""
        return int(np.sum(self.knapsack()[:, 1]))

    def fitness(self):
        """Returns the fitness of the chromosome"""
        return self.total_value() if self.total_weight() <= MAX_WEIGHT else 0

    def one_point_crossover(self, parent_2, generation):
        """Performs a one point crossover mutation starting at a random position"""
        x_over_point = random.randint(1, len(self.genome)-1)
        new_genome = []

        for idx in range(0, len(self.genome)):
            if idx < x_over_point:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        return Chromosome(generation, new_genome, "child of chr (%d, %d, %s)" %
                          (self.number, parent_2.number, 'one point crossover @%d' % x_over_point))

    def two_point_crossover(self, parent_2, generation):
        """Performs a two point crossover mutation at a random starting and ending position"""
        x_over_point_1 = random.randint(1, len(self.genome)-2)
        x_over_point_2 = random.randint(x_over_point_1+1, len(self.genome) - 1)
        new_genome = []

        for idx in range(0, len(self.genome)):
            if idx < x_over_point_1 or idx >= x_over_point_2:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        return Chromosome(generation, new_genome, "child of chr (%d, %d, %s)" %
                          (self.number, parent_2.number,
                           'two points crossover @(%s, %s)' % (x_over_point_1, x_over_point_2)))

    def uniform_crossover(self, parent_2, generation):
        """Performs a uniform crossover, i.e. randomly selecting genes from parent 1 or 2"""
        new_genome = []

        for idx in range(0, len(self.genome)):
            if random.uniform(0, 1) > 0.5:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        return Chromosome(generation, new_genome, "child of chr (%d, %d, %s)" %
                          (self.number, parent_2.number,
                           'uniform_crossover'))

    def try_mutation(self):
        """Decides whether one or multiple mutations are adequate according to the frequency and flips the gene"""
        while random.uniform(0, 1) < MUTATION_FREQUENCY:
            mutation_point = random.randint(0, len(self.genome)-1)
            self.genome[mutation_point].flip()
            self.n_mutation += 1

    def __repr__(self):
        """Returns a string representation """
        return 'Chromosome %4s: %s   Value: %2s   Weight: %2s   Fitness: %2s   From generation: %3s   Origin: %s %s' % \
               (str(self.number), self.as_np_array(), str(self.total_value()), str(self.total_weight()),
                str(self.fitness()), str(self.generation), self.origin,
                'n Mutations: %s' % self.n_mutation if self.n_mutation > 0 else '')


class Population(object):
    """Defines the Population object"""
    def __init__(self):
        self.generation = 1
        self.chromosomes = [Chromosome(self.generation) for _ in range(0, POPULATION_SIZE)]
        self.n_selection = int(SELECTED_RATIO * POPULATION_SIZE)
        output("Initial population (generation %d)\n%s\n" % (self.generation, self))

    def next_generation(self):
        """Moves on to the next generation by selecting the winners, performing the crossovers and mutating random genes"""
        self.selection()
        self.generation += 1
        self.crossover()
        self.mutations()

    def selection(self):
        """Selects the top chromosomes according to the ratio of selection"""
        self.chromosomes.sort(key=lambda x: x.fitness(), reverse=True)
        self.chromosomes = self.chromosomes[:self.n_selection]
        output("Selection at the end of generation %d\n%s\n" % (self.generation, self))

    def crossover(self):
        """Generates new children via crossover of the genomes. 3 types of crossover are used randomly"""
        for i in range(self.n_selection, POPULATION_SIZE):
            parent_1_id = random.randint(0, self.n_selection-1)
            parent_2_id = random.randint(0, self.n_selection-1)
            if parent_2_id == parent_1_id:
                parent_2_id = (parent_2_id + 1) % self.n_selection

            parent_1 = self.chromosomes[parent_1_id]
            parent_2 = self.chromosomes[parent_2_id]

            crossover_type = random.randint(0, 3)

            if crossover_type == 0:
                self.chromosomes.append(parent_1.one_point_crossover(parent_2, self.generation))
            elif crossover_type == 1:
                self.chromosomes.append(parent_1.two_point_crossover(parent_2, self.generation))
            elif crossover_type == 2:
                self.chromosomes.append(parent_1.uniform_crossover(parent_2, self.generation))

    def best_of_population(self):
        """Returns the best chromosome for a given generation"""
        best_chr = self.chromosomes[0]
        best_chr_fitness = best_chr.fitness()
        for chr in self.chromosomes[1:]:
            fitness = chr.fitness()
            if fitness > best_chr_fitness:
                best_chr = chr
                best_chr_fitness = fitness

        return best_chr

    def mutations(self):
        """Tries mutations for each chromosome, based on a random factor"""
        for idx in range(0, len(self.chromosomes)):
            self.chromosomes[idx].try_mutation()


    def __repr__(self):
        """String representation of the population"""
        return '\n'.join([str(chromosome) for chromosome in self.chromosomes])


def main():
    """Main function"""
    output('Available items:\n%s\n' % ITEMS)

    # Init population
    pop = Population()

    # Initialize the population
    best_fitness = 0
    best_knapsack = None
    while pop.generation < MAX_GENERATION:
        pop.next_generation()
        output("Genration %d\n%s\n" % (pop.generation, pop))

        # Identify the winner
        best_of_gen = pop.best_of_population()
        if best_of_gen.fitness() > best_fitness:
            best_fitness = best_of_gen.fitness()
            winner = "Best Chromosome:\n%s\n* %s *\n%s\n" % ('*' * (4+len(str(best_of_gen))), best_of_gen,
                                                                 '*' * (4+len(str(best_of_gen))))
            best_knapsack = best_of_gen.knapsack()
            best_knapsack = best_knapsack[np.all(best_knapsack != 0, axis=1)]

    output(winner)
    output('Best Knapsack:\n%s\n%s\n%s' % ('*' * 14, best_knapsack, '*' * 14))


def run_tests():
    """Tests the integrity of various objects"""
    test_chr = Chromosome(0, [Gene(1) for _ in range(0, N_ITEMS)])
    assert (test_chr.total_value() == np.sum(ITEMS[:, 0]))
    assert (test_chr.total_weight() == np.sum(ITEMS[:, 1]))

    test_chr.genome[0].flip()
    assert (test_chr.genome[0].value == 0)


if __name__ == '__main__':
    # First Activate Logging
    if LOGGING_ENABLED:
        logging.basicConfig(filename=LOGFILE,level=logging.INFO)

    # Perform tests
    try:
        run_tests()
    except Exception as e:
        logging.error("ERROR Running Tets: %s" % e)
    else:
        output("TESTING OK.")

    # Run Main Program
    main()
