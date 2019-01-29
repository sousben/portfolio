import numpy as np
import random
import logging
from time import monotonic
from concurrent.futures import ThreadPoolExecutor
# import multiprocessing
import matplotlib.pyplot as plt


# SUDOKU BOARD
FILENAME = 'medium2.txt'
INITIAL_SUDOKU = np.loadtxt(FILENAME)
START_BOARD = INITIAL_SUDOKU.ravel()
MISSING_INDICES = np.arange(81)[START_BOARD == 0]
N_ITEMS = len(MISSING_INDICES)

# LOGGING
LOGFILE = 'sudoku_logs.txt'
LOGGING_ENABLED = True

# Genetic Algorithm parameters
POPULATION_SIZE = 1000
MAX_GENERATION = 200000
MUTATION_FREQUENCY = 0.6
# Ratio of selection from one generation to the next
SELECTED_RATIO = 0.1

# Sanctuary max size
SANCTUARY_LIMIT = 0.4 * POPULATION_SIZE
# Freeze few chromosomes from [most] generations to bring them back to life as necessary
CRYO_PER_GEN = int(SANCTUARY_LIMIT/4)
CRYO_MAX_SIZE = 50000
cryo_dict = {}

DEBUG = False


# fitness function parameters
CORRECT_SEQUENCE = 10
# 9 rows, 9 columns, 9 - 3x3 square. 27 in total
MAX_SCORE = 27 * CORRECT_SEQUENCE


def output(message):
    """Output function - prints and saves to logfile if enabled"""
    if LOGGING_ENABLED:
        logging.info('\n' + message)
    
    print(message)


class Gene(object):
    """Defines the Gene object"""
    def __init__(self, value=-1):
        if value > -1:
            self.value = value
        else:
            self.value = random.randint(1, 9)

    def mutate(self):
        """Mutate the gene"""
        self.value = random.randint(1, 9)

    def __eq__(self, other):
        """True if both genes are equal"""
        return self.value == other.value


class Chromosome(object):
    """Defines the Chromosome object"""
    total_n_chromosomes = 0

    def __init__(self, generation, genome=None, in_sanctuary=0, origin="Genesis"):
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
        self.board = self.sudoku()
        self.fit_score = self.fitness()
        self.winner = False
        self.in_sanctuary = in_sanctuary

    def as_np_array(self):
        """Returns the genome as a numpy array (useful for calculations)"""
        return np.array([gene.value for gene in self.genome])

    def sudoku(self):
        """Returns the sudoku board as defined by the genome"""
        sudoku = INITIAL_SUDOKU.copy()
        for idx in MISSING_INDICES:
            sudoku[int(idx/9)][idx % 9] = self.genome[np.where(MISSING_INDICES == idx)[0][0]].value

        return sudoku

    def fitness(self):
        """Returns the fitness of the chromosome"""
        fitness_score = 0
        # Fill in the sudoku board with the values from the chromosome
        sudoku = self.board

        for i in range(9):
            # count unique values in rows
            row_unique = len(set(sudoku[i]))
            fitness_score += CORRECT_SEQUENCE if row_unique == 9 else row_unique
            # count unique values in rows
            col_unique = len(set(sudoku[:, i]))
            fitness_score += CORRECT_SEQUENCE if col_unique == 9 else col_unique

            # 3x3 zone count unique values
            zone_unique = len(set(sudoku[3*int(i/3):3*(1+int(i/3)), 3*(i % 3):3*(1+(i % 3))].ravel()))
            fitness_score += CORRECT_SEQUENCE if zone_unique == 9 else zone_unique

        return fitness_score

    def one_point_crossover(self, parent_2, population):
        """Performs a one point crossover mutation starting at a random position"""
        x_over_point = random.randint(1, len(self.genome)-1)
        new_genome = []

        for idx in range(0, len(self.genome)):
            if idx < x_over_point:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        in_sanctuary = max(self.in_sanctuary-1, parent_2.in_sanctuary-1, 0)

        return Chromosome(population.generation, new_genome, in_sanctuary, "child of chr (%d, %d, %s)" %
                           (self.number, parent_2.number, 'one point crossover @%d' % x_over_point))

    def two_point_crossover(self, parent_2, population):
        """Performs a two point crossover mutation at a random starting and ending position"""
        x_over_point_1 = random.randint(1, len(self.genome)-2)
        x_over_point_2 = random.randint(x_over_point_1+1, len(self.genome) - 1)
        new_genome = []

        for idx in range(0, len(self.genome)):
            if idx < x_over_point_1 or idx >= x_over_point_2:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        in_sanctuary = max(self.in_sanctuary-1, parent_2.in_sanctuary-1, 0)

        return Chromosome(population.generation, new_genome, in_sanctuary, "child of chr (%d, %d, %s)" %
                           (self.number, parent_2.number,
                            'two points crossover @(%s, %s)' % (x_over_point_1, x_over_point_2)))

    def uniform_crossover(self, parent_2, population):
        """Performs a uniform crossover, i.e. randomly selecting genes from parent 1 or 2"""
        new_genome = []

        for idx in range(0, len(self.genome)):
            if random.uniform(0, 1) > 0.5:
                new_genome.append(Gene(self.genome[idx].value))
            else:
                new_genome.append(Gene(parent_2.genome[idx].value))

        in_sanctuary = max(self.in_sanctuary-1, parent_2.in_sanctuary-1, 0)

        return Chromosome(population.generation, new_genome, in_sanctuary, "child of chr (%d, %d, %s)" %
                           (self.number, parent_2.number, 'uniform_crossover'))

    def try_mutation(self):
        """Decides whether one or multiple mutations are adequate according to the frequency and flips the gene"""
        if self.winner:
            return

        has_mutated = False
        while random.uniform(0, 1) < MUTATION_FREQUENCY:
            mutation_point = random.randint(0, len(self.genome)-1)
            self.genome[mutation_point].mutate()
            self.n_mutation += 1
            has_mutated = True

        if has_mutated:
            self.board = self.sudoku()
            self.fit_score = self.fitness()

    def __repr__(self):
        """Returns a string representation """
        return 'Chromosome %4s: %s   Fitness: %2s   From generation: %3s   Origin: %s %s %s' % \
               (str(self.number), self.as_np_array(), str(self.fit_score), str(self.generation), self.origin,
                'n Mutations: %s' % self.n_mutation if self.n_mutation > 0 else '',
                'in sanctuary %d' % self.in_sanctuary if self.in_sanctuary > 0 else '')

    def __eq__(self, other):
        """True if two chromosomes have same genome"""
        for idx, gene in enumerate(self.genome):
            if gene != other.genome[idx]:
                return False
        return True


class Population(object):
    """Defines the Population object"""
    def __init__(self):
        """Instances the Population object"""
        self.generation = 1
        self.chromosomes = [Chromosome(self.generation) for _ in range(0, POPULATION_SIZE)]
        self.size = POPULATION_SIZE
        self.n_selection = int(SELECTED_RATIO * POPULATION_SIZE)
        self.total_selection = self.n_selection
        self.sort_generation()
        self.best_score = 0
        self.stalling_generation = 0
        self.generation_lookback = 0
        self.sanctuary = []

    def set_size(self, size):
        """Resets the size of the population"""
        self.size = size
        self.n_selection = int(SELECTED_RATIO * POPULATION_SIZE)
        self.total_selection = self.n_selection

    def next_generation(self):
        """Moves on to the next generation by selecting the winners, performing the crossovers and mutations randomly"""

        previous_score = self.best_score
        self.selection()
        self.generation += 1
        self.crossover()
        self.mutations()
        self.sort_generation()
        if previous_score == self.best_score:
            self.stalling_generation += 1
        else:
            self.stalling_generation = 0

    def sort_generation(self):
        """Sorts the generation in ascending order of fit score. This order is useful when selecting winners of gen"""
        self.chromosomes.sort(key=lambda x: x.fit_score, reverse=True)
        self.best_score = self.chromosomes[0].fit_score
        self.chromosomes[0].winner = True
        for idx, chrom in enumerate(self.chromosomes):
            if chrom.fit_score < self.best_score:
                self.n_best_scores = idx
                break

    def selection(self):
        """Selects the top chromosomes according to the ratio of selection"""
        # Select the winners of the generation - avoid duplicates
        selected_chromosomes = []
        n_selected = 0
        for chrom in self.chromosomes:
            if chrom not in selected_chromosomes:
                selected_chromosomes.append(chrom)
                chrom.winner = True
                n_selected += 1
                if n_selected >= self.n_selection:
                    break

        # Add chromosomes randomly (higher chance for best of generation)
        # selected_chromosomes = [self.chromosomes[0]]
        # n_added = 1
        # while n_added < self.n_selection:
        #     for idx in range(1, len(self.chromosomes)):
        #         if random.randint(1, 5*POPULATION_SIZE) <= POPULATION_SIZE-idx:
        #             selected_chromosomes.append(self.chromosomes[idx])
        #             n_added += 1
        #             # print('adding idx %d' % idx)
        #             if n_added >= self.n_selection:
        #                 break

        # Freeze some chromosomes if the generation wasn't a stalled one
        if self.stalling_generation == 0:
            for _ in range(CRYO_PER_GEN):
                rnd_index = random.randint(self.n_selection, self.size-1)
                if self.generation not in cryo_dict:
                    cryo_dict[self.generation] = [self.chromosomes[rnd_index]]
                else:
                    cryo_dict[self.generation].append(self.chromosomes[rnd_index])

        old_len = len(self.sanctuary)
        # Add chromosomes from sanctuary
        for idx, chrom in reversed(list(enumerate(self.sanctuary))):
            selected_chromosomes.append(chrom)
            chrom.in_sanctuary -= 1
            if chrom.in_sanctuary <= 0:
                self.sanctuary.pop(idx)

        # Implement chromosome lookback to resuscitate frozen genomes
        if self.stalling_generation > 5 and len(self.sanctuary) <= SANCTUARY_LIMIT/2:
            in_sanctuary = 0
            while True:
                while True:
                    self.generation_lookback += 1
                    look_at_generation = max(self.generation - self.stalling_generation - self.generation_lookback, 1)
                    if look_at_generation in cryo_dict:
                        in_sanctuary = int(1.5 * len({k: v for k, v in filter(lambda t: t[0] > look_at_generation, cryo_dict.items())}))
                        break

                if look_at_generation > 1:
                    print('Adding frozen chromosomes from generation %d - in sanctuary for %d gens' %
                          (look_at_generation, in_sanctuary))
                else:
                    print('Adding random chromosomes! - in sanctuary for %d gens' % in_sanctuary)

                if look_at_generation == 1:
                    for _ in range(int(SANCTUARY_LIMIT/4)):
                        new_chrom = Chromosome(self.generation, in_sanctuary=in_sanctuary)
                        selected_chromosomes.append(new_chrom)
                        self.add_to_sanctuary(new_chrom)
                else:
                    for lookback_chromosome in cryo_dict[look_at_generation]:
                        lookback_chromosome.in_sanctuary = in_sanctuary
                        # int((self.generation - self.stalling_generation - look_at_generation) * 1.5)
                        selected_chromosomes.append(lookback_chromosome)
                        self.add_to_sanctuary(lookback_chromosome)

                if len(self.sanctuary) >= SANCTUARY_LIMIT/4:
                    break

        if self.stalling_generation == 0:
            self.generation_lookback = 0
            for chrom in self.sanctuary:
                chrom.in_sanctuary = 0
            self.sanctuary = []

        self.total_selection = len(selected_chromosomes)
        self.chromosomes = selected_chromosomes

    def crossover(self):
        """Generates new children via crossover of the genomes. 3 types of crossover are used randomly"""
        # try:
        #     # Create multiprocessing pool
        #     pool = multiprocessing.Pool(processes=4)
        #     children = pool.starmap(Population.one_random_crossover, [(self.chromosomes, self.total_selection, self)
        #                                                               for _ in range(self.total_selection, self.size)])
        #     pool.close()
        #     pool.join()
        #
        #     self.chromosomes.extend(children)
        #     if len(self.sanctuary) > 0:
        #         for chrom in children:
        #             if chrom.in_sanctuary > 0:
        #                 self.add_to_sanctuary(chrom)
        #
        # except Exception as e:
        #print('Error with multiprocessing... Running crossover by hand')

        children = []

        while len(children) < self.size - self.total_selection:
            child = self.one_random_crossover(self.chromosomes, self.total_selection, self)
            children.append(child)

        self.chromosomes.extend(children)
        if len(self.sanctuary) > 0:
            for chrom in children:
                if chrom.in_sanctuary > 0:
                    self.add_to_sanctuary(chrom)

    @staticmethod
    def one_random_crossover(chromosomes, total_selection, generation):
        """Performs a crossver randomly. The method has been made static to help with multithreading"""
        parent_1_id = random.randint(0, total_selection-1)
        parent_2_id = random.randint(0, total_selection-1)

        parent_1 = chromosomes[parent_1_id]
        parent_2 = chromosomes[parent_2_id]

        crossover_type = random.randint(0, 2)

        if crossover_type == 0:
            return parent_1.one_point_crossover(parent_2, generation)
        elif crossover_type == 1:
            return parent_1.two_point_crossover(parent_2, generation)
        elif crossover_type == 2:
            return parent_1.uniform_crossover(parent_2, generation)

    def add_to_sanctuary(self, chromosome):
        """Adds a chromosome to sanctuary if not full"""
        if len(self.sanctuary) < SANCTUARY_LIMIT:
            self.sanctuary.append(chromosome)
        else:
            chromosome.in_sanctuary = 0

    def best_of_population(self):
        """Returns the best chromosome for a given generation"""
        return self.chromosomes[0]

    def mutations(self):
        """Tries mutations for each chromosome, based on a random factor"""
        pool = ThreadPoolExecutor()
        for idx in range(0, len(self.chromosomes)):
            pool.submit(self.mutate_chromosome_at_index, idx)

    def mutate_chromosome_at_index(self, idx):
        """Tries a mutation on the chromosoe at the parameter index"""
        self.chromosomes[idx].try_mutation()

    def __repr__(self):
        """String representation of the population"""
        return '\n'.join([str(chromosome) for chromosome in self.chromosomes])


def main():
    """Main function"""
    # Init timer
    start = monotonic()

    # Init population
    pop = Population()

    # Initialize the population
    best_fitness = 0
    best_sudoku = None

    chart_data = np.zeros((MAX_GENERATION, 2))
    chart_data[pop.generation] = [monotonic() - start, pop.best_of_population().fit_score]

    # last_fitness = []

    while pop.generation < MAX_GENERATION and best_fitness < MAX_SCORE:
        generation = monotonic()
        pop.next_generation()

        # Identify the winner
        best_of_gen = pop.best_of_population()

        if best_of_gen.fit_score > best_fitness:
            best_fitness = best_of_gen.fit_score
            winner = "Best Chromosome:\n%s\n* %s *\n%s\n" % ('*' * (4+len(str(best_of_gen))), best_of_gen,
                                                             '*' * (4+len(str(best_of_gen))))
            best_sudoku = best_of_gen.board

        duration = monotonic() - generation
        output("Generation %d - best fitness so far - %d - best of gen - %d - Generation took %.3fsec - "
               "%d chromosomes have highest score" %
               (pop.generation, best_fitness, best_of_gen.fit_score, duration, pop.n_best_scores))

        #output("best of gen:\n%s" % best_of_gen.board)

        # last_fitness.append(best_of_gen.fit_score)
        # # Increase pop size if generation stalls
        # if len(last_fitness) >= 20:
        #     last_fitness.pop(0)
        #     if last_fitness[0] == best_of_gen.fit_score and pop.size < MAX_POP_SIZE:
        #         pop.set_size(pop.size * 2 if pop.size * 2 < MAX_POP_SIZE else MAX_POP_SIZE)
        #         print('Increasing population size to %d' % pop.size)
        #         last_fitness = []
        #
        # # Reset pop size
        # if len(last_fitness) > 0 and pop.size > POPULATION_SIZE and last_fitness[0] < best_of_gen.fit_score:
        #     print('Decreasing population size back to %d' % POPULATION_SIZE)
        #     pop.set_size(POPULATION_SIZE)

        chart_data[pop.generation] = [monotonic() - start, pop.best_of_population().fit_score]

    output(winner)
    output('Initial board:\n%s\n%s\n%s' % ('*' * 14, INITIAL_SUDOKU, '*' * 14))
    output('Best Board:\n%s\n%s\n%s' % ('*' * 14, best_sudoku, '*' * 14))

    chart_data = chart_data[chart_data[:, 0] != 0]

    plt.title("Resolved - '%s' Gen=%d P=%d SR=%.2f MG=%d MF=%.2f CS=%d" %
              (FILENAME, pop.generation, POPULATION_SIZE, SELECTED_RATIO, MAX_GENERATION,
               MUTATION_FREQUENCY, CORRECT_SEQUENCE))
    plt.margins(x=0)
    plt.plot([0, chart_data[:, 0].max()/60], [MAX_SCORE, MAX_SCORE], '--', c='green')
    plt.plot(chart_data[:, 0]/60, chart_data[:, 1])
    plt.xlabel('Execution Time (min)')
    plt.ylabel('Fit Score')
    plt.show()


def run_tests():
    """Tests the integrity of various objects"""
    test_chr = Chromosome(0, [Gene(1) for _ in range(0, N_ITEMS)])
    assert (test_chr.genome[0].value == 1)
    assert (test_chr.genome[1].value == 1)
    assert (test_chr.fitness() > 0)

    test_chr.genome[0].mutate()
    assert (test_chr.genome[0].value > 0)


if __name__ == '__main__':
    # First Activate Logging
    if LOGGING_ENABLED:
        logging.basicConfig(filename=LOGFILE, level=logging.INFO)

    # Perform tests
    try:
        run_tests()
        pass
    except Exception as e:
        logging.error("ERROR Running Tets: %s" % e)
    else:
        output("TESTING OK.")

    # Run Main Program
    main()

