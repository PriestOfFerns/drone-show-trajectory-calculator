import numpy as np
import drone
import random
import wandb

from typing import List, Tuple
from drone import DronePathGenome
from time import perf_counter


class GeneticAlgorithm:
    def __init__(self, targets: List[np.ndarray], start_bounding: Tuple[np.ndarray, np.ndarray],
                 position_bounding: Tuple[float, float], mutation_gauss: Tuple[float, float],
                 amount_gauss: Tuple[float, float], position_gauss: Tuple[float, float],
                 fitness_coefficients: Tuple[float, float, float, float], logging_parameters: Tuple[bool, bool, bool],
                 genome_count: int = 10000, breeding_pairs: int = 1000, hall_of_fame_size: int = 10,
                 mutation_chance: float = 0.01, mutation_change_chance: float = 0.9,
                 mutation_add_chance: float = 0.1):
        self.targets: List[np.ndarray] = targets
        self.start_bounding: Tuple[np.ndarray, np.ndarray] = start_bounding
        self.position_bounding: Tuple[
            float, float] = position_bounding  # This parameter seems unused in the current code
        self.mutation_gauss: Tuple[float, float] = mutation_gauss
        self.amount_gauss: Tuple[float, float] = amount_gauss
        self.position_gauss: Tuple[float, float] = position_gauss
        self.fitness_coefficients: Tuple[float, float, float, float] = fitness_coefficients
        self.logging_parameters: Tuple[bool, bool, bool] = logging_parameters
        self.genome_count: int = genome_count
        self.breeding_pairs: int = breeding_pairs
        self.mutation_chance = mutation_chance
        self.mutation_change_chance = mutation_change_chance
        self.mutation_add_chance = mutation_add_chance

        self.droneGenomes: np.ndarray = None
        self.generate_random_genomes()

        self.hall_of_fame: List[Tuple[float, DronePathGenome]] = []
        self.hall_of_fame_size: int = hall_of_fame_size
        self.last_fitness: float = None

        self.wandb_log = {}

    def generate_random_genomes(self):
        """Generates an initial population of random DronePathGenomes."""
        genomes = [drone.random_drone_path_genome(self.start_bounding[0], self.start_bounding[1], self.targets,
                                                  self.amount_gauss, self.position_gauss, self.mutation_chance,
                                                  self.mutation_gauss)
                   for _ in range(self.genome_count)]
        self.droneGenomes = np.array(genomes)

    def cross_genomes(self, genomes: List[DronePathGenome]) -> List[DronePathGenome]:
        """
        Performs crossover operations on a list of genomes to create new ones.

        Args:
            genomes: A list of parent genomes selected for breeding.

        Returns:
            A new list of genomes generated through crossover.
        """
        new_genomes: List[DronePathGenome] = []
        # Ensure there's an even number of genomes for pairing, or handle the last one if odd
        if len(genomes) % 2 != 0:
            # Optionally handle the odd one out, e.g., by duplicating a random one or discarding
            # For simplicity, we'll just proceed, meaning the last one might not be paired.
            # Or, if we ensure `most_fit` in `cycle` is always even, this won't be an issue.
            pass

        # Take `breeding_pairs` number of pairs to create new genomes
        # Each pair will produce `self.genome_count // self.breeding_pairs` new genomes
        for i in range(self.breeding_pairs):
            if 2 * i + 1 < len(genomes):  # Ensure there are enough genomes for a pair
                genomeA = genomes[2 * i]
                genomeB = genomes[2 * i + 1]
                for _ in range(self.genome_count // self.breeding_pairs):
                    new_genomes.append(genomeA.cross(genomeB))
            else:
                # If there aren't enough pairs to fulfill `self.breeding_pairs`, break
                break
        return new_genomes

    def calculate_fitness(self) -> List[Tuple[float, DronePathGenome]]:
        """
        Calculates the fitness for all genomes in the current population.

        Returns:
            A list of tuples, where each tuple contains (fitness_score, genome_object).
        """
        fitness_genome_pairs: List[Tuple[float, DronePathGenome]] = []
        for genome in self.droneGenomes:
            # Pass the required coefficients to the fitness method
            fitness = genome.fitness(*self.fitness_coefficients)
            fitness_genome_pairs.append((fitness, genome))
        return fitness_genome_pairs

    @staticmethod
    def average_fitness(fitness_genome_pairs: List[Tuple[float, DronePathGenome]]) -> float:
        """
        Calculates the average fitness of a list of fitness-genome pairs.

        Args:
            fitness_genome_pairs: A list of (fitness_score, genome_object) tuples.

        Returns:
            The average fitness score.
        """
        if not fitness_genome_pairs:
            return 0.0
        return sum(pair[0] for pair in fitness_genome_pairs) / len(fitness_genome_pairs)

    def calculate_most_fit(self, fitness_genome_pairs: List[Tuple[float, DronePathGenome]],
                           n: int) -> List[DronePathGenome]:
        """
        Selects the 'n' most fit genomes from a list of fitness-genome pairs.

        Args:
            fitness_genome_pairs: A list of (fitness_score, genome_object) tuples.
            n: The number of most fit genomes to select.

        Returns:
            A list of the 'n' most fit DronePathGenome objects.
        """
        # Sort pairs by fitness in descending order (higher fitness is better)
        sorted_fitness_genome_pairs = sorted(fitness_genome_pairs + self.hall_of_fame, key=lambda item: item[0],
                                             reverse=True)
        # Select the top n genomes
        most_fit_genomes = [pair[1] for pair in sorted_fitness_genome_pairs[:n]]
        return most_fit_genomes

    def add_to_hall_of_fame(self, fitness_genome_pairs: List[Tuple[float, DronePathGenome]]):
        """
        Adds fit genomes to the hall of fame, maintaining a sorted list of the best genomes found so far.

        Args:
            fitness_genome_pairs: A list of (fitness_score, genome_object) tuples.
        """
        for new_pair in fitness_genome_pairs:

            if new_pair in self.hall_of_fame:
                continue

            added = False
            for i, existing_pair in enumerate(self.hall_of_fame):

                if new_pair[0] > existing_pair[0]:
                    self.hall_of_fame.insert(i, new_pair)
                    added = True
                    break
            if not added and len(self.hall_of_fame) < self.hall_of_fame_size:
                self.hall_of_fame.append(new_pair)
            # Keep only the top `hall_of_fame_size` elements
            self.hall_of_fame = self.hall_of_fame[:self.hall_of_fame_size]

    def cycle(self):
        """Performs one generation (cycle) of the genetic algorithm."""
        fitness_pairs = self.calculate_fitness()

        # Get average fitness for logging purposes
        average_fitness = self.average_fitness(fitness_pairs)
        self.wandb_log["Average Fitness"] = average_fitness

        if self.logging_parameters[0]:
            print(f"Average Fitness: {average_fitness}")  # Added for console output

        self.add_to_hall_of_fame(fitness_pairs)

        # Select the most fit genomes for breeding
        # Ensure an even number for breeding pairs for simpler pairing in cross_genomes
        num_to_select = 2 * self.breeding_pairs
        most_fit = self.calculate_most_fit(fitness_pairs, num_to_select)
        random.shuffle(most_fit)  # Shuffle to introduce randomness in breeding pairs

        self.droneGenomes = np.array(self.cross_genomes(most_fit))

    def cycle_log(self, epoch_time):

        if self.logging_parameters[0]:
            print(f"Genome amount: {len(self.droneGenomes)}")
            print(f"Time Per Epoch: {epoch_time}")

        self.wandb_log["Time Per Epoch"] = epoch_time

        if self.logging_parameters[2]:
            waypoint_count = sum(
                (sum((len(_drone.waypoints) for _drone in _genome.drones)) for _genome in self.droneGenomes))
            if self.logging_parameters[0]:
                print(f"Waypoint amount: {waypoint_count}")
            if self.logging_parameters[1]:
                self.wandb_log["Waypoint Amount"] = waypoint_count

        if self.logging_parameters[1]:
            wandb.log(self.wandb_log)
            self.wandb_log.clear()

    def run(self, epochs: int):
        """
        Runs the genetic algorithm for a specified number of epochs.

        Args:
            epochs: The number of generations to run the algorithm.

        Returns:
            The hall of fame (best genomes found) after all epochs.
        """
        last_time = perf_counter()
        for epoch in range(epochs):

            if self.logging_parameters[0]:
                print(f"Epoch {epoch + 1}/{epochs}")  # Added for console output

            self.cycle()
            self.cycle_log(perf_counter() - last_time)

            last_time = perf_counter()
        return self.hall_of_fame


if __name__ == "__main__":
    wandb.login()

    # Initialize wandb run
    run = wandb.init(
        project="drone-path-optimization",  # Specify your project name
        config={
            "genome_count": 1000,
            "breeding_pairs": 10,
            "epochs": 50,
            "hall_of_fame_size": 5,
            "distance_coefficient": 1.0,
            "waypoint_coefficient": 1.5,
            "drone_collision_coefficient": 1000.0,
            "target_collision_coefficient": 0.0,
            "mutation_gauss_mean": 5,
            "mutation_gauss_std": 2,
            "amount_gauss_mean": 5,
            "amount_gauss_std": 2,
            "position_gauss_mean": 5,
            "position_gauss_std": 5,
        }
    )

    # Define parameters for the GeneticAlgorithm
    bounding_start = np.array([0.0, 0.0, 0.0], dtype=float)
    bounding_end = np.array([50.0, 50.0, 50.0], dtype=float)
    targets = [np.array([10.0, 10.0, 10.0], dtype=float),
               np.array([25.0, 25.0, 25.0], dtype=float),
               np.array([40.0, 40.0, 40.0], dtype=float)]

    mutation_gauss = (run.config.mutation_gauss_mean, run.config.mutation_gauss_std)
    amount_gauss = (run.config.amount_gauss_mean, run.config.amount_gauss_std)
    position_gauss = (run.config.position_gauss_mean, run.config.position_gauss_std)

    fitness_coefficients = (
        run.config.distance_coefficient,
        run.config.waypoint_coefficient,
        run.config.drone_collision_coefficient,
        run.config.target_collision_coefficient
    )

    # Create an instance of the GeneticAlgorithm
    ga = GeneticAlgorithm(
        targets=targets,
        start_bounding=(bounding_start, bounding_end),
        position_bounding=(0.0, 0.0),
        # This seems unused in the current `random_drone_path_genome` and `GeneticAlgorithm`
        mutation_gauss=mutation_gauss,
        amount_gauss=amount_gauss,
        position_gauss=position_gauss,
        fitness_coefficients=fitness_coefficients,
        genome_count=run.config.genome_count,
        breeding_pairs=run.config.breeding_pairs,
        hall_of_fame_size=run.config.hall_of_fame_size,
        logging_parameters=(True, True, True)
    )

    # Run the genetic algorithm
    print("\nStarting Genetic Algorithm run...")
    hall_of_fame_results = ga.run(epochs=run.config.epochs)

    print("\nGenetic Algorithm finished.")
    print("\nHall of Fame:")
    for i, (fitness, genome) in enumerate(hall_of_fame_results):
        print(f"Rank {i + 1}: Fitness = {fitness:.2f}")
        # Optionally print more details about the best drone path, e.g.,
        # print(f"  Genome: {genome}")
        # For a more readable output, you might want to iterate through drone.waypoints:
        for j, drone_obj in enumerate(genome.drones):
            print(
                f"    Drone {j + 1}: Start: {drone_obj.start}, End: {drone_obj.end}, Waypoints: {len(drone_obj.waypoints)}")
            print(f"      Path Length: {drone_obj.distance():.2f}")
            print(f"      Collisions with others: {drone_obj.get_intersecting_count(genome.drones)}")

    # Finish the wandb run
    wandb.finish()
