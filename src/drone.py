from random import gauss, randrange, randint, random, choice
from typing import List, Tuple, Optional

from warnings import warn

import numpy as np
import capsule  # Assuming 'capsule' is a custom module for capsule intersection

# Define a type alias for position vectors
Position = np.ndarray


def random_pos_in_range(range_start: Position, range_end: Position, step: int = 1) -> Position:
    """
    Generates a random 3D position within a given range.

    Args:
        range_start: The starting point of the range (inclusive).
        range_end: The ending point of the range (exclusive).
        step: The step size for generating coordinates.

    Returns:
        A numpy array representing the random 3D position.
    """
    x = randrange(int(range_start[0]), int(range_end[0]), step)
    y = randrange(int(range_start[1]), int(range_end[1]), step)
    z = randrange(int(range_start[2]), int(range_end[2]), step)

    return np.array([x, y, z], dtype=float)  # Use float dtype for consistency


def random_pos_from_gauss(origin: Position, gauss_args: Tuple[float, float]) -> Position:
    """
    Generates a random 3D position based on a Gaussian distribution around an origin.

    Args:
        origin: The center point for the Gaussian distribution.
        gauss_args: A tuple containing the mean and standard deviation for the Gaussian distribution.

    Returns:
        A numpy array representing the random 3D position.
    """
    return origin + np.array((gauss(*gauss_args), gauss(*gauss_args), gauss(*gauss_args)), dtype=float)


class Drone:
    """
    Represents a drone with a path defined by waypoints.
    """

    def __init__(self, start_pos: Position, end_pos: Position, radius: float = 0.5,
                 mutation_gauss: Optional[Tuple[float, float]] = None, waypoints: Optional[List[Position]] = None):
        """
        Initializes a Drone object.

        Args:
            start_pos: The starting position of the drone.
            end_pos: The target ending position of the drone.
            radius: The radius of the drone (used for collision detection).
            mutation_gauss: A tuple (mean, std_dev) for Gaussian mutation of waypoints.
                            Defaults to (10, 5).
            waypoints: A list of intermediate waypoints for the drone's path.
                       Defaults to an empty list.
        """
        self.start: Position = start_pos
        self.end: Position = end_pos
        self.radius: float = radius
        self.mutation_gauss: Tuple[float, float] = mutation_gauss if mutation_gauss is not None else (10, 5)
        self.waypoints: List[Position] = waypoints if waypoints is not None else []

    def distance(self) -> float:
        """
        Calculates the total distance of the drone's path.

        Returns:
            The total distance as a float.
        """

        path_points = [self.start] + self.waypoints + [self.end]
        # Use numpy's diff and norm for potentially faster calculation
        total_distance = np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
        return total_distance

    def get_colliders(self) -> List["capsule.Capsule"]:
        """
        Generates a list of capsules representing the drone's path segments for collision detection.

        Returns:
            A list of Capsule objects.
        """
        collider_list: List["capsule.Capsule"] = []
        path_points = [self.start] + self.waypoints + [self.end]
        for i in range(len(path_points) - 1):
            collider_list.append(capsule.Capsule(path_points[i], path_points[i + 1], self.radius))
        return collider_list

    def get_intersecting_count(self, drones: List["Drone"]) -> int:
        """
        Calculates the number of intersections between this drone's path and other drones' paths.

        Args:
            drones: A list of other Drone objects to check for intersections against.

        Returns:
            The total count of intersecting capsule pairs.
        """
        collision_count = 0
        self_colliders = self.get_colliders()
        for other_drone in drones:
            # Avoid checking collision with itself
            if other_drone is self:
                continue
            other_colliders = other_drone.get_colliders()
            for self_collider in self_colliders:
                for other_collider in other_colliders:
                    # Assuming capsule.are_capsules_intersecting checks for intersection
                    if capsule.are_capsules_intersecting(self_collider, other_collider):
                        collision_count += 1
        return collision_count

    def randomize_waypoint(self, waypoint_index: int):
        """
        Randomly adjusts a specific waypoint or the start/end point using a Gaussian distribution.

        Args:
            waypoint_index: The index of the waypoint to randomize.
                            0 for the start point, 1 to len(self.waypoints) for waypoints,
                            len(self.waypoints) + 1 for the end point.
        """
        if waypoint_index == 0:
            self.start = random_pos_from_gauss(self.start, self.mutation_gauss)

        elif 1 <= waypoint_index <= len(self.waypoints):
            self.waypoints[waypoint_index - 1] = random_pos_from_gauss(self.waypoints[waypoint_index - 1],
                                                                       self.mutation_gauss)

    def mutation(self, change_chance: float = 0.8, add_chance: float = 0.1):
        """
        Applies a random mutation to the drone's path (change, add, or remove a waypoint).

        Args:
            change_chance: The probability of changing an existing waypoint/start/end.
            add_chance: The probability of adding a new waypoint.
                        The probability of removing a waypoint is 1 - change_chance - add_chance.
        """
        random_result = random()

        if random_result <= change_chance:
            # Change an existing point (start, waypoint, or end)
            # Index 0 for start, 1 to len(waypoints) for waypoints, len(waypoints)+1 for end
            index_to_mutate = randint(0, len(self.waypoints))
            self.randomize_waypoint(index_to_mutate)
        elif random_result <= change_chance + add_chance:
            # Add a new waypoint
            insert_index = randint(0, len(self.waypoints))

            # Get origin point for new waypoint
            origin = self.start
            if self.waypoints:
                origin = self.waypoints[insert_index-1]

            # Initialize new waypoint at a position based on its neighbors or a default
            self.waypoints.insert(insert_index, origin)
            # Randomize the newly added waypoint
            self.randomize_waypoint(insert_index + 1)  # +1 because it's a waypoint
        else:
            # Remove a waypoint (only if there are waypoints to remove)
            if self.waypoints:
                remove_index = randint(0, len(self.waypoints) - 1)
                self.waypoints.pop(remove_index)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Drone object.
        """
        return f"Drone: (Start: {self.start}, End: {self.end}, Waypoints: {self.waypoints})"

    def __eq__(self, other: ["Drone"]):
        return self.start == other.start and self.waypoints == other.waypoints and self.end == other.end


def random_drone(end_pos: Position, bounding_start: Position, bounding_end: Position,
                 amount_gauss: Tuple[float, float], position_gauss: Tuple[float, float], mutation_gauss: Tuple[float, float] = None) -> Drone:
    """
    Generates a random Drone with a path.

    Args:
        end_pos: The target ending position for the drone.
        bounding_start: The starting point of the bounding box for initial waypoint generation.
        bounding_end: The ending point of the bounding box for initial waypoint generation.
        amount_gauss: Gaussian parameters (mean, std_dev) for the number of waypoints.
        position_gauss: Gaussian parameters (mean, std_dev) for generating waypoint positions relative to the previous point.
        mutation_gauss: Gaussian parameters (mean, std_dev) for mutation.

    Returns:
        A randomly generated Drone object.
    """
    start_pos = random_pos_in_range(bounding_start, bounding_end)

    # Ensure waypoint amount is non-negative
    waypoint_amount = max(0, round(gauss(*amount_gauss)))
    waypoints: List[Position] = []

    # Generate waypoints based on Gaussian distribution from the previous point
    current_pos = start_pos
    for _ in range(waypoint_amount):
        next_waypoint = random_pos_from_gauss(current_pos, position_gauss)
        waypoints.append(next_waypoint)
        current_pos = next_waypoint

    return Drone(start_pos, end_pos, 0.5, mutation_gauss, waypoints)  # Using position_gauss for mutation_gauss as well


class DronePathGenome:
    """
    Represents a genome in a genetic algorithm, consisting of a set of drones.
    """

    def __init__(self, targets: List[Position], mutation_chance=0.005, drones: Optional[List[Drone]] = None):
        """
        Initializes a DronePathGenome.

        Args:
            targets: A list of target positions for the drones.
            drones: A list of Drone objects. Defaults to an empty list if None.
        """
        self.targets: List[Position] = targets
        self.mutation_chance = mutation_chance
        # Ensure drones list has the same length as targets, potentially creating empty drones
        # if the initial list is shorter or None. Or, assume drones list is provided correctly.
        # Assuming drones list is provided correctly or is None for now.
        self.drones: List[Drone] = drones if drones is not None else []

        # Basic check for consistency
        if drones is not None and len(drones) != len(targets):
            warn("Warning: Number of drones does not match number of targets.")

    def fitness(self, distance_coefficient: float, waypoint_coefficient: float,
                drone_collision_coefficient: float, target_collision_coefficient: float) -> float:
        """
        Calculates the fitness of the genome based on various criteria.

        Args:
            distance_coefficient: Weight for the total distance of drone paths.
            waypoint_coefficient: Weight for the number of waypoints.
            drone_collision_coefficient: Weight for collisions between drones.
            target_collision_coefficient: Weight for collisions with targets (this part is missing in the original logic).

        Returns:
            The calculated fitness score.
        """
        resulting_fitness = 0.0

        # Calculate fitness components for each drone
        for i, drone in enumerate(self.drones):
            resulting_fitness -= distance_coefficient * drone.distance()
            resulting_fitness -= waypoint_coefficient * len(drone.waypoints)

            # Calculate collisions with other drones
            # Pass the whole list and let the method handle self-exclusion
            resulting_fitness -= drone_collision_coefficient * drone.get_intersecting_count(self.drones)

            # TODO: Implement target collision detection and penalize fitness
            # This would require knowing the size/collider of the targets.
            # resulting_fitness -= target_collision_coefficient * drone.check_target_collisions(self.targets[i])

        return resulting_fitness

    def cross(self, other: "DronePathGenome") -> "DronePathGenome":
        """
        Performs a crossover operation with another genome to create a new genome.
        Assumes both genomes have the same number of drones/targets.

        Args:
            other: The other DronePathGenome to cross with.

        Returns:
            A new DronePathGenome resulting from the crossover.
        """
        # Ensure genomes have the same number of drones for meaningful crossover
        if len(self.drones) != len(other.drones):
            # Handle this case, perhaps by raising an error or returning one of the parents
            warn("Warning: Cannot perform crossover on genomes with different numbers of drones.")
            return choice([self, other])  # Return one of the parents randomly

        random_count = randint(1, len(self.drones))

        # Create a new list of drones by combining segments from self and other
        new_drones = self.drones[:random_count] + other.drones[random_count:]

        for drone in new_drones:
            if random() < self.mutation_chance / len(new_drones):
                drone.mutation()

        unique_targets = set()
        for drone in new_drones:
            unique_targets.add(tuple(drone.end))
        if len(unique_targets) != len(new_drones):
            warn("WARNING: Drone has duplicate targets. Did something go wrong?")

        return DronePathGenome(targets=self.targets, drones=new_drones)

    def __repr__(self) -> str:
        """
        Returns a string representation of the DronePathGenome object.
        """
        return f"Drone Genome: (Targets: {self.targets}, Drones: {self.drones})"

    def __eq__(self, other: ["DronePathGenome"]):
        return self.drones == other.drones


def random_drone_path_genome(bounding_start: Position, bounding_end: Position, targets: List[Position],
                             amount_gauss: Tuple[float, float], position_gauss: Tuple[float, float],
                             mutation_chance: float = None, mutation_gauss: Tuple[float, float] = None) -> DronePathGenome:
    """
    Generates a random DronePathGenome with drones targeting the specified positions.

    Args:
        bounding_start: The starting point of the bounding box for initial drone/waypoint generation.
        bounding_end: The ending point of the bounding box for initial drone/waypoint generation.
        targets: A list of target positions for each drone.
        amount_gauss: Gaussian parameters for the number of waypoints per drone.
        position_gauss: Gaussian parameters for generating waypoint positions.
        mutation_gauss: Gaussian parameters (mean, std_dev) for mutation.
        mutation_chance: Chance for drone mutation every cross.

    Returns:
        A randomly generated DronePathGenome.
    """
    # Create a list of random drones, one for each target
    drones = [random_drone(target, bounding_start, bounding_end, amount_gauss, position_gauss, mutation_gauss) for target in targets]

    return DronePathGenome(targets=targets, mutation_chance=mutation_chance,  drones=drones)

# #  # Example Usage (uncomment to run)
# if __name__ == "__main__":
#     # Define bounding box and targets
#     bounding_start = np.array([0.0, 0.0, 0.0])
#     bounding_end = np.array([50.0, 50.0, 50.0])
#     targets = [np.array([10.0, 10.0, 10.0]), np.array([25.0, 25.0, 25.0]), np.array([40.0, 40.0, 40.0])]
#
#     # Define Gaussian parameters for waypoint amount and position
#     amount_gauss = (5, 2) # Mean 5 waypoints, std_dev 2
#     position_gauss = (5, 5) # Mean displacement 5, std_dev 5
#
#     # Generate a random genome
#     genome = random_drone_path_genome(bounding_start, bounding_end, targets, amount_gauss, position_gauss)
#     print(genome)
#
#     # Example of calculating fitness (assuming coefficients)
#     fitness_score = genome.fitness(distance_coefficient=1.0, waypoint_coefficient=0.5,
#                                    drone_collision_coefficient=100.0, target_collision_coefficient=0.0)
#     print(f"Fitness: {fitness_score}")
#
#     # Example of mutation
#     print("\nMutating genome...")
#     genome.drones[0].mutation() # Mutate the first drone
#     print(genome)
#
#     # Example of crossover (create another genome first)
#     other_genome = random_drone_path_genome(bounding_start, bounding_end, targets, amount_gauss, position_gauss)
#     crossed_genome = genome.cross(other_genome)
#     print("\nCrossed genome:")
#     print(crossed_genome)
