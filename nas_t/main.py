from evolutionary_algorithm import NASDifferentialEvolution
import time

overall_start_time = time.time()

if __name__ == "__main__":
    nas_de = NASDifferentialEvolution(verbose=True)
    nas_de.evolve()

    overall_end_time = time.time()