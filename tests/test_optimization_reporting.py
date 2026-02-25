from ipd_core import TFT, ALLD, ALLC
from optimization import GeneticAlgorithm, EDA


def test_optimizer_reports_fitness_evaluations():
    opponents = [TFT, ALLD, ALLC]

    ga = GeneticAlgorithm(population_size=10, memory_depth=1, num_rounds=10)
    r_ga = ga.evolve(opponents, generations=3)
    assert r_ga.fitness_evaluations > 0

    eda = EDA(population_size=10, memory_depth=1, num_rounds=10)
    r_eda = eda.evolve(opponents, generations=3)
    assert r_eda.fitness_evaluations > 0
