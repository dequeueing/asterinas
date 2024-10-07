import random

# 目标字符串
TARGET = "Hello, World!"
# 基因池
GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,.-;:_!"

# 遗传算法参数
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
GENERATIONS = 5000

# 生成初始种群
def generate_population(size, length):
    return [''.join(random.choice(GENES) for _ in range(length)) for _ in range(size)]

# 适应度函数
def fitness(target, individual):
    return sum(t == i for t, i in zip(target, individual))

# 选择函数
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=selection_probs, k=2)

# 交叉函数
def crossover(parent1, parent2):
    pos = random.randint(1, len(parent1) - 1)
    return parent1[:pos] + parent2[pos:], parent2[:pos] + parent1[pos:]

# 变异函数
def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(GENES)
    return ''.join(individual)

# 主遗传算法函数
def genetic_algorithm(target, population_size, mutation_rate, generations):
    population = generate_population(population_size, len(target))
    for generation in range(generations):
        fitnesses = [fitness(target, individual) for individual in population]
        if max(fitnesses) == len(target):
            break
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population += [mutate(child1, mutation_rate), mutate(child2, mutation_rate)]
        population = new_population
        if generation % 100 == 0:
            print(f"Generation {generation}: Best result = {max(fitnesses)}/{len(target)}")
    best_fitness = max(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    return best_individual, best_fitness

# 运行遗传算法
best_individual, best_fitness = genetic_algorithm(TARGET, POPULATION_SIZE, MUTATION_RATE, GENERATIONS)
print(f"Best individual: {best_individual}")
print(f"Fitness: {best_fitness}/{len(TARGET)}")
