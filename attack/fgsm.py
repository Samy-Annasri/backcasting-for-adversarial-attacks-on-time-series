import torch
import numpy as np
import random

def evaluate_population_fitness(model, original_seq, population, device):
    with torch.no_grad():
        model.eval()
        inputs = torch.stack(population).to(device)
        preds = model(inputs).squeeze().cpu().numpy()
        
        orig_pred = model(original_seq.unsqueeze(0).to(device)).item()

        fitness = np.abs(preds - orig_pred)
    return fitness

def genetic_attack(
    model,
    original_seq,
    epsilon=0.1,
    population_size=20,
    generations=30,
    mutation_rate=0.1,
    crossover_rate=0.5,
    device="cpu"
):
    original_seq = original_seq.to(device)
    population = [original_seq + (torch.rand_like(original_seq) - 0.5) * 2 * epsilon for _ in range(population_size)]
    
    for gen in range(generations):
        fitness_scores = evaluate_population_fitness(model, original_seq, population, device)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        new_population = sorted_population[:population_size//2]

        while len(new_population) < population_size:
            parents = random.sample(new_population[:population_size // 4], 2)
            p1, p2 = parents
            if random.random() < crossover_rate:
                alpha = random.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.clone()

            if random.random() < mutation_rate:
                mutation = (torch.rand_like(original_seq) - 0.5) * 2 * epsilon * 0.1
                child = torch.clamp(child + mutation, original_seq - epsilon, original_seq + epsilon)

            new_population.append(child)

        population = new_population

    # Final best perturbation
    final_fitness = evaluate_population_fitness(model, original_seq, population, device)
    best_idx = np.argmax(final_fitness)
    return population[best_idx]

