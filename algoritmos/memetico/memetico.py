print("Iniciando Otimização com Algoritmo Memético...")

import numpy as np
np.random.seed(42)
import pandas as pd

meta_dataset = pd.read_csv('..\..\meta_dataset_bestcol.csv')

classifier_cols = [c for c in meta_dataset.columns if c in ['DecisionTree', 'SVM', 'KNN',
                                                           'LogisticRegression', 'Perceptron', 'MLP']]
meta_feature_cols = [c for c in meta_dataset.columns if c not in ['dataset', 'Best'] + classifier_cols]

# Parâmetros do AG Base
pop_size = 20
n_generations = 15 # Menos gerações pois a busca local consome mais avaliações
mutation_rate = 0.1
crossover_rate = 0.8

num_features = len(meta_feature_cols)
population_mem = np.random.randint(2, size=(pop_size, num_features))
fitness_cache_mem = {}

def fitness_memetic(ind):
    """Calcula o F1-score utilizando cache"""
    key = tuple(ind.tolist())
    if key not in fitness_cache_mem:
        features = [meta_feature_cols[i] for i in range(num_features) if ind[i] == 1]
        if len(features) == 0:
            # Força pelo menos uma feature ativa se o vetor for todo zero
            ind[np.random.randint(num_features)] = 1
            features = [meta_feature_cols[i] for i in range(num_features) if ind[i] == 1]
        fitness_cache_mem[key] = evaluate_feature_subset(features)
    return fitness_cache_mem[key]

def hill_climbing_search(ind, max_steps=5):
    """Busca Local: Inverte aleatoriamente genes para tentar melhorar o fitness individual"""
    best_ind = ind.copy()
    best_fit = fitness_memetic(best_ind)

    for _ in range(max_steps):
        neighbor = best_ind.copy()
        # Escolhe um gene aleatório para inverter (0 -> 1 ou 1 -> 0)
        idx = np.random.randint(num_features)
        neighbor[idx] = 1 - neighbor[idx]

        neighbor_fit = fitness_memetic(neighbor)
        if neighbor_fit > best_fit:
            best_ind = neighbor
            best_fit = neighbor_fit

    return best_ind, best_fit

# Avaliação Inicial com refinamento local (Memetismo Inicial)
scores_mem = []
for i in range(pop_size):
    population_mem[i], fit = hill_climbing_search(population_mem[i])
    scores_mem.append(fit)

history_memetic = []

# --- Loop Evolucionário Memético ---
for generation in range(n_generations):
    # Seleção por Torneio
    new_population = []
    for _ in range(pop_size):
        i1, i2 = np.random.choice(pop_size, 2, replace=False)
        winner = population_mem[i1] if scores_mem[i1] > scores_mem[i2] else population_mem[i2]
        new_population.append(winner.copy())

    # Crossover
    for i in range(0, pop_size, 2):
        if np.random.rand() < crossover_rate and i+1 < pop_size:
            crossover_point = np.random.randint(1, num_features)
            # Troca de pedaços
            new_population[i][crossover_point:], new_population[i+1][crossover_point:] = \
                new_population[i+1][crossover_point:].copy(), new_population[i][crossover_point:].copy()

    # Mutação seguido de Busca Local (O toque Memético)
    for i in range(pop_size):
        # Mutação tradicional
        for j in range(num_features):
            if np.random.rand() < mutation_rate:
                new_population[i][j] = 1 - new_population[i][j]

        # Otimização Individual (Busca Local) antes de consolidar a geração
        new_population[i], _ = hill_climbing_search(new_population[i])

    population_mem = np.array(new_population)
    scores_mem = [fitness_memetic(ind) for ind in population_mem]

    best_score = np.max(scores_mem)
    mean_score = np.mean(scores_mem)
    best_ind = population_mem[np.argmax(scores_mem)]
    num_selected = np.sum(best_ind)

    history_memetic.append({
        'generation': generation,
        'best_f1': best_score,
        'mean_f1': mean_score,
        'features_count': num_selected
    })
    print(f'Geração {generation:02d}: Melhor F1 = {best_score:.4f} | Média F1 = {mean_score:.4f} | Features Ativas = {num_selected}')

# --- Resultados Finais do Memético ---
best_idx_mem = int(np.argmax(scores_mem))
selected_meta_features_memetic = [meta_feature_cols[i] for i in range(num_features) if population_mem[best_idx_mem][i] == 1]
best_memetic_f1 = scores_mem[best_idx_mem]

history_memetic_df = pd.DataFrame(history_memetic)

# Experimento Final via LOO
memetic_dataset_filter = meta_dataset[['Dataset'] + selected_meta_features_memetic + classifier_cols + ['Best']]
summary_df_memetic, meta_model_accuracy_memetic, meta_model_f1_memetic, feature_importances_memetic = train_and_evaluate_meta_model(memetic_dataset_filter)