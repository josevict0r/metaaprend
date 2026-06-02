import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed


meta_dataset = pd.read_csv('..\..\meta_dataset_bestcol.csv')

classifier_cols = [c for c in meta_dataset.columns if c in ['DecisionTree', 'SVM', 'KNN',
                                                           'LogisticRegression', 'Perceptron', 'MLP']]
meta_feature_cols = [c for c in meta_dataset.columns if c not in ['dataset', 'Best'] + classifier_cols]

def evaluate_feature_subset(feature_subset):
    X = meta_dataset[feature_subset].copy()
    y = meta_dataset['Best'].values

    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = X.columns[X.isnull().all()].tolist()
    X = X.drop(columns=all_nan_cols)

    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )
    #f1 = f1_score(y, y_pred, average='weighted')
    #fitness = f1 - 0.01 * (len(feature_subset) / len(meta_feature_cols))
    return f1_score(y, y_pred, average='weighted')

def random_individual():
    ind = np.random.choice([0, 1], size=len(meta_feature_cols), p=[0.9, 0.1]) #####
    if ind.sum() == 0:
        ind[np.random.randint(len(ind))] = 1
    return ind

def decode_individual(ind):
    return [meta_feature_cols[i] for i, bit in enumerate(ind) if bit == 1]

fitness_cache = {}
def fitness(individual):
    key = tuple(individual.tolist())
    if key not in fitness_cache:
        fitness_cache[key] = evaluate_feature_subset(decode_individual(individual))
    return fitness_cache[key]

def tournament_selection(population, scores, k=3):
    contenders = np.random.choice(len(population), size=k, replace=False)
    best_idx = max(contenders, key=lambda idx: scores[idx])
    return population[best_idx].copy()

def one_point_crossover(parent1, parent2):
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(individual, mutation_prob=0.05):
    for i in range(len(individual)):
        if np.random.rand() < mutation_prob:
            individual[i] = 1 - individual[i]
    if individual.sum() == 0:
        individual[np.random.randint(len(individual))] = 1

np.random.seed(42)

pop_size = 140
n_generations = 30
crossover_prob = 0.8
mutation_prob = 0.05
elitism = 2

population = [random_individual() for _ in range(pop_size)]
scores = Parallel(
    n_jobs=-1,
    backend="loky"
)(
    delayed(fitness)(ind)
    for ind in population
)


history = []
for generation in range(n_generations):
    ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    best_individual, best_score = ranked[0]
    mean_score = np.mean(scores)
    history.append({
        'generation': generation,
        'best_f1': best_score,
        'mean_f1': mean_score,
        'features_count': int(best_individual.sum()),
        'Best individual': decode_individual(best_individual)
    })
    print(f'Generation {generation}: best F1 = {best_score:.4f}, mean F1 = {mean_score:.4f}, features = {int(best_individual.sum())}')

    new_population = [ind.copy() for ind, _ in ranked[:elitism]]

    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        if np.random.rand() < crossover_prob:
            child1, child2 = one_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        mutate(child1, mutation_prob)
        mutate(child2, mutation_prob)
        new_population.extend([child1, child2])

    population = new_population[:pop_size]
    scores = Parallel(
    n_jobs=-1,
    backend="loky"
    )(
    delayed(fitness)(ind)
    for ind in population
    )

best_idx = int(np.argmax(scores))
best_individual = population[best_idx]
selected_meta_features_ag = decode_individual(best_individual) # Renamed to ag for clarity later
best_meta_model_f1 = scores[best_idx]

print('\nBest feature subset found:')
print(selected_meta_features_ag)
print(f'Best meta-model F1 = {best_meta_model_f1:.4f}')
print(f'Number of selected meta-features = {len(selected_meta_features_ag)}')

history_df = pd.DataFrame(history)
history_df.to_json('history.json')