import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

from util.train_evaluate import train_and_evaluate_meta_model

np.random.seed(42)

meta_dataset = pd.read_csv('..\meta_dataset_bestcol.csv')

classifier_cols = [c for c in meta_dataset.columns if c in ['DecisionTree', 'SVM', 'KNN',
                                                           'LogisticRegression', 'Perceptron', 'MLP']]
meta_feature_cols = [c for c in meta_dataset.columns if c not in ['dataset', 'Best'] + classifier_cols]

# Parâmetros
pop_size = 50       # População expandida para o ecossistema do BRKGA
n_generations = 30
pe = 0.20           # Proporção da população de elite (20%)
pm = 0.15           # Proporção de mutantes introduzidos a cada geração (15%)
rhoe = 0.70         # Probabilidade de herdar o alelo do pai de elite (70%)

n_elite = int(pop_size * pe)
n_mutants = int(pop_size * pm)
n_crossovers = pop_size - n_elite - n_mutants

# Inicialização do Cache de Fitness para o BRKGA
fitness_cache_brkga = {}

# Número de meta-features (Tamanho do cromossomo)
num_features = len(meta_feature_cols)

# Inicialização da população com chaves aleatórias pura: matriz (pop_size x num_features)
population_brkga = np.random.rand(pop_size, num_features)


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
    
    f1 = f1_score(y, y_pred, average='weighted')

    fitness = f1 - 0.01 * (len(feature_subset) / len(meta_feature_cols))

    return fitness


def decode_brkga(chrom, cols=meta_feature_cols, threshold=0.5):
    """
    Decodifica o vetor de chaves aleatórias [0, 1).
    Valores acima do threshold ativam a feature. Garante ao menos uma ativa.
    """
    selected_mask = chrom > threshold
    if not np.any(selected_mask):
        # Se nenhuma ficou acima do corte, força a maior chave a ser ativa
        selected_mask[np.argmax(chrom)] = True

    return [cols[i] for i, active in enumerate(selected_mask) if active]


def fitness_brkga(chrom):
    """Calcula o F1-score utilizando cache para evitar reprocessamento"""
    key = tuple(chrom.tolist())
    if key not in fitness_cache_brkga:
        features = decode_brkga(chrom)
        fitness_cache_brkga[key] = evaluate_feature_subset(features)
    return fitness_cache_brkga[key]


# Avaliação da população inicial
scores_brkga = [fitness_brkga(ind) for ind in population_brkga]

history_brkga = []

#  Loop Evolucionário
for generation in range(n_generations):
    # Ordenar população pelo fitness de forma decrescente
    ranked_indices = np.argsort(scores_brkga)[::-1]
    population_brkga = population_brkga[ranked_indices]
    scores_brkga = [scores_brkga[idx] for idx in ranked_indices]

    best_chrom = population_brkga[0]
    best_score = scores_brkga[0]
    mean_score = np.mean(scores_brkga)
    num_selected = len(decode_brkga(best_chrom))

    history_brkga.append({
        'generation': generation,
        'best_f1': best_score,
        'mean_f1': mean_score,
        'features_count': num_selected
    })

    print(f'Geração {generation:02d}: Melhor F1 = {best_score:.4f} | Média F1 = {mean_score:.4f} | Features Ativas = {num_selected}')

    # Separação estrita do BRKGA: Elite vs Não-Elite
    elite = population_brkga[:n_elite]
    non_elite = population_brkga[n_elite:]

    # 1. Copia a elite diretamente para a próxima geração (Elitismo)
    new_population = [ind.copy() for ind in elite]

    # 2. Crossover Biased (Elite + Não-Elite) baseado no parâmetro rhoe
    for _ in range(n_crossovers):
        p_elite = elite[np.random.randint(n_elite)]
        p_non_elite = non_elite[np.random.randint(len(non_elite))]

        # Se o vetor randômico for menor que rhoe, herda do pai elite, senão do não-elite
        child = np.where(np.random.rand(num_features) < rhoe, p_elite, p_non_elite)
        new_population.append(child)

    # 3. Introdução de Mutantes (Chaves Aleatórias Puras no espaço [0, 1))
    if n_mutants > 0:
        mutants = np.random.rand(n_mutants, num_features)
        new_population.extend(mutants)

    # Atualiza a população e calcula os novos scores
    population_brkga = np.array(new_population)
    scores_brkga = [fitness_brkga(ind) for ind in population_brkga]

#  Resultados Finais da Otimização BRKGA
best_idx_brkga = int(np.argmax(scores_brkga))
selected_meta_features_brkga = decode_brkga(population_brkga[best_idx_brkga])
best_brkga_f1 = scores_brkga[best_idx_brkga]

print('\n' + '='*50)
print('Melhor subconjunto encontrado pelo BRKGA:')
print(f'F1-Score obtido internamente: {best_brkga_f1:.4f}')
print(f'Quantidade de Meta-features selecionadas: {len(selected_meta_features_brkga)}')
print('='*50)

history_brkga_df = pd.DataFrame(history_brkga)

# EXPERIMENTO: Avaliação de Impacto do BRKGA no Meta-Modelo Final
print("\nExecutando o Experimento e Avaliação com subconjunto BRKGA via LOO...")

# Filtrando o meta_dataset para conter apenas as características selecionadas pelo BRKGA
brkga_dataset_filter = meta_dataset[['Dataset'] + selected_meta_features_brkga + classifier_cols + ['Best']]

# Rodando a validação Leave-One-Out estruturada no Passo 5
summary_df_brkga, meta_model_accuracy_brkga, meta_model_f1_brkga, feature_importances_brkga = train_and_evaluate_meta_model(brkga_dataset_filter)

print(f'\n[RESULTADOS DO META-MODELO COM BRKGA]')
print(f'Meta-model Accuracy (BRKGA): {meta_model_accuracy_brkga:.2f}')
print(f'Meta-model F1-score (BRKGA): {meta_model_f1_brkga:.2f}')