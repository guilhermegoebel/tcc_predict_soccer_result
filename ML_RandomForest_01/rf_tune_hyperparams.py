"""
Busca de hiperparâmetros para o modelo Random Forest de resultado de
partidas.

Espelha exatamente a lógica de tune_xgboost_hyperparams.py — mesmo
pool de busca, mesmo esquema de CV temporal agrupado por data, mesma
métrica de seleção — adaptada às particularidades do Random Forest:
não há early stopping (não existe um "número de árvores decidido
automaticamente" como no XGBoost), então n_estimators entra na busca
como qualquer outro hiperparâmetro; e como o RF não trata NaN
nativamente, a imputação de mediana entra dentro de um Pipeline, para
que cada fold da CV impute com a mediana do PRÓPRIO fold de treino
(evitando vazamento entre folds).

Uso:
    python tune_rf_hyperparams.py

Saída:
    rf_best_hyperparams.json — carregado automaticamente por
    train_rf.py na próxima execução, se presente.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

# =============================================================
# Config (mantém consistência com train_rf.py e com
# tune_xgboost_hyperparams.py)
# =============================================================

INPUT_FILE = '../football_matches_ml.csv'

TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023

# Mesma lógica de exclusão de colunas usada em train_rf.py.
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',
    'away_goals',
    'match_result'
]
NON_FEATURE_COLUMNS = [
    'home_team',
    'away_team',
    'competition',
    'competition_bucket',
    'home_team_freq',
    'away_team_freq',
    'home_team_seen',
    'away_team_seen',
]
TARGET_COLUMN = 'match_result'

N_SPLITS = 5          # nº de folds do TimeSeriesSplit — igual ao XGBoost
N_ITER = 40            # nº de combinações amostradas — igual ao XGBoost
RANDOM_STATE = 42
SCORING = 'f1_macro'   # mesma métrica usada para reportar performance no script principal
N_JOBS = -1

OUTPUT_JSON = 'rf_best_hyperparams.json'
CV_RESULTS_CSV = 'cv_results_hyperparam_search_rf.csv'


# =============================================================
# 1. CARREGAR DADOS E MONTAR O POOL DE BUSCA (treino + validação)
# =============================================================

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

# Pool de busca = tudo até VAL_END_YEAR (treino + validação do script
# principal). O teste (> VAL_END_YEAR) fica de fora da busca.
search_mask = df['year'] <= VAL_END_YEAR
df_search = df.loc[search_mask].sort_values('date').reset_index(drop=True)

if df_search.empty:
    raise ValueError(
        'Pool de busca vazio. Verifique VAL_END_YEAR contra o range '
        'real de anos da base (df["year"].min()/max()).'
    )

existing_drop_columns = [c for c in DROP_COLUMNS if c in df.columns]
feature_columns = [
    col for col in df_search.columns
    if col not in existing_drop_columns and col not in NON_FEATURE_COLUMNS
]

X_search = df_search[feature_columns]
y_search = df_search[TARGET_COLUMN]

print(f'Pool de busca (treino+validação, até {VAL_END_YEAR}): {len(X_search):,} partidas')
print(f'Features usadas na busca ({len(feature_columns)}): {feature_columns}')


# =============================================================
# 2. CV TEMPORAL (idêntico ao tune_xgboost_hyperparams.py)
# =============================================================
# Mesmo problema, mesma correção: o TimeSeriesSplit "cru" do sklearn
# corta por ÍNDICE DE LINHA, não por data, o que pode partir um mesmo
# dia entre treino e validação de um fold. Split feito sobre as
# DATAS ÚNICAS, depois mapeado de volta para as partidas de cada dia.

def temporal_splits_by_date(dates: pd.Series, n_splits: int):
    """
    Gera splits de CV no estilo TimeSeriesSplit (janela expansível),
    mas agrupando por data única antes de cortar, para que nenhuma
    partida do mesmo dia seja dividida entre treino e validação.
    """
    unique_dates = np.sort(dates.unique())
    date_splitter = TimeSeriesSplit(n_splits=n_splits)
    date_positions = np.arange(len(unique_dates))

    for train_date_pos, val_date_pos in date_splitter.split(date_positions):
        train_dates_fold = set(unique_dates[train_date_pos])
        val_dates_fold = set(unique_dates[val_date_pos])
        train_idx = np.where(dates.isin(train_dates_fold))[0]
        val_idx = np.where(dates.isin(val_dates_fold))[0]
        yield train_idx, val_idx


# Materializa a lista de folds (RandomizedSearchCV itera o cv várias
# vezes — uma por combinação de hiperparâmetros — então precisa de uma
# lista reaproveitável, não de um gerador que se esgota na 1ª volta).
cv_folds = list(temporal_splits_by_date(df_search['date'], N_SPLITS))

print('\nVerificação dos folds temporais (agrupados por data):')
for i, (train_idx, val_idx) in enumerate(cv_folds):
    train_dates = df_search.loc[train_idx, 'date']
    val_dates = df_search.loc[val_idx, 'date']
    print(
        f'  Fold {i}: treino até {train_dates.max().date()} '
        f'({len(train_idx)} partidas) -> valida '
        f'{val_dates.min().date()} a {val_dates.max().date()} '
        f'({len(val_idx)} partidas)'
    )
    assert train_dates.max() < val_dates.min(), (
        f'Vazamento temporal detectado no fold {i}: treino contém '
        f'data posterior ao início da validação.'
    )


# =============================================================
# 3. ESPAÇO DE BUSCA E MODELO BASE
# =============================================================
# A imputação entra dentro de um Pipeline (em vez de ser feita uma
# única vez sobre X_search) para que cada fold da CV impute com a
# mediana do PRÓPRIO treino daquele fold — evita vazamento entre
# folds, o mesmo cuidado que o XGBoost dispensa por lidar com NaN
# nativamente. class_weight='balanced' segue a mesma lógica do
# sample_weight='balanced' usado no XGBoost: penaliza mais os erros
# nas classes minoritárias (empate, vitória visitante).

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=1,   # paralelismo fica no RandomizedSearchCV (n_jobs abaixo)
    )),
])

param_distributions = {
    'rf__n_estimators': randint(150, 900),
    'rf__max_depth': randint(3, 25),
    'rf__min_samples_leaf': randint(1, 20),
    'rf__min_samples_split': randint(2, 30),
    'rf__max_features': ['sqrt', 'log2', None],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=cv_folds,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=2,
    refit=False,   # o refit "de verdade" é feito manualmente em train_rf.py,
                   # só com X_train (ver passo 5 de train_rf.py)
)


# =============================================================
# 4. RODAR A BUSCA
# =============================================================

print(f'\nIniciando RandomizedSearchCV: {N_ITER} combinações x {N_SPLITS} folds '
      f'= {N_ITER * N_SPLITS} treinos de Random Forest. Isso pode demorar.')

search.fit(X_search, y_search)

print(f'\nMelhor {SCORING} médio na CV temporal: {search.best_score_:.4f}')
print('Melhores hiperparâmetros encontrados:')
for k, v in search.best_params_.items():
    print(f'  {k}: {v}')

cv_results_df = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
cv_results_df.to_csv(CV_RESULTS_CSV, index=False)
print(f'\nResultado completo de todas as combinações testadas salvo em {CV_RESULTS_CSV}')


# =============================================================
# 5. LIMPAR PREFIXO DO PIPELINE E SALVAR
# =============================================================
# search.best_params_ vem com o prefixo 'rf__' (por causa do
# Pipeline). train_rf.py espera um dict que possa ser passado direto
# como RandomForestClassifier(**best_params, class_weight='balanced',
# random_state=42, n_jobs=-1) — removemos o prefixo aqui para que o
# JSON salvo já esteja no formato que train_rf.py consome.

best_params = {
    key.replace('rf__', '', 1): value
    for key, value in search.best_params_.items()
}

print('\nHiperparâmetros finais recomendados (sem prefixo do pipeline):')
for k, v in best_params.items():
    print(f'  {k}: {v}')

output_path = Path(OUTPUT_JSON)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)

print(f'\nHiperparâmetros salvos em {output_path.resolve()}')
print(
    'train_rf.py carrega esse arquivo automaticamente na próxima '
    'execução (se existir no mesmo diretório), pulando a busca interna '
    'via RandomizedSearchCV + PredefinedSplit.'
)