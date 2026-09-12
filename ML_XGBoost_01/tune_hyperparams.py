"""
Busca de hiperparâmetros para o modelo XGBoost de resultado de partidas.

Uso:
    python tune_xgboost_hyperparams.py

Saída:
    best_hyperparams.json — carregado automaticamente por
    train_xgboost.py na próxima execução, se presente.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

# =============================================================
# Config (mantém consistência com train_xgboost.py)
# =============================================================

INPUT_FILE = '../football_matches_ml.csv'

TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023

DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',
    'away_goals',
    'match_result'
]
TARGET_COLUMN = 'match_result'

N_SPLITS = 5          # nº de folds do TimeSeriesSplit
N_ITER = 40            # nº de combinações amostradas pelo RandomizedSearchCV
RANDOM_STATE = 42
SCORING = 'f1_macro'   # mesma métrica usada para reportar performance no script principal
N_JOBS = -1

OUTPUT_JSON = 'best_hyperparams.json'


def bucket_competition(comp):
    """Idêntico ao train_xgboost.py — mantém as mesmas features."""
    comp = str(comp).lower()
    if 'friendl' in comp:
        return 'friendly'
    if 'world cup' in comp and ('qualif' not in comp):
        return 'world_cup_final'
    if 'qualif' in comp:
        return 'qualifiers'
    if 'cup' in comp or 'championship' in comp:
        return 'cup_or_championship'
    return 'other'


# =============================================================
# 1. CARREGAR DADOS E MONTAR O POOL DE BUSCA (treino + validação)
# =============================================================

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

df['competition_bucket'] = df['competition'].apply(bucket_competition)

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
    if col not in existing_drop_columns
    and col not in ['home_team', 'away_team', 'competition', 'competition_bucket',
                     'home_team_freq', 'away_team_freq',
                     'home_team_seen', 'away_team_seen']
]

X_search = df_search[feature_columns]
y_search = df_search[TARGET_COLUMN]

print(f'Pool de busca (treino+validação, até {VAL_END_YEAR}): {len(X_search):,} partidas')
print(f'Features usadas na busca ({len(feature_columns)}): {feature_columns}')

# Mesmo racional do script principal: pondera classes pelo inverso da
# frequência, sem alterar os dados. Calculado uma única vez sobre todo
# o pool (mesma abordagem do train_xgboost.py); o RandomizedSearchCV
# recorta esse vetor automaticamente pelos índices de cada fold.
sample_weight_search = compute_sample_weight(class_weight='balanced', y=y_search)


# =============================================================
# 2. CV TEMPORAL
# =============================================================
# IMPORTANTE: o TimeSeriesSplit "cru" do sklearn corta o dataset por
# ÍNDICE DE LINHA, não por data. Como várias partidas acontecem no
# mesmo dia (datas FIFA, fases de grupo de Copa etc.), um corte de
# fold pode cair NO MEIO de um dia — parte das partidas daquele dia
# vai para o treino, parte para a validação do mesmo fold. Isso não é
# vazamento temporal "para o futuro" clássico (partidas do mesmo dia
# são simultâneas, uma não usa resultado da outra como feature), mas
# ainda assim quebra a garantia de "treino estritamente antes da
# validação" que queremos manter e polui a leitura do experimento.
#
# Correção: fazemos o split sobre as DATAS ÚNICAS (não sobre as
# linhas), e só depois mapeamos cada data de volta para as partidas
# daquele dia. Isso garante que o corte de cada fold sempre cai na
# fronteira entre dois dias distintos, nunca no meio de um dia.

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
# n_estimators entra na busca porque, sem early stopping (não é
# trivial usar eval_set por fold dentro do RandomizedSearchCV do
# sklearn), o número de árvores também precisa ser tratado como
# hiperparâmetro. Depois da busca, refinamos esse número com early
# stopping de verdade no split real treino/validação (etapa 5).

param_distributions = {
    'n_estimators': randint(150, 900),
    'learning_rate': uniform(0.01, 0.19),       # 0.01 - 0.20
    'max_depth': randint(3, 9),                  # 3 - 8
    'min_child_weight': randint(1, 12),
    'subsample': uniform(0.6, 0.4),               # 0.6 - 1.0
    'colsample_bytree': uniform(0.6, 0.4),        # 0.6 - 1.0
    'gamma': uniform(0.0, 0.5),
    'reg_lambda': uniform(0.5, 4.5),              # 0.5 - 5.0
    'reg_alpha': uniform(0.0, 1.0),
}

base_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=1,   # paralelismo fica no RandomizedSearchCV (n_jobs abaixo)
)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=cv_folds,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=2,
    refit=False,   # refit "de verdade" é feito na etapa 5, com early stopping
)


# =============================================================
# 4. RODAR A BUSCA
# =============================================================

print(f'\nIniciando RandomizedSearchCV: {N_ITER} combinações x {N_SPLITS} folds '
      f'= {N_ITER * N_SPLITS} treinos de XGBoost. Isso pode demorar.')

search.fit(X_search, y_search, sample_weight=sample_weight_search)

print(f'\nMelhor {SCORING} médio na CV temporal: {search.best_score_:.4f}')
print('Melhores hiperparâmetros encontrados:')
for k, v in search.best_params_.items():
    print(f'  {k}: {v}')

cv_results_df = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
cv_results_df.to_csv('cv_results_hyperparam_search.csv', index=False)
print('\nResultado completo de todas as combinações testadas salvo em '
      'cv_results_hyperparam_search.csv')


# =============================================================
# 5. REFINAR n_estimators COM EARLY STOPPING NO SPLIT REAL
# =============================================================
# A busca acima trata n_estimators como um hiperparâmetro qualquer,
# o que é uma aproximação (sem early stopping por fold). Agora usamos
# os MELHORES hiperparâmetros de regularização/estrutura encontrados,
# mas deixamos o nº de árvores ser decidido de novo por early stopping
# de verdade, no split real treino -> validação (o mesmo split
# temporal usado no script principal), que é o critério mais confiável
# para essa escolha.

best_params = dict(search.best_params_)
best_params.pop('n_estimators', None)  # será substituído pelo early stopping

train_mask = df['year'] <= TRAIN_END_YEAR
val_mask = (df['year'] > TRAIN_END_YEAR) & (df['year'] <= VAL_END_YEAR)

df_train = df.loc[train_mask].copy()
df_val = df.loc[val_mask].copy()

X_train, y_train = df_train[feature_columns], df_train[TARGET_COLUMN]
X_val, y_val = df_val[feature_columns], df_val[TARGET_COLUMN]

sample_weight_train = compute_sample_weight(class_weight='balanced', y=y_train)

refine_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_estimators=3000,          # teto alto; early stopping decide o real
    early_stopping_rounds=30,
    **best_params,
)

refine_model.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

final_n_estimators = int(refine_model.best_iteration) + 1
best_params['n_estimators'] = final_n_estimators

print(f'\nn_estimators final (via early stopping no split real): {final_n_estimators}')
print('\nHiperparâmetros finais recomendados:')
for k, v in best_params.items():
    print(f'  {k}: {v}')


# =============================================================
# 6. SALVAR
# =============================================================

output_path = Path(OUTPUT_JSON)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)

print(f'\nHiperparâmetros salvos em {output_path.resolve()}')
print(
    'train_xgboost.py carrega esse arquivo automaticamente na próxima '
    'execução (se existir no mesmo diretório), sem precisar editar o código.'
)
