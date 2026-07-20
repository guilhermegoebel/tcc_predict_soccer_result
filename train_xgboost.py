"""
Treino de um modelo XGBoost para prever o resultado de partidas
(match_result: 0 = vitória visitante, 1 = empate, 2 = vitória mandante)
a partir do dataset gerado por script.py.

Requisitos:
    pip install xgboost scikit-learn pandas numpy matplotlib --break-system-packages

Uso:
    python train_xgboost.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# =============================================================
# CONFIGURAÇÕES
# =============================================================

INPUT_FILE = 'football_matches_ml.csv'

# Split temporal: nunca aleatório, porque features como h2h_* e
# recent_* dependem do histórico acumulado cronologicamente.
# Ajuste os anos conforme o range real da sua base.
TRAIN_END_YEAR = 2021      # treino: até este ano (inclusive)
VAL_END_YEAR = 2023        # validação (early stopping): anos seguintes
                            # teste: tudo depois de VAL_END_YEAR

# Colunas que nunca devem entrar como feature (identificadores, ou
# redundantes/já decididas para remoção — ver conversas anteriores).
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',       # vazamento: só se sabe depois do jogo
    'away_goals',       # vazamento: idem
    'rank_diff',        # redundante com home_rank/away_rank (se existir no CSV)
    'points_diff',      # redundante com home_points/away_points (se existir)
    'match_result'      # é o target
]

TARGET_COLUMN = 'match_result'


# =============================================================
# 1. CARREGAR DADOS
# =============================================================

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

print(f'Total de partidas: {len(df):,}')
print(f'Distribuição do target (match_result):')
print(df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100)


# =============================================================
# 2. FEATURE ENGINEERING LEVE
# =============================================================

def bucket_competition(comp):
    """
    Agrupa o texto livre de 'competition' em poucas categorias,
    sem reintroduzir a lógica de peso do Elo — só um bucket
    categórico simples para o modelo diferenciar o tipo de jogo.
    """

    comp = str(comp).lower()

    if 'friendl' in comp:
        return 'friendly'

    if 'world cup' in comp and (
        'qualif' not in comp
    ):
        return 'world_cup_final'

    if 'qualif' in comp:
        return 'qualifiers'

    if 'cup' in comp or 'championship' in comp:
        return 'cup_or_championship'

    return 'other'


df['competition_bucket'] = df['competition'].apply(bucket_competition)


# =============================================================
# 3. SPLIT TEMPORAL (treino / validação / teste)
# =============================================================

train_mask = df['year'] <= TRAIN_END_YEAR
val_mask = (df['year'] > TRAIN_END_YEAR) & (df['year'] <= VAL_END_YEAR)
test_mask = df['year'] > VAL_END_YEAR

df_train = df.loc[train_mask].copy()
df_val = df.loc[val_mask].copy()
df_test = df.loc[test_mask].copy()

print(
    f'\nTreino: {len(df_train):,} partidas '
    f'(até {TRAIN_END_YEAR})'
)
print(
    f'Validação: {len(df_val):,} partidas '
    f'({TRAIN_END_YEAR + 1}-{VAL_END_YEAR})'
)
print(
    f'Teste: {len(df_test):,} partidas '
    f'(a partir de {VAL_END_YEAR + 1})'
)

if len(df_val) == 0 or len(df_test) == 0:
    raise ValueError(
        'Val ou teste ficaram vazios. Ajuste TRAIN_END_YEAR/'
        'VAL_END_YEAR para o range real de anos da sua base '
        '(veja df["year"].min()/max()).'
    )


# =============================================================
# 4. ENCODING DE TIME (frequency encoding, ajustado só no treino)
# =============================================================
# Evita explosão de dimensionalidade do one-hot em centenas de times
# e evita vazamento: as frequências são calculadas SÓ com dados de
# treino e depois aplicadas em val/teste. Times nunca vistos no
# treino recebem 0 (tratado como faixa "raro/desconhecido").

def fit_frequency_encoding(train_df, column):
    freq = train_df[column].value_counts(normalize=True)
    return freq.to_dict()


def apply_frequency_encoding(df_in, column, freq_map):
    return df_in[column].map(freq_map).fillna(0.0)


home_team_freq = fit_frequency_encoding(df_train, 'home_team')
away_team_freq = fit_frequency_encoding(df_train, 'away_team')

# Times mandante e visitante compartilham o mesmo "universo" de
# seleções, então combinamos as duas colunas para um mapa único de
# força relativa de presença no dataset de treino.
combined_team_freq = fit_frequency_encoding(
    pd.concat([
        df_train[['home_team']].rename(columns={'home_team': 'team'}),
        df_train[['away_team']].rename(columns={'away_team': 'team'})
    ]),
    'team'
)

for split_df in (df_train, df_val, df_test):
    split_df['home_team_freq'] = apply_frequency_encoding(
        split_df, 'home_team', combined_team_freq
    )
    split_df['away_team_freq'] = apply_frequency_encoding(
        split_df, 'away_team', combined_team_freq
    )

competition_bucket_dummies_train = pd.get_dummies(
    df_train['competition_bucket'], prefix='comp'
)
competition_bucket_columns = competition_bucket_dummies_train.columns.tolist()

for split_df in (df_train, df_val, df_test):
    dummies = pd.get_dummies(
        split_df['competition_bucket'], prefix='comp'
    )
    # Garante as mesmas colunas em treino/val/teste, mesmo que uma
    # categoria não apareça em algum split.
    for col in competition_bucket_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    split_df[competition_bucket_columns] = dummies[competition_bucket_columns]


# =============================================================
# 5. MONTAR X / y
# =============================================================

existing_drop_columns = [
    col for col in DROP_COLUMNS if col in df.columns
]

feature_columns = [
    col for col in df_train.columns
    if col not in existing_drop_columns
    and col not in ['home_team', 'away_team', 'competition', 'competition_bucket']
]

print(f'\nFeatures usadas ({len(feature_columns)}):')
print(feature_columns)

X_train = df_train[feature_columns]
y_train = df_train[TARGET_COLUMN]

X_val = df_val[feature_columns]
y_val = df_val[TARGET_COLUMN]

X_test = df_test[feature_columns]
y_test = df_test[TARGET_COLUMN]

# NaN (ex.: home_market_value_avg ausente) é tratado nativamente
# pelo XGBoost via split direction aprendido — não precisamos
# imputar manualmente.


# =============================================================
# 6. PESOS DE AMOSTRA (corrige o viés de mandante)
# =============================================================
# 'balanced' pondera cada classe pelo inverso da sua frequência no
# treino, sem alterar os dados em si (alternativa ao oversampling).

sample_weight = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)


# =============================================================
# 7. TREINO DO MODELO
# =============================================================

model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=30
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f'\nMelhor iteração (early stopping): {model.best_iteration}')


# =============================================================
# 8. AVALIAÇÃO NO TESTE
# =============================================================

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE')
print('===================================')
print(f'F1-macro: {test_f1_macro:.4f}')
print(f'Log-loss: {test_log_loss:.4f}')

print('\nClassification report (0=away, 1=draw, 2=home):')
print(
    classification_report(
        y_test,
        y_pred,
        target_names=['away_win', 'draw', 'home_win']
    )
)

# Baseline de comparação: "sempre prever vitória do mandante" —
# serve para confirmar que o modelo aprendeu algo além do viés.
baseline_pred = np.full_like(y_test, fill_value=2)
baseline_f1 = f1_score(y_test, baseline_pred, average='macro')
print(
    f'Baseline "sempre mandante vence" — F1-macro: {baseline_f1:.4f} '
    f'(o modelo deve superar isso claramente)'
)


# =============================================================
# 9. MATRIZ DE CONFUSÃO
# =============================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['away_win', 'draw', 'home_win']
)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de confusão — conjunto de teste')
fig.tight_layout()
fig.savefig('confusion_matrix.png', dpi=150)
print('\nMatriz de confusão salva em confusion_matrix.png')


# =============================================================
# 10. IMPORTÂNCIA DAS FEATURES
# =============================================================

importances = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
importances.head(20).sort_values().plot(kind='barh', ax=ax)
ax.set_title('Top 20 features mais importantes (XGBoost)')
ax.set_xlabel('Importância (gain relativo)')
fig.tight_layout()
fig.savefig('feature_importance.png', dpi=150)
print('Importância de features salva em feature_importance.png')

print('\nTop 15 features:')
print(importances.head(15).round(4).to_string())


# =============================================================
# 11. SALVAR MODELO
# =============================================================

model.save_model('xgb_match_result.json')
print('\nModelo salvo em xgb_match_result.json')