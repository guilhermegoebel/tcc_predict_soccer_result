"""
Gera um CSV comparativo (resultado real x resultado predito) aplicando
o modelo já treinado (xgb_match_result.json) na base football_matches_2026.

Pré-requisito: ter rodado train_xgboost.py antes, para gerar:
    - xgb_match_result.json
    - preprocessing_artifacts.pkl

Uso:
    python predict_2026.py
"""

import pickle

import pandas as pd
from xgboost import XGBClassifier

# =============================================================

INPUT_FILE = 'football_matches_2026.csv'
MODEL_FILE = 'xgb_match_result.json'
ARTIFACTS_FILE = 'preprocessing_artifacts.pkl'
OUTPUT_FILE = 'comparativo_real_vs_predito_2026.csv'

RESULT_LABELS = {0: 'away_win', 1: 'draw', 2: 'home_win'}

# Mesma lista usada no treino, para não vazar colunas indevidas
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',
    'away_goals',
    'rank_diff',
    'points_diff',
    'match_result',
]

TARGET_COLUMN = 'match_result'


# =============================================================
# 1. CARREGAR MODELO E ARTEFATOS DO TREINO
# =============================================================

model = XGBClassifier()
model.load_model(MODEL_FILE)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)

combined_team_freq = artifacts['combined_team_freq']
competition_bucket_columns = artifacts['competition_bucket_columns']
feature_columns = artifacts['feature_columns']


# =============================================================
# 2. CARREGAR DADOS DE 2026
# =============================================================

df = pd.read_csv(INPUT_FILE)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

has_real_result = TARGET_COLUMN in df.columns

print(f'Total de partidas em 2026: {len(df):,}')


# =============================================================
# 3. MESMA BUCKETIZAÇÃO DE COMPETITION DO TREINO
# =============================================================

def bucket_competition(comp):
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


df['competition_bucket'] = df['competition'].apply(bucket_competition)


# =============================================================
# 4. APLICAR (NÃO REFAZER) O FREQUENCY ENCODING DO TREINO
# =============================================================
# Times que não existiam no treino (freq_map) recebem 0.0, igual
# ao comportamento original em val/teste.

df['home_team_freq'] = df['home_team'].map(combined_team_freq).fillna(0.0)
df['away_team_freq'] = df['away_team'].map(combined_team_freq).fillna(0.0)

dummies = pd.get_dummies(df['competition_bucket'], prefix='comp')
for col in competition_bucket_columns:
    if col not in dummies.columns:
        dummies[col] = 0
df[competition_bucket_columns] = dummies[competition_bucket_columns]


# =============================================================
# 5. MONTAR X GARANTINDO AS MESMAS FEATURES DO TREINO
# =============================================================
# Se alguma feature esperada não existir na base 2026 (ex.: um
# indicador novo que não foi calculado), avisa e preenche com NaN
# (o XGBoost trata nativamente).

missing_features = [c for c in feature_columns if c not in df.columns]
if missing_features:
    print(f'\n[AVISO] Features ausentes na base 2026, preenchidas com NaN:')
    print(missing_features)
    for col in missing_features:
        df[col] = float('nan')

X = df[feature_columns]


# =============================================================
# 6. PREDIÇÃO
# =============================================================

y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)

df['resultado_predito'] = pd.Series(y_pred).map(RESULT_LABELS)
df['prob_away_win'] = y_pred_proba[:, 0]
df['prob_draw'] = y_pred_proba[:, 1]
df['prob_home_win'] = y_pred_proba[:, 2]

if has_real_result:
    df['resultado_real'] = df[TARGET_COLUMN].map(RESULT_LABELS)
    df['acertou'] = df['resultado_real'] == df['resultado_predito']
else:
    df['resultado_real'] = 'desconhecido'
    df['acertou'] = None
    print('\n[AVISO] Coluna match_result não encontrada em 2026 — '
          'resultado_real ficará vazio (base sem gabarito ainda).')


# =============================================================
# 7. MONTAR CSV COMPARATIVO
# =============================================================

output_columns = []
if 'match_id' in df.columns:
    output_columns.append('match_id')
if 'date' in df.columns:
    output_columns.append('date')
output_columns += ['home_team', 'away_team']
if 'competition' in df.columns:
    output_columns.append('competition')
output_columns += [
    'resultado_real',
    'resultado_predito',
    'acertou',
    'prob_away_win',
    'prob_draw',
    'prob_home_win',
]

df_out = df[output_columns].copy()
df_out.to_csv(OUTPUT_FILE, index=False)

print(f'\nCSV comparativo salvo em {OUTPUT_FILE}')
print(df_out.head(10).to_string())

if has_real_result:
    acc = df_out['acertou'].mean()
    print(f'\nAcurácia simples na base 2026: {acc:.4f}')
