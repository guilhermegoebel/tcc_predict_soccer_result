"""
Gera um CSV comparativo (resultado real x resultado predito) aplicando
o modelo já treinado (xgb_match_result.json) na base football_matches_2026.

Pré-requisito: ter rodado train_xgboost.py antes, para gerar:
    - xgb_match_result.json
    - preprocessing_artifacts.pkl

Uso:
    python predict_2026.py

-----------------------------------------------------------------
CHANGELOG (melhorias sobre a versão anterior):
1. Removida a lista DROP_COLUMNS, que não era usada em nenhum
   lugar do script (código morto) e ainda divergia do treino,
   podendo confundir quem for manter o código.
2. Adicionada a flag "time visto no treino" (home_team_seen /
   away_team_seen), na mesma lógica usada no train_xgboost.py.
   Importante para 2026: seleções estreantes em Copa do Mundo vão
   ser marcadas explicitamente como "não vistas", em vez de ficarem
   escondidas atrás de uma frequência baixa igual à de um time raro.
3. As probabilidades agora passam pela calibração (isotonic
   regression) salva no treino, antes de ir para o CSV final —
   ficam mais próximas da chance real, e não só da preferência do
   modelo balanceado.
4. Aviso de features ausentes agora mostra também quantas partidas
   (%) ficaram com NaN em cada uma, não só o nome da coluna.
-----------------------------------------------------------------
"""

import json
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# =============================================================

INPUT_FILE = 'football_matches_2026.csv'
MODEL_FILE = 'xgb_match_result.json'
ARTIFACTS_FILE = 'preprocessing_artifacts.pkl'
OUTPUT_FILE = 'comparativo_real_vs_predito_2026.csv'

RESULT_LABELS = {0: 'away_win', 1: 'draw', 2: 'home_win'}

TARGET_COLUMN = 'match_result'


# =============================================================
# 1. CARREGAR MODELO E ARTEFATOS DO TREINO
# =============================================================

model = XGBClassifier()
model.load_model(MODEL_FILE)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)

combined_team_freq = artifacts.get('combined_team_freq')
known_teams = artifacts.get('known_teams')
competition_bucket_columns = artifacts['competition_bucket_columns']
feature_columns = artifacts['feature_columns']
calibrators = artifacts.get('calibrators')  # pode não existir em modelos antigos


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
# ao comportamento original em val/teste. Além disso, marcamos
# explicitamente quais times são "novos" para o modelo — relevante
# para seleções estreantes na Copa de 2026.

if combined_team_freq is not None:
    df['home_team_freq'] = df['home_team'].map(combined_team_freq).fillna(0.0)
    df['away_team_freq'] = df['away_team'].map(combined_team_freq).fillna(0.0)

if known_teams is not None:
    df['home_team_seen'] = df['home_team'].isin(known_teams).astype(int)
    df['away_team_seen'] = df['away_team'].isin(known_teams).astype(int)
else:
    # Modelos treinados sem frequency encoding não têm um universo de
    # times salvo para calcular essas flags; mantém a saída compatível.
    df['home_team_seen'] = 0
    df['away_team_seen'] = 0
    num_new_home = 0
    num_new_away = 0

if known_teams is not None:
    num_new_home = (df['home_team_seen'] == 0).sum()
    num_new_away = (df['away_team_seen'] == 0).sum()
if num_new_home or num_new_away:
    print(
        f'\n[AVISO] Times não vistos no treino: '
        f'{num_new_home} partidas com mandante novo, '
        f'{num_new_away} partidas com visitante novo. '
        f'Previsões para esses jogos tendem a ser menos confiáveis.'
    )

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
    for col in missing_features:
        df[col] = float('nan')
        print(f'  - {col}: 100.0% das partidas sem esse dado')

# Para features que existem mas têm NaN espalhado (não 100% ausentes),
# mostra a cobertura real — ajuda a perceber degradação silenciosa.
partially_missing = [
    c for c in feature_columns
    if c not in missing_features and df[c].isna().any()
]
if partially_missing:
    print('\n[AVISO] Features com NaN parcial na base 2026:')
    for col in partially_missing:
        pct = df[col].isna().mean() * 100
        print(f'  - {col}: {pct:.1f}% das partidas sem esse dado')

X = df[feature_columns]


# =============================================================
# 6. PREDIÇÃO
# =============================================================

y_pred = model.predict(X)
y_pred_proba_raw = model.predict_proba(X)

if calibrators:
    y_pred_proba = np.zeros_like(y_pred_proba_raw)
    for class_idx, iso in calibrators.items():
        y_pred_proba[:, class_idx] = iso.predict(y_pred_proba_raw[:, class_idx])
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_proba = y_pred_proba / row_sums
else:
    # Compatibilidade com artefatos gerados antes da calibração existir.
    y_pred_proba = y_pred_proba_raw

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
    'home_team_seen',
    'away_team_seen',
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
    prediction_metrics = {
        'num_matches': int(len(df_out)),
        'has_real_result': True,
        'accuracy': float(acc),
        'new_home_teams': int(num_new_home),
        'new_away_teams': int(num_new_away),
    }
else:
    prediction_metrics = {
        'num_matches': int(len(df_out)),
        'has_real_result': False,
        'accuracy': None,
        'new_home_teams': int(num_new_home),
        'new_away_teams': int(num_new_away),
    }

with open('prediction_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(prediction_metrics, f, indent=2, ensure_ascii=False)

print('Métricas de predição salvas em prediction_metrics.json')