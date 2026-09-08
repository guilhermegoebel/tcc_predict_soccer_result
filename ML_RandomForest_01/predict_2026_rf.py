"""
Gera um CSV comparativo (resultado real x resultado predito) aplicando
o modelo Random Forest já treinado (rf_match_result_model.pkl) na base
football_matches_2026.

Pré-requisito: ter rodado train_rf.py antes, na mesma pasta, para gerar:
    - rf_match_result_model.pkl
    - rf_imputer.pkl
    - rf_calibrators.pkl
    - rf_preprocessing_artifacts.pkl

Uso:
    python predict_2026_rf.py

-----------------------------------------------------------------
CHANGELOG (alinhado com as melhorias feitas em predict_2026.py):
1. Não depende mais do preprocessing_artifacts.pkl do XGBoost. Desde
   que train_rf.py passou a calcular feature_columns localmente (sem
   frequency encoding de time nem dummies de competição), o RF ficou
   independente do XGBoost também para prever — a lista de features
   agora vem de rf_preprocessing_artifacts.pkl, gerado pelo próprio
   train_rf.py.
2. Removidos por completo o bucket_competition, o frequency encoding
   (home_team_freq/away_team_freq) e as flags home_team_seen/
   away_team_seen: o RF não usa nem nunca usou essas colunas como
   feature (elas ficaram só no XGBoost). Mantê-las aqui seria código
   morto e, pior, mascarava times novos de 2026 com uma frequência
   igual a zero em vez de simplesmente não existir a informação — só
   fazia sentido enquanto o RF lia esse artefato do XGBoost.
3. As probabilidades agora passam pela calibração isotônica salva em
   rf_calibrators.pkl (mesmo princípio do passo 6 do predict_2026.py),
   em vez de sair cru do predict_proba do Random Forest.
4. O aviso de features ausentes agora mostra também o percentual de
   partidas afetadas, e passou a existir um segundo aviso para
   features parcialmente ausentes (NaN espalhado, não 100% faltante).
-----------------------------------------------------------------
"""

import json
import pickle

import numpy as np
import pandas as pd

# =============================================================
# ARQUIVOS E CAMINHOS
# =============================================================
# Volta uma pasta para ler a base de 2026 na raiz do projeto
INPUT_FILE = '../football_matches_2026.csv'
MODEL_FILE = 'rf_match_result_model.pkl'
IMPUTER_FILE = 'rf_imputer.pkl'
CALIBRATORS_FILE = 'rf_calibrators.pkl'
ARTIFACTS_FILE = 'rf_preprocessing_artifacts.pkl'
OUTPUT_FILE = 'comparativo_real_vs_predito_2026_rf.csv'

RESULT_LABELS = {0: 'away_win', 1: 'draw', 2: 'home_win'}
TARGET_COLUMN = 'match_result'


# =============================================================
# 1. CARREGAR MODELO E ARTEFATOS DO TREINO (TUDO DO PRÓPRIO RF)
# =============================================================

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

with open(IMPUTER_FILE, 'rb') as f:
    imputer = pickle.load(f)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)
feature_columns = artifacts['feature_columns']

# Pode não existir em modelos treinados antes da calibração ser
# adicionada — mantém compatibilidade retroativa.
try:
    with open(CALIBRATORS_FILE, 'rb') as f:
        calibrators = pickle.load(f)
except FileNotFoundError:
    calibrators = None
    print(f'[AVISO] {CALIBRATORS_FILE} não encontrado — usando probabilidades sem calibração.')


# =============================================================
# 2. CARREGAR DADOS DE 2026
# =============================================================

df = pd.read_csv(INPUT_FILE)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

has_real_result = TARGET_COLUMN in df.columns

print(f'Total de partidas em 2026 a prever: {len(df):,}')


# =============================================================
# 3. MONTAR X GARANTINDO AS MESMAS FEATURES DO TREINO
# =============================================================
# Se alguma feature esperada não existir na base 2026, avisa e
# preenche com NaN (o imputer do treino cuida do resto).

missing_features = [c for c in feature_columns if c not in df.columns]
if missing_features:
    print(f'\n[AVISO] Features ausentes na base 2026, preenchidas com NaN:')
    for col in missing_features:
        df[col] = float('nan')
        print(f'  - {col}: 100.0% das partidas sem esse dado')

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

# Imputação de NaNs com a mediana do treino (necessário para o RF,
# que ao contrário do XGBoost não lida nativamente com NaN).
X_imp = imputer.transform(X)


# =============================================================
# 4. PREDIÇÃO
# =============================================================

y_pred = model.predict(X_imp)
y_pred_proba_raw = model.predict_proba(X_imp)

if calibrators:
    y_pred_proba = np.zeros_like(y_pred_proba_raw)
    for class_idx, iso in calibrators.items():
        y_pred_proba[:, class_idx] = iso.predict(y_pred_proba_raw[:, class_idx])
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_proba = y_pred_proba / row_sums
else:
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
# 5. MONTAR CSV COMPARATIVO
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

print(f'\nCSV comparativo do Random Forest salvo em {OUTPUT_FILE}')
print(df_out.head(10).to_string())

if has_real_result:
    acc = df_out['acertou'].mean()
    print(f'\nAcurácia simples na base 2026: {acc:.4f}')
    prediction_metrics = {
        'num_matches': int(len(df_out)),
        'has_real_result': True,
        'accuracy': float(acc),
    }
else:
    prediction_metrics = {
        'num_matches': int(len(df_out)),
        'has_real_result': False,
        'accuracy': None,
    }

with open('rf_prediction_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(prediction_metrics, f, indent=2, ensure_ascii=False)

print('Métricas de predição salvas em rf_prediction_metrics.json')