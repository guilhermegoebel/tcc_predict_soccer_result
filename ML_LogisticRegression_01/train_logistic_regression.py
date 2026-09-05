"""
Treino de um modelo de Regressão Logística para prever o resultado de partidas
(match_result: 0 = vitória visitante, 1 = empate, 2 = vitória mandante)
a partir do dataset gerado por script.py - football_matches_ml.csv

Requisitos:
    pip install scikit-learn pandas numpy matplotlib --break-system-packages

Uso:
    python train_logistic_regression.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

# =============================================================

INPUT_FILE = 'football_matches_ml.csv'

# Split temporal: nunca aleatório, porque features como h2h_* e
# recent_* dependem do histórico acumulado cronologicamente.
TRAIN_END_YEAR = 2021      # treino: até este ano (inclusive)
VAL_END_YEAR = 2023        # validação (tuning): anos seguintes
                           # teste: tudo depois de VAL_END_YEAR

# Colunas que não entrarão como feature (identificadores, ou redundantes)
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',           # vazamento: só se sabe depois do jogo
    'away_goals',           # vazamento: idem
    'rank_diff',            # redundante para modelos lineares
    'points_diff',          # redundante para modelos lineares
    'market_value_avg_diff',# redundante para modelos lineares
    'match_result'          # é o target
]

TARGET_COLUMN = 'match_result'

# Valores para penalizar seleções obscuras sem ranking oficial
WORST_RANK = 215.0
WORST_POINTS = 0.0

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
# 2. TRATAMENTO DE COMPETITION (bucketização)
# =============================================================

def bucket_competition(comp):
    comp = str(comp).lower()
    if 'friendl' in comp: return 'friendly'
    if 'world cup' in comp and ('qualif' not in comp): return 'world_cup_final'
    if 'qualif' in comp: return 'qualifiers'
    if 'cup' in comp or 'championship' in comp: return 'cup_or_championship'
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
    f'\nTreino: {len(df_train):,} partidas (até {TRAIN_END_YEAR})'
)
print(
    f'Validação: {len(df_val):,} partidas ({TRAIN_END_YEAR + 1}-{VAL_END_YEAR})'
)
print(
    f'Teste: {len(df_test):,} partidas (a partir de {VAL_END_YEAR + 1})'
)

if len(df_val) == 0 or len(df_test) == 0:
    raise ValueError('Val ou teste ficaram vazios. Ajuste os anos.')

# =============================================================
# 5. ENCODING DE TIME E COMPETIÇÃO (Ajustado só no treino)
# =============================================================

def fit_frequency_encoding(train_df, column):
    freq = train_df[column].value_counts(normalize=True)
    return freq.to_dict()

def apply_frequency_encoding(df_in, column, freq_map):
    return df_in[column].map(freq_map).fillna(0.0)

combined_team_freq = fit_frequency_encoding(
    pd.concat([
        df_train[['home_team']].rename(columns={'home_team': 'team'}),
        df_train[['away_team']].rename(columns={'away_team': 'team'})
    ]), 'team'
)

for split_df in (df_train, df_val, df_test):
    split_df['home_team_freq'] = apply_frequency_encoding(split_df, 'home_team', combined_team_freq)
    split_df['away_team_freq'] = apply_frequency_encoding(split_df, 'away_team', combined_team_freq)

competition_bucket_dummies_train = pd.get_dummies(df_train['competition_bucket'], prefix='comp', drop_first=True)
competition_bucket_columns = competition_bucket_dummies_train.columns.tolist()

for split_df in (df_train, df_val, df_test):
    dummies = pd.get_dummies(split_df['competition_bucket'], prefix='comp', drop_first=True)
    for col in competition_bucket_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    split_df[competition_bucket_columns] = dummies[competition_bucket_columns]


# =============================================================
# 6. MONTAR X / y
# =============================================================

existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
feature_columns = [
    col for col in df_train.columns
    if col not in existing_drop_columns
    and col not in ['home_team', 'away_team', 'competition', 'competition_bucket']
]

X_train = df_train[feature_columns]
y_train = df_train[TARGET_COLUMN]

X_val = df_val[feature_columns]
y_val = df_val[TARGET_COLUMN]

X_test = df_test[feature_columns]
y_test = df_test[TARGET_COLUMN]


# =============================================================
# 7. IMPUTAÇÃO PELA MEDIANA E PADRONIZAÇÃO
# =============================================================

# 1. Imputação (aprende a mediana apenas no treino)
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# 2. Padronização (aprende a escala apenas no treino imputado)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# =============================================================
# 8. TREINO DO MODELO COM BUSCA DE HIPERPARÂMETROS (TUNING)
# =============================================================

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': ['balanced'],
    'solver': ['lbfgs', 'saga']
}

best_f1 = 0.0
best_params = None
best_model = None

# Loop pelas combinações usando ParameterGrid
for params in ParameterGrid(param_grid):
    # Instancia o modelo com os parâmetros do loop
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        **params
    )
    
    # Treina nos dados de TREINO
    model.fit(X_train, y_train)
    
    # Avalia nos dados de VALIDAÇÃO
    y_val_pred = model.predict(X_val)
    
    # Calcula o F1-macro em vez de Acurácia
    f1_val = f1_score(y_val, y_val_pred, average='macro')
    
    print(f"Testando {params} -> F1-Macro Val: {f1_val:.4f}")
    
    # Atualiza o melhor modelo se o F1 for maior
    if f1_val > best_f1:
        best_f1 = f1_val
        best_params = params
        best_model = model

print(f'\n[!] Melhor combinação encontrada: {best_params}')
print(f'[!] F1-Macro na Validação: {best_f1:.4f}')

model = best_model

y_val_pred_best = model.predict(X_val)

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE VALIDAÇÃO (MELHOR LR)')
print('===================================')
print(f'Accuracy: {accuracy_score(y_val, y_val_pred_best):.4f}')
print(f'F1-macro: {f1_score(y_val, y_val_pred_best, average="macro"):.4f}')
print(f'Balanced accuracy: {balanced_accuracy_score(y_val, y_val_pred_best):.4f}')
print(f'Precision macro: {precision_score(y_val, y_val_pred_best, average="macro"):.4f}')
print(f'Recall macro: {recall_score(y_val, y_val_pred_best, average="macro"):.4f}')

# =============================================================
# 9. AVALIAÇÃO NO TESTE FINAL COM O MELHOR MODELO
# =============================================================

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE (MELHOR LR)')
print('===================================')
print(f'Acurácia: {test_accuracy:.4f}')
print(f'F1-macro: {test_f1_macro:.4f}')
print(f'Log-loss: {test_log_loss:.4f}')

print('\nClassification report (0=away, 1=draw, 2=home):')
print(classification_report(y_test, y_pred, target_names=['away_win', 'draw', 'home_win']))

baseline_pred = np.full_like(y_test, fill_value=2)
baseline_f1 = f1_score(y_test, baseline_pred, average='macro')
print(f'\nBaseline "sempre mandante vence" — F1-macro: {baseline_f1:.4f}')


# =============================================================
# 10. MATRIZ DE CONFUSÃO
# =============================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['away_win', 'draw', 'home_win'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de Confusão - Melhor Regressão Logística')
fig.tight_layout()
fig.savefig('lr_confusion_matrix.png', dpi=150)
print('\nMatriz de confusão salva em lr_confusion_matrix.png')


# =============================================================
# 11. IMPORTÂNCIA DAS FEATURES
# =============================================================

importance_values = np.mean(np.abs(model.coef_), axis=0)

importances = pd.Series(
    importance_values,
    index=feature_columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
importances.head(20).sort_values().plot(kind='barh', ax=ax, color='blue')
ax.set_title('Top 20 features mais importantes (Melhor LR)')
ax.set_xlabel('Importância (Peso Absoluto Médio)')
fig.tight_layout()
fig.savefig('lr_feature_importance.png', dpi=150)
print('Importância de features salva em lr_feature_importance.png')

print('\nTop 15 features:')
print(importances.head(15).round(4).to_string())
print('\nGráficos gerados e salvos com sucesso!')


# =============================================================
# 12. SALVAR MODELO E ARTEFATOS
# =============================================================

# Salvando o modelo treinado
with open('lr_match_result_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModelo Regressão Logística otimizado salvo em 'lr_match_result_model.pkl'")

# Salvando o Imputer
with open('lr_imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("Imputer da mediana salvo em 'lr_imputer.pkl'")

# Salvando o Scaler (específico da LR)
with open('lr_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler salvo em 'lr_scaler.pkl'")

# Salvando as listas e dicionários de colunas
preprocessing_artifacts = {
    'combined_team_freq': combined_team_freq,
    'competition_bucket_columns': competition_bucket_columns,
    'feature_columns': feature_columns
}

with open('lr_preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump(preprocessing_artifacts, f)

print("Artefatos de pré-processamento salvos em 'lr_preprocessing_artifacts.pkl'")