"""
Treino de um modelo Random Forest para prever o resultado de partidas
(match_result: 0 = vitória visitante, 1 = empate, 2 = vitória mandante)
a partir do dataset gerado por script.py - football_matches_ml.csv

Pré-requisito:
    Ter rodado o script train_xgboost.py previamente 
    para gerar o arquivo preprocessing_artifacts.pkl.

Requisitos:
    pip install scikit-learn pandas numpy matplotlib --break-system-packages

Uso:
    python train_rf.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

# =============================================================
# CAMINHOS DOS ARQUIVOS (ajustar caso necessário)
# =============================================================
INPUT_FILE = '../football_matches_ml.csv'
ARTIFACTS_FILE = '../ML_XGBoost_01/preprocessing_artifacts.pkl'
TARGET_COLUMN = 'match_result'

# =============================================================
# 1. CARREGAR ARTEFATOS DO XGBOOST
# =============================================================
with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)

combined_team_freq = artifacts['combined_team_freq']
competition_bucket_columns = artifacts['competition_bucket_columns']
feature_columns = artifacts['feature_columns']

print("Artefatos do XGBoost carregados com sucesso.")

# =============================================================
# 2. CARREGAR E PREPARAR DADOS
# =============================================================
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

def bucket_competition(comp):
    comp = str(comp).lower()
    if 'friendl' in comp: return 'friendly'
    if 'world cup' in comp and ('qualif' not in comp): return 'world_cup_final'
    if 'qualif' in comp: return 'qualifiers'
    if 'cup' in comp or 'championship' in comp: return 'cup_or_championship'
    return 'other'

df['competition_bucket'] = df['competition'].apply(bucket_competition)

df['home_team_freq'] = df['home_team'].map(combined_team_freq).fillna(0.0)
df['away_team_freq'] = df['away_team'].map(combined_team_freq).fillna(0.0)

dummies = pd.get_dummies(df['competition_bucket'], prefix='comp')
for col in competition_bucket_columns:
    if col not in dummies.columns:
        dummies[col] = 0
df[competition_bucket_columns] = dummies[competition_bucket_columns]

# =============================================================
# 3. SPLIT TEMPORAL EXATO
# =============================================================
TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023

train_mask = df['year'] <= TRAIN_END_YEAR
val_mask = (df['year'] > TRAIN_END_YEAR) & (df['year'] <= VAL_END_YEAR)
test_mask = df['year'] > VAL_END_YEAR

df_train = df.loc[train_mask].copy()
df_val = df.loc[val_mask].copy()
df_test = df.loc[test_mask].copy()

X_train = df_train[feature_columns]
y_train = df_train[TARGET_COLUMN]

X_val = df_val[feature_columns]
y_val = df_val[TARGET_COLUMN]

X_test = df_test[feature_columns]
y_test = df_test[TARGET_COLUMN]

# =============================================================
# 4. IMPUTAÇÃO DE VALORES NULOS (Treino e Validação)
# =============================================================
# Ajustamos o imputer APENAS no treino, e aplicamos nas demais bases
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)    # Validação recebe a mediana do treino
X_test_imp = imputer.transform(X_test)  # Teste recebe a mediana do treino

# =============================================================
# 5. BUSCA DE HIPERPARÂMETROS COM SPLIT TEMPORAL (PredefinedSplit)
# =============================================================
print("\nIniciando testes com diferentes configurações de árvores...")

# Empilhamos o Treino e a Validação para passar para o otimizador
X_search = np.vstack((X_train_imp, X_val_imp))
y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

# Lista dizendo a qual conjunto cada linha pertence
# -1 = Usar para treinar a árvore
#  0 = Usar para testar/validar o resultado
test_fold = np.concatenate([
    np.full(X_train_imp.shape[0], -1), 
    np.full(X_val_imp.shape[0], 0)     
])

ps = PredefinedSplit(test_fold)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10],            
    'min_samples_leaf': [5, 10]         
}

rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=10,             
    scoring='f1_macro', 
    cv=ps,                  
    random_state=42,
    n_jobs=-1
)

# Agora o fit roda na base empilhada, mas respeitando estritamente o tempo
random_search.fit(X_search, y_search)

# --- IMPRIMINDO TODAS AS CONFIGURAÇÕES TESTADAS ---
print("\n=============================================")
print("RESULTADOS INDIVIDUAIS DE CADA CONFIGURAÇÃO")
print("=============================================")
results = random_search.cv_results_

for i in range(len(results['params'])):
    print(f"Configuração {i+1}: {results['params'][i]}")
    print(f" -> F1-Macro Médio (Base de Validação 22-23): {results['mean_test_score'][i]:.4f}\n")

print("=============================================")
print(f"MELHORES PARÂMETROS ENCONTRADOS:\n{random_search.best_params_}")
print("=============================================\n")

best_model = random_search.best_estimator_

# =============================================================
# 6. AVALIAÇÃO NO TESTE COM O MELHOR MODELO
# =============================================================
y_pred = best_model.predict(X_test_imp)
y_pred_proba = best_model.predict_proba(X_test_imp)

test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE (MELHOR RF)')
print('===================================')
print(f'F1-macro: {test_f1_macro:.4f}')
print(f'Log-loss: {test_log_loss:.4f}')

print('\nClassification report (0=away, 1=draw, 2=home):')
print(classification_report(y_test, y_pred, target_names=['away_win', 'draw', 'home_win']))

# =============================================================
# 7. SALVAR MODELO E GRÁFICOS
# =============================================================
# Salvando o modelo treinado em disco para uso futuro nas predições de 2026
with open('rf_match_result_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\nModelo Random Forest otimizado salvo em 'rf_match_result_model.pkl'")

# Salvando o Imputer treinado
with open('rf_imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("Imputer da mediana salvo em 'rf_imputer.pkl'")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['away_win', 'draw', 'home_win'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de Confusão - Melhor Random Forest')
fig.tight_layout()
fig.savefig('rf_confusion_matrix.png', dpi=150)

importances = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 8))
importances.head(20).sort_values().plot(kind='barh', ax=ax, color='green')
ax.set_title('Top 20 features mais importantes (Melhor RF)')
ax.set_xlabel('Importância')
fig.tight_layout()
fig.savefig('rf_feature_importance.png', dpi=150)

print('Gráficos gerados e salvos com sucesso!')