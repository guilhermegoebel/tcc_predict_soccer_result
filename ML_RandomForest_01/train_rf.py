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

-----------------------------------------------------------------
CHANGELOG (alterações para manter coerência com train_xgboost.py):
1. O RF não depende mais de preprocessing_artifacts.pkl para saber
   quais colunas usar como feature. Ele replica localmente a mesma
   lógica de exclusão de colunas do train_xgboost.py (DROP_COLUMNS +
   identificadores de time/competição), então roda sozinho, sem
   precisar que train_xgboost.py tenha sido executado antes. Se o
   pickle existir, o script só faz uma checagem de sanidade: compara
   a lista calculada aqui com artifacts['feature_columns'] e avisa
   (sem travar a execução) se estiverem diferentes — isso pega
   divergência silenciosa caso alguém mude a lógica de features em
   só um dos dois scripts e esqueça do outro.
2. O modelo final do RF (usado para calibração e para o teste) agora
   é treinado APENAS em X_train, nunca em X_train+X_val. Antes,
   RandomizedSearchCV(refit=...) reajustava o "melhor modelo" no
   treino+validação combinados, o que tornaria a calibração na
   validação inválida (o modelo já teria visto esses rótulos). O
   PredefinedSplit continua sendo usado só para ESCOLHER os
   hiperparâmetros; o refit final é manual e usa só o treino,
   espelhando a disciplina do XGBoost (val nunca entra no fit).
3. Adicionada calibração de probabilidade (isotonic regression por
   classe, ajustada em X_val), igual ao passo 8b do train_xgboost.py,
   para o log-loss dos dois modelos ser comparável.
4. Métricas de validação e teste, baseline "sempre mandante vence" e
   os arquivos evaluation_metrics.json/csv agora seguem o mesmo
   esquema do train_xgboost.py (accuracy, f1_macro, log_loss bruto/
   calibrado, balanced_accuracy, precision_macro, recall_macro), para
   dar para comparar os dois modelos lado a lado.
-----------------------------------------------------------------
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

# =============================================================
# CAMINHOS DOS ARQUIVOS (ajustar caso necessário)
# =============================================================
INPUT_FILE = '../football_matches_ml.csv'
# Não é mais uma dependência obrigatória: só usado para uma checagem
# de sanidade opcional (ver passo 1). Se o arquivo não existir, o
# script segue normalmente.
ARTIFACTS_FILE = '../ML_XGBoost_01/preprocessing_artifacts.pkl'
TARGET_COLUMN = 'match_result'

TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023

# Mesma lista de DROP_COLUMNS do train_xgboost.py (identificadores e
# colunas que vazam o resultado do jogo).
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',
    'away_goals',
    'match_result'
]

# Mesmas colunas identificadoras/derivadas de time e competição que o
# train_xgboost.py exclui do conjunto de features (o frequency
# encoding e as dummies de competição estão desativados na versão
# atual do XGBoost, então essas colunas nem chegam a existir no df,
# mas a exclusão é mantida aqui por robustez/paridade).
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


def export_summary_table(metrics_df: pd.DataFrame, output_path='rf_resumo_executivo.png') -> Path:
    """Mesma tabela de resumo do train_xgboost.py, para comparação lado a lado."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_df = metrics_df[
        ['split', 'accuracy', 'f1_macro', 'balanced_accuracy', 'precision_macro', 'recall_macro']
    ].copy()
    table_df.columns = ['Conjunto', 'Accuracy', 'F1-macro', 'Balanced accuracy', 'Precision macro', 'Recall macro']

    fig, ax = plt.subplots(figsize=(11, max(2.7, 1.2 + len(table_df) * 0.55)))
    ax.axis('off')

    cell_text = table_df.round(4).astype(str).to_numpy()
    table = ax.table(
        cellText=cell_text,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#dcf0df'] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor('#b9c7d6')
        if row == 0:
            cell.set_facecolor('#bfe6c8')
            cell.set_text_props(weight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def export_class_performance_table(class_report_df: pd.DataFrame, output_path='rf_desempenho_por_classe.png') -> Path:
    """Mesma tabela de desempenho por classe do train_xgboost.py."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_df = class_report_df[['precision', 'recall', 'f1-score']].copy().round(2)
    table_df.index = [
        {'away_win': 'Vitória Visitante (0)', 'draw': 'Empate (1)', 'home_win': 'Vitória Mandante (2)'}.get(idx, idx)
        for idx in table_df.index
    ]
    table_df.columns = ['Precisão', 'Recall', 'F1-score']

    fig, ax = plt.subplots(figsize=(8, max(2.8, 1.2 + len(table_df) * 0.7)))
    ax.axis('off')

    table = ax.table(
        cellText=table_df.to_numpy(),
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        loc='center',
        cellLoc='center',
        colColours=['#dcf0df'] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor('#b9c7d6')
        if row == 0:
            cell.set_facecolor('#bfe6c8')
            cell.set_text_props(weight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def calibrate_proba(proba_raw, calibrators_dict):
    calibrated = np.zeros_like(proba_raw)
    for class_idx, iso in calibrators_dict.items():
        calibrated[:, class_idx] = iso.predict(proba_raw[:, class_idx])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return calibrated / row_sums


# =============================================================
# 1. CARREGAR DADOS E CALCULAR feature_columns LOCALMENTE
# =============================================================
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]

feature_columns = [
    col for col in df.columns
    if col not in existing_drop_columns and col not in NON_FEATURE_COLUMNS
]

print(f"Features usadas ({len(feature_columns)}): {feature_columns}")

# Checagem de sanidade opcional: se o preprocessing_artifacts.pkl do
# XGBoost existir, compara com a lista calculada aqui. Só avisa —
# nunca trava a execução — porque o RF não depende mais desse
# arquivo para rodar.
artifacts_path = Path(ARTIFACTS_FILE)
if artifacts_path.exists():
    try:
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        xgb_feature_columns = artifacts.get('feature_columns')
        if xgb_feature_columns is not None and set(xgb_feature_columns) != set(feature_columns):
            only_in_xgb = sorted(set(xgb_feature_columns) - set(feature_columns))
            only_in_rf = sorted(set(feature_columns) - set(xgb_feature_columns))
            print(
                "\n[AVISO] feature_columns do RF diverge do preprocessing_artifacts.pkl do XGBoost!"
            )
            if only_in_xgb:
                print(f"  Só no XGBoost: {only_in_xgb}")
            if only_in_rf:
                print(f"  Só no RF: {only_in_rf}")
            print("  Verifique se a lógica de exclusão de colunas está igual nos dois scripts.\n")
        else:
            print("Checagem de sanidade: feature_columns bate com preprocessing_artifacts.pkl do XGBoost.")
    except Exception as exc:
        print(f"[AVISO] Não foi possível checar preprocessing_artifacts.pkl: {exc}")
else:
    print(
        f"Aviso: {ARTIFACTS_FILE} não encontrado — pulando checagem de sanidade "
        "contra o XGBoost (não é obrigatório para rodar o RF)."
    )

# =============================================================
# 2. SPLIT TEMPORAL EXATO (idêntico ao train_xgboost.py)
# =============================================================
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
# 3. IMPUTAÇÃO DE VALORES NULOS
# =============================================================
# (o RF, ao contrário do XGBoost, não lida nativamente com NaN)
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# =============================================================
# 4. BUSCA DE HIPERPARÂMETROS (só para ESCOLHER os parâmetros;
#    o refit final é feito manualmente no passo 6, só com X_train)
# =============================================================
print("\nIniciando testes com diferentes configurações de árvores...")

X_search = np.vstack((X_train_imp, X_val_imp))
y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

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

scoring_metrics = {
    'f1_macro': 'f1_macro',
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro'
}

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=10,
    scoring=scoring_metrics,
    refit=False,  # NÃO reajustar automaticamente em treino+validação;
                  # o modelo final é treinado manualmente no passo 6,
                  # só com X_train, para manter a validação "limpa"
                  # (necessária para a calibração isotônica).
    cv=ps,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_search, y_search)

# --- IMPRIMINDO RESULTADOS ---
print("\n=============================================")
print("RESULTADOS INDIVIDUAIS DE CADA CONFIGURAÇÃO")
print("=============================================")
results = random_search.cv_results_

for i in range(len(results['params'])):
    print(f"Configuração {i+1}: {results['params'][i]}")
    print(f" -> F1-Macro (Base de Validação {TRAIN_END_YEAR + 1}-{VAL_END_YEAR}): {results['mean_test_f1_macro'][i]:.4f}\n")

best_index = int(np.argmax(results['mean_test_f1_macro']))
best_params = results['params'][best_index]

print("=============================================")
print(f"MELHORES PARÂMETROS ENCONTRADOS:\n{best_params}")
print("=============================================")

# =============================================================
# 5. REFIT MANUAL DO MELHOR MODELO — SÓ EM X_TRAIN
# =============================================================
# Diferente do refit automático do RandomizedSearchCV (que juntaria
# treino+validação), aqui treinamos só com X_train, exatamente como
# o XGBoost faz (early stopping usa a validação sem incluí-la no
# ajuste dos pesos). Isso deixa X_val livre para calibrar as
# probabilidades sem vazamento.
best_model = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train_imp, y_train)

# =============================================================
# 6. AVALIAÇÃO EM VALIDAÇÃO E TESTE (previsões de classe)
# =============================================================
y_val_pred = best_model.predict(X_val_imp)
y_val_pred_proba_raw = best_model.predict_proba(X_val_imp)

y_pred = best_model.predict(X_test_imp)
y_pred_proba_raw = best_model.predict_proba(X_test_imp)

val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
val_precision_macro = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
val_recall_macro = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
val_accuracy = (y_val == y_val_pred).mean()

test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
test_precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
test_recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
test_accuracy = (y_test == y_pred).mean()

# =============================================================
# 6b. CALIBRAÇÃO DE PROBABILIDADE (isotonic regression por classe)
# =============================================================
# Igual ao passo 8b do train_xgboost.py: ajustada só com a validação
# (que aqui NÃO entrou no treino do best_model, ver passo 6), para
# o log-loss reportado ser comparável entre os dois modelos.
calibrators = {}
for class_idx in [0, 1, 2]:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred_proba_raw[:, class_idx], (y_val == class_idx).astype(int))
    calibrators[class_idx] = iso

y_val_pred_proba = calibrate_proba(y_val_pred_proba_raw, calibrators)
y_pred_proba = calibrate_proba(y_pred_proba_raw, calibrators)

val_log_loss_raw = log_loss(y_val, y_val_pred_proba_raw, labels=[0, 1, 2])
val_log_loss = log_loss(y_val, y_val_pred_proba, labels=[0, 1, 2])
test_log_loss_raw = log_loss(y_test, y_pred_proba_raw, labels=[0, 1, 2])
test_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE VALIDAÇÃO (MELHOR RF)')
print('===================================')
print(f'F1-macro: {val_f1_macro:.4f}')
print(f'Log-loss (bruto / calibrado): {val_log_loss_raw:.4f} / {val_log_loss:.4f}')
print(f'Balanced accuracy: {val_balanced_accuracy:.4f}')
print(f'Precision macro: {val_precision_macro:.4f}')
print(f'Recall macro: {val_recall_macro:.4f}')
print(f'Accuracy: {val_accuracy:.4f}')

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE (MELHOR RF)')
print('===================================')
print(f'F1-macro: {test_f1_macro:.4f}')
print(f'Log-loss (bruto / calibrado): {test_log_loss_raw:.4f} / {test_log_loss:.4f}')
print(f'Balanced accuracy: {test_balanced_accuracy:.4f}')
print(f'Precision macro: {test_precision_macro:.4f}')
print(f'Recall macro: {test_recall_macro:.4f}')
print(f'Accuracy: {test_accuracy:.4f}')

print('\nClassification report (0=away, 1=draw, 2=home):')
print(classification_report(y_test, y_pred, target_names=['away_win', 'draw', 'home_win']))

# Mesmo baseline do train_xgboost.py, para as duas comparações
# usarem a mesma régua.
baseline_pred = np.full_like(y_test, fill_value=2)
baseline_f1 = f1_score(y_test, baseline_pred, average='macro')
baseline_balanced_accuracy = balanced_accuracy_score(y_test, baseline_pred)
baseline_precision_macro = precision_score(y_test, baseline_pred, average='macro', zero_division=0)
baseline_recall_macro = recall_score(y_test, baseline_pred, average='macro', zero_division=0)
baseline_accuracy = (y_test == baseline_pred).mean()
print(
    f'Baseline "sempre mandante vence" — F1-macro: {baseline_f1:.4f} '
    f'(o modelo deve superar isso claramente)'
)

# =============================================================
# 6c. RELATÓRIO DE MÉTRICAS (mesmo esquema do train_xgboost.py)
# =============================================================
evaluation_metrics = {
    'validation': {
        'split': 'validation',
        'num_samples': int(len(y_val)),
        'accuracy': float(val_accuracy),
        'f1_macro': float(val_f1_macro),
        'log_loss_raw': float(val_log_loss_raw),
        'log_loss_calibrated': float(val_log_loss),
        'balanced_accuracy': float(val_balanced_accuracy),
        'precision_macro': float(val_precision_macro),
        'recall_macro': float(val_recall_macro),
        'classification_report': classification_report(
            y_val, y_val_pred, target_names=['away_win', 'draw', 'home_win'], output_dict=True
        )
    },
    'test': {
        'split': 'test',
        'num_samples': int(len(y_test)),
        'accuracy': float(test_accuracy),
        'f1_macro': float(test_f1_macro),
        'log_loss_raw': float(test_log_loss_raw),
        'log_loss_calibrated': float(test_log_loss),
        'balanced_accuracy': float(test_balanced_accuracy),
        'precision_macro': float(test_precision_macro),
        'recall_macro': float(test_recall_macro),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['away_win', 'draw', 'home_win'], output_dict=True
        )
    },
    'baseline_test': {
        'split': 'baseline_test',
        'num_samples': int(len(y_test)),
        'accuracy': float(baseline_accuracy),
        'f1_macro': float(baseline_f1),
        'log_loss_raw': None,
        'log_loss_calibrated': None,
        'balanced_accuracy': float(baseline_balanced_accuracy),
        'precision_macro': float(baseline_precision_macro),
        'recall_macro': float(baseline_recall_macro),
        'classification_report': None
    }
}

with open('rf_evaluation_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_metrics, f, indent=2, ensure_ascii=False)

metrics_df = pd.DataFrame([
    {
        'split': row['split'],
        'num_samples': row['num_samples'],
        'accuracy': row['accuracy'],
        'f1_macro': row['f1_macro'],
        'log_loss_raw': row['log_loss_raw'],
        'log_loss_calibrated': row['log_loss_calibrated'],
        'balanced_accuracy': row['balanced_accuracy'],
        'precision_macro': row['precision_macro'],
        'recall_macro': row['recall_macro']
    }
    for row in evaluation_metrics.values()
])
metrics_df.to_csv('rf_evaluation_metrics.csv', index=False)
print('\nRelatório de métricas salvo em rf_evaluation_metrics.csv')
print('Relatório de métricas estruturado salvo em rf_evaluation_metrics.json')

summary_table_path = export_summary_table(metrics_df, 'rf_resumo_executivo.png')
print(f'Tabela principal do relatório executivo salva em {summary_table_path}')

class_report_df = pd.DataFrame(evaluation_metrics['test']['classification_report']).T
class_report_df = class_report_df.loc[['away_win', 'draw', 'home_win'], ['precision', 'recall', 'f1-score']].copy()
class_table_path = export_class_performance_table(class_report_df, 'rf_desempenho_por_classe.png')
print(f'Tabela de desempenho por classe salva em {class_table_path}')

# =============================================================
# 7. SALVAR MODELO, IMPUTER E CALIBRADORES
# =============================================================
with open('rf_match_result_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\nModelo Random Forest otimizado salvo em 'rf_match_result_model.pkl'")

with open('rf_imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("Imputer da mediana salvo em 'rf_imputer.pkl'")

with open('rf_calibrators.pkl', 'wb') as f:
    pickle.dump(calibrators, f)
print("Calibradores isotônicos salvos em 'rf_calibrators.pkl'")

# feature_columns precisa ser persistido: como o RF agora calcula essa
# lista localmente (não depende mais do preprocessing_artifacts.pkl do
# XGBoost), o script de predição em 2026 precisa receber essa mesma
# lista/ordem de algum lugar — não dá para recalculá-la a partir da
# base de 2026, que não tem match_result/home_goals/away_goals.
with open('rf_preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump({'feature_columns': feature_columns}, f)
print("feature_columns do RF salvo em 'rf_preprocessing_artifacts.pkl'")

# =============================================================
# 8. MATRIZ DE CONFUSÃO
# =============================================================
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['away_win', 'draw', 'home_win'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de Confusão - Melhor Random Forest')
fig.tight_layout()
fig.savefig('rf_confusion_matrix.png', dpi=150)
print('Matriz de confusão salva em rf_confusion_matrix.png')

# =============================================================
# 9. IMPORTÂNCIA DAS FEATURES
# =============================================================
importances = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 8))
importances.head(20).sort_values().plot(kind='barh', ax=ax, color='green')
ax.set_title('Top 20 features mais importantes (Melhor RF)')
ax.set_xlabel('Importância')
fig.tight_layout()
fig.savefig('rf_feature_importance.png', dpi=150)
print('Importância de features salva em rf_feature_importance.png')

print('\nTop 15 features:')
print(importances.head(15).round(4).to_string())

print('\nGráficos e relatórios gerados e salvos com sucesso!')