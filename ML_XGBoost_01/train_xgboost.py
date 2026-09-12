"""
Treino de um modelo XGBoost para prever o resultado de partidas
(match_result: 0 = vitória visitante, 1 = empate, 2 = vitória mandante)
a partir do dataset gerado por script.py - football_matches_ml.csv

Requisitos:
    pip install xgboost scikit-learn pandas numpy matplotlib --break-system-packages

Uso:
    python train_xgboost.py

-----------------------------------------------------------------
CHANGELOG (melhorias sobre a versão anterior):
1. Calibração de probabilidade (isotonic regression por classe,
   ajustada na validação). Corrige o log-loss alto causado pelo
   sample_weight='balanced', sem mudar as previsões de classe.
2. Flag "time visto no treino" (home_team_seen / away_team_seen).
   Antes, um time nunca visto no treino e um time raro de verdade
   recebiam o mesmo valor de frequência (perto de 0) — o modelo
   não conseguia distinguir os dois casos. Agora ele consegue.
3. Comentário de rank_diff/points_diff corrigido para refletir o
   que o código realmente faz (mantém essas colunas como feature).
-----------------------------------------------------------------
"""

import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    precision_score,
    recall_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBClassifier

# =============================================================

INPUT_FILE = '../football_matches_ml.csv'

# Split temporal: nunca aleatório, porque features como h2h_* e
# recent_* dependem do histórico acumulado cronologicamente.
TRAIN_END_YEAR = 2021      # treino: até este ano (inclusive)
VAL_END_YEAR = 2023        # validação (early stopping): anos seguintes
                           # teste: tudo depois de VAL_END_YEAR

# Colunas que não entram como feature (identificadores, ou vazamento)
DROP_COLUMNS = [
    'match_id',
    'date',
    'home_goals',       # vazamento: só se sabe depois do jogo
    'away_goals',       # vazamento: idem
    'match_result'      # é o target
]
# rank_diff e points_diff são MANTIDOS como feature de propósito:
# mesmo sendo calculados a partir de home_rank/away_rank e
# home_points/away_points, uma diferença pré-calculada facilita o
# XGBoost a encontrar o split relevante direto, sem precisar
# "descobrir" a subtração através de múltiplas árvores. Não é
# vazamento (é informação pré-jogo), então mantemos.

TARGET_COLUMN = 'match_result'

# Hiperparâmetros: por padrão usamos os valores fixos abaixo (escolhidos
# manualmente). Se existir um best_hyperparams.json no mesmo diretório —
# gerado por tune_xgboost_hyperparams.py via RandomizedSearchCV com CV
# temporal (TimeSeriesSplit), sem tocar no conjunto de teste — ele é
# carregado automaticamente e sobrescreve os valores default abaixo.
DEFAULT_HYPERPARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_lambda': 1.0,
}

HYPERPARAMS_FILE = Path('best_hyperparams.json')

if HYPERPARAMS_FILE.exists():
    with open(HYPERPARAMS_FILE, 'r', encoding='utf-8') as f:
        tuned_hyperparams = json.load(f)
    model_hyperparams = {**DEFAULT_HYPERPARAMS, **tuned_hyperparams}
    print(f'Hiperparâmetros carregados de {HYPERPARAMS_FILE} '
          f'(gerados por tune_xgboost_hyperparams.py):')
    for k, v in model_hyperparams.items():
        print(f'  {k}: {v}')
else:
    model_hyperparams = DEFAULT_HYPERPARAMS
    warnings.warn(
        f'{HYPERPARAMS_FILE} não encontrado — usando hiperparâmetros fixos '
        f'default (não otimizados). Rode tune_xgboost_hyperparams.py para '
        f'gerar hiperparâmetros ajustados por RandomizedSearchCV com CV '
        f'temporal.'
    )


def export_summary_table(metrics_df: pd.DataFrame, output_path='resumo_executivo.png') -> Path:
    """Exporta a tabela dos principais resultados do relatório executivo,
    sem incluir a métrica de log-loss.
    """
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
        colColours=['#dfeaf7'] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor('#b9c7d6')
        if row == 0:
            cell.set_facecolor('#bfcfe6')
            cell.set_text_props(weight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def export_class_performance_table(class_report_df: pd.DataFrame, output_path='desempenho_por_classe.png') -> Path:
    """Exporta a tabela de desempenho por classe com precisão, recall e F1."""
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
        colColours=['#dfeaf7'] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor('#b9c7d6')
        if row == 0:
            cell.set_facecolor('#bfcfe6')
            cell.set_text_props(weight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


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
    """
    Agrupa o texto livre de 'competition' em poucas categorias
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

""" def fit_frequency_encoding(train_df, column):
    freq = train_df[column].value_counts(normalize=True)
    return freq.to_dict()


def apply_frequency_encoding(df_in, column, freq_map):
    return df_in[column].map(freq_map).fillna(0.0)


#home_team_freq = fit_frequency_encoding(df_train, 'home_team')
#away_team_freq = fit_frequency_encoding(df_train, 'away_team')


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

# Conjunto de times conhecidos no treino, para a flag "seen".
known_teams = set(combined_team_freq.keys())


def add_team_features(df_in, freq_map, known_set):
    #Adiciona frequência de time + flag binária indicando se o time
    #já apareceu no treino. A flag existe porque um time nunca visto
    #e um time raro de verdade (mas visto) recebiam o mesmo valor de
    #frequência (perto de 0) — o modelo não conseguia diferenciar
    #"raro" de "desconhecido". Isso é especialmente importante para
    #aplicar o modelo em 2026, quando seleções estreantes em Copas do
    #Mundo (sem histórico no treino) vão aparecer na base.

    df_in['home_team_freq'] = apply_frequency_encoding(
        df_in, 'home_team', freq_map
    )
    df_in['away_team_freq'] = apply_frequency_encoding(
        df_in, 'away_team', freq_map
    )
    df_in['home_team_seen'] = df_in['home_team'].isin(known_set).astype(int)
    df_in['away_team_seen'] = df_in['away_team'].isin(known_set).astype(int)
    return df_in

for split_df in (df_train, df_val, df_test):
    add_team_features(split_df, combined_team_freq, known_teams)


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
"""

# O modelo atual não usa frequency encoding nem as dummies de competition.
# Mantém o contrato do artefato explícito para a etapa de inferência.
competition_bucket_columns = []

# =============================================================
# 5. MONTAR X / y
# =============================================================

existing_drop_columns = [
    col for col in DROP_COLUMNS if col in df.columns
]

feature_columns = [
    col for col in df_train.columns
    if col not in existing_drop_columns
    and col not in ['home_team', 'away_team', 'competition', 'competition_bucket',
                    'home_team_freq',
                    'away_team_freq',
                    'home_team_seen',
                    'away_team_seen']
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
# Efeito colateral conhecido: melhora F1-macro, mas distorce as
# probabilidades (o modelo passa a "achar" empate mais provável do
# que é na realidade). Corrigimos isso na etapa 8b (calibração),
# sem mudar a decisão de classe aqui.

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
    random_state=42,
    early_stopping_rounds=30,
    **model_hyperparams,
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
# 8. AVALIAÇÃO NA VALIDAÇÃO E NO TESTE (previsões de classe)
# =============================================================

y_val_pred = model.predict(X_val)
y_val_pred_proba_raw = model.predict_proba(X_val)

y_pred = model.predict(X_test)
y_pred_proba_raw = model.predict_proba(X_test)

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
# 8b. CALIBRAÇÃO DE PROBABILIDADE (isotonic regression por classe)
# =============================================================
# O treino usou sample_weight='balanced', o que ajuda a classificar
# melhor mas deixa as probabilidades "erradas" (log-loss alto).
# Aqui ajustamos uma regressão isotônica por classe, usando SÓ o
# conjunto de validação, para reescalar a probabilidade bruta do
# modelo para a probabilidade real observada. A classe prevista
# (y_pred, usada no F1/accuracy acima) não muda — só as probabi-
# lidades que vão para o CSV final ficam mais confiáveis.

calibrators = {}
for class_idx in [0, 1, 2]:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred_proba_raw[:, class_idx], (y_val == class_idx).astype(int))
    calibrators[class_idx] = iso


def calibrate_proba(proba_raw, calibrators_dict):
    calibrated = np.zeros_like(proba_raw)
    for class_idx, iso in calibrators_dict.items():
        calibrated[:, class_idx] = iso.predict(proba_raw[:, class_idx])
    # Renormaliza cada linha para somar 1 (a calibração por classe,
    # feita separadamente, não garante isso sozinha).
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # evita divisão por zero
    return calibrated / row_sums


y_val_pred_proba = calibrate_proba(y_val_pred_proba_raw, calibrators)
y_pred_proba = calibrate_proba(y_pred_proba_raw, calibrators)

val_log_loss_raw = log_loss(y_val, y_val_pred_proba_raw, labels=[0, 1, 2])
val_log_loss = log_loss(y_val, y_val_pred_proba, labels=[0, 1, 2])
test_log_loss_raw = log_loss(y_test, y_pred_proba_raw, labels=[0, 1, 2])
test_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE VALIDAÇÃO')
print('===================================')
print(f'F1-macro: {val_f1_macro:.4f}')
print(f'Log-loss (bruto / calibrado): {val_log_loss_raw:.4f} / {val_log_loss:.4f}')
print(f'Balanced accuracy: {val_balanced_accuracy:.4f}')
print(f'Precision macro: {val_precision_macro:.4f}')
print(f'Recall macro: {val_recall_macro:.4f}')
print(f'Accuracy: {val_accuracy:.4f}')

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE')
print('===================================')
print(f'F1-macro: {test_f1_macro:.4f}')
print(f'Log-loss (bruto / calibrado): {test_log_loss_raw:.4f} / {test_log_loss:.4f}')
print(f'Balanced accuracy: {test_balanced_accuracy:.4f}')
print(f'Precision macro: {test_precision_macro:.4f}')
print(f'Recall macro: {test_recall_macro:.4f}')
print(f'Accuracy: {test_accuracy:.4f}')

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
baseline_balanced_accuracy = balanced_accuracy_score(y_test, baseline_pred)
baseline_precision_macro = precision_score(y_test, baseline_pred, average='macro', zero_division=0)
baseline_recall_macro = recall_score(y_test, baseline_pred, average='macro', zero_division=0)
baseline_accuracy = (y_test == baseline_pred).mean()
print(
    f'Baseline "sempre mandante vence" — F1-macro: {baseline_f1:.4f} '
    f'(o modelo deve superar isso claramente)'
)

# =============================================================
# 8c. RELATÓRIO DE MÉTRICAS
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
            y_val,
            y_val_pred,
            target_names=['away_win', 'draw', 'home_win'],
            output_dict=True
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
            y_test,
            y_pred,
            target_names=['away_win', 'draw', 'home_win'],
            output_dict=True
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

with open('evaluation_metrics.json', 'w', encoding='utf-8') as f:
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

metrics_df.to_csv('evaluation_metrics.csv', index=False)
print('\nRelatório de métricas salvo em evaluation_metrics.csv')
print('Relatório de métricas estruturado salvo em evaluation_metrics.json')

summary_table_path = export_summary_table(metrics_df, 'resumo_executivo.png')
print(f'Tabela principal do relatório executivo salva em {summary_table_path}')

class_report_df = pd.DataFrame(evaluation_metrics['test']['classification_report']).T
class_report_df = class_report_df.loc[['away_win', 'draw', 'home_win'], ['precision', 'recall', 'f1-score']].copy()
class_table_path = export_class_performance_table(class_report_df, 'desempenho_por_classe.png')
print(f'Tabela de desempenho por classe salva em {class_table_path}')


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
# 11. SALVAR MODELO E ARTEFATOS
# =============================================================

model.save_model('xgb_match_result.json')
print('\nModelo salvo em xgb_match_result.json')

# Artefatos de pré-processamento + calibração: precisam ser
# reaplicados exatamente assim (sem "refit") em qualquer dataset
# novo (ex.: temporada 2026) para as features e probabilidades
# baterem com o que o modelo aprendeu.
preprocessing_artifacts = {
    #'combined_team_freq': combined_team_freq,
    #'known_teams': known_teams,
    'competition_bucket_columns': competition_bucket_columns,
    'feature_columns': feature_columns,
    'calibrators': calibrators,
}

with open('preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump(preprocessing_artifacts, f)

print('Artefatos de pré-processamento salvos em preprocessing_artifacts.pkl')