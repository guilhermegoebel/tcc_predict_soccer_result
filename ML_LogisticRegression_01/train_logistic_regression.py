import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.isotonic import IsotonicRegression

INPUT_FILE = '../football_matches_ml.csv'

# Split temporal: nunca aleatório, porque features como h2h_* e
# recent_* dependem do histórico acumulado cronologicamente.
TRAIN_END_YEAR = 2021      # treino: até este ano (inclusive)
VAL_END_YEAR = 2023        # validação: anos seguintes
                            # teste: tudo depois de VAL_END_YEAR

DROP_COLUMNS = [
    'match_id',
    'date',

    # Vazamento: só conhecidos depois da partida
    'home_goals',
    'away_goals',

    # Redundantes para o modelo linear
    'rank_diff',
    'points_diff',
    'market_value_avg_diff',

    # Target
    'match_result'
]

TARGET_COLUMN = 'match_result'


# =============================================================
# FUNÇÕES AUXILIARES PARA OUTPUTS
# =============================================================

def export_summary_table(
    metrics_df: pd.DataFrame,
    output_path='resumo_executivo.png'
) -> Path:
    """
    Exporta uma tabela dos principais resultados do relatório executivo,
    sem incluir métricas de log-loss.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_df = metrics_df[
        [
            'split',
            'accuracy',
            'f1_macro',
            'balanced_accuracy',
            'precision_macro',
            'recall_macro'
        ]
    ].copy()

    table_df.columns = [
        'Conjunto',
        'Accuracy',
        'F1-macro',
        'Balanced accuracy',
        'Precision macro',
        'Recall macro'
    ]

    fig, ax = plt.subplots(
        figsize=(11, max(2.7, 1.2 + len(table_df) * 0.55))
    )

    ax.axis('off')

    cell_text = table_df.round(4).astype(str).to_numpy()

    table = ax.table(
        cellText=cell_text,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#dfeaf7'] * len(table_df.columns),
        bbox=[0, 0, 1, 1]
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


def export_class_performance_table(
    class_report_df: pd.DataFrame,
    output_path='desempenho_por_classe.png'
) -> Path:
    """
    Exporta a tabela de desempenho por classe
    com precisão, recall e F1-score.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_df = class_report_df[
        ['precision', 'recall', 'f1-score']
    ].copy().round(2)

    table_df.index = [
        {
            'away_win': 'Vitória Visitante (0)',
            'draw': 'Empate (1)',
            'home_win': 'Vitória Mandante (2)'
        }.get(idx, idx)
        for idx in table_df.index
    ]

    table_df.columns = [
        'Precisão',
        'Recall',
        'F1-score'
    ]

    fig, ax = plt.subplots(
        figsize=(8, max(2.8, 1.2 + len(table_df) * 0.7))
    )

    ax.axis('off')

    table = ax.table(
        cellText=table_df.to_numpy(),
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        loc='center',
        cellLoc='center',
        colColours=['#dfeaf7'] * len(table_df.columns),
        bbox=[0, 0, 1, 1]
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

df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
)

df = df.sort_values('date').reset_index(drop=True)

print(f'Total de partidas: {len(df):,}')

print('Distribuição do target (match_result):')
print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .sort_index() * 100
)


# =============================================================
# 2. TRATAMENTO DE COMPETITION (BUCKETIZAÇÃO)
# =============================================================

def bucket_competition(comp):
    """
    Agrupa o texto livre de 'competition' em poucas categorias
    categóricas para o modelo diferenciar o tipo de jogo.
    """

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


df['competition_bucket'] = df['competition'].apply(
    bucket_competition
)


# =============================================================
# 3. SPLIT TEMPORAL (TREINO / VALIDAÇÃO / TESTE)
# =============================================================

train_mask = df['year'] <= TRAIN_END_YEAR

val_mask = (
    (df['year'] > TRAIN_END_YEAR)
    & (df['year'] <= VAL_END_YEAR)
)

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
        'Validação ou teste ficaram vazios. '
        'Ajuste TRAIN_END_YEAR/VAL_END_YEAR '
        'para o range real de anos da base.'
    )

def fit_frequency_encoding(train_df, column):
    """
    Aprende a frequência relativa de cada valor utilizando
    exclusivamente o conjunto de treino.
    """

    freq = train_df[column].value_counts(
        normalize=True
    )

    return freq.to_dict()


def apply_frequency_encoding(
    df_in,
    column,
    freq_map
):
    """
    Aplica o frequency encoding previamente aprendido.

    Valores desconhecidos recebem 0.0.
    """

    return (
        df_in[column]
        .map(freq_map)
        .fillna(0.0)
    )


# Universo combinado de times mandantes e visitantes.
combined_team_freq = fit_frequency_encoding(
    pd.concat([
        df_train[['home_team']].rename(
            columns={'home_team': 'team'}
        ),
        df_train[['away_team']].rename(
            columns={'away_team': 'team'}
        )
    ]),
    'team'
)


# Conjunto de times conhecidos no treino.
known_teams = set(
    combined_team_freq.keys()
)


def add_team_features(
    df_in,
    freq_map,
    known_set
):
    """
    Cria as features relacionadas ao time.

    Essas features são calculadas e preservadas para manter
    o pipeline organizado, mas NÃO entram nas feature_columns
    da versão atual da Regressão Logística.
    """

    df_in['home_team_freq'] = apply_frequency_encoding(
        df_in,
        'home_team',
        freq_map
    )

    df_in['away_team_freq'] = apply_frequency_encoding(
        df_in,
        'away_team',
        freq_map
    )

    df_in['home_team_seen'] = (
        df_in['home_team']
        .isin(known_set)
        .astype(int)
    )

    df_in['away_team_seen'] = (
        df_in['away_team']
        .isin(known_set)
        .astype(int)
    )

    return df_in


for split_df in (
    df_train,
    df_val,
    df_test
):
    add_team_features(
        split_df,
        combined_team_freq,
        known_teams
    )

competition_bucket_dummies_train = pd.get_dummies(
    df_train['competition_bucket'],
    prefix='comp'
)

competition_bucket_columns = (
    competition_bucket_dummies_train.columns.tolist()
)


for split_df in (
    df_train,
    df_val,
    df_test
):

    dummies = pd.get_dummies(
        split_df['competition_bucket'],
        prefix='comp'
    )

    # Garante as mesmas colunas nos três conjuntos.
    for col in competition_bucket_columns:
        if col not in dummies.columns:
            dummies[col] = 0

    split_df[
        competition_bucket_columns
    ] = dummies[
        competition_bucket_columns
    ]

competition_bucket_columns = []


# =============================================================
# 5. MONTAR X / y
# =============================================================

existing_drop_columns = [
    col
    for col in DROP_COLUMNS
    if col in df.columns
]


feature_columns = [
    col
    for col in df_train.columns

    if col not in existing_drop_columns

    # Variáveis categóricas originais
    and col not in [
        'home_team',
        'away_team',
        'competition',
        'competition_bucket'
    ]

    # Features criadas pelo frequency encoding:
    # mantidas no DataFrame, mas NÃO utilizadas.
    and col not in [
        'home_team_freq',
        'away_team_freq',
        'home_team_seen',
        'away_team_seen'
    ]

    # Dummies de competição:
    # mantidas no DataFrame, mas NÃO utilizadas.
    and col not in competition_bucket_columns
]


print(
    f'\nFeatures usadas ({len(feature_columns)}):'
)

print(feature_columns)


X_train = df_train[feature_columns]
y_train = df_train[TARGET_COLUMN]

X_val = df_val[feature_columns]
y_val = df_val[TARGET_COLUMN]

X_test = df_test[feature_columns]
y_test = df_test[TARGET_COLUMN]

imputer = SimpleImputer(
    strategy='median'
)

X_train_imp = imputer.fit_transform(
    X_train
)

X_val_imp = imputer.transform(
    X_val
)

X_test_imp = imputer.transform(
    X_test
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imp
)

X_val_scaled = scaler.transform(
    X_val_imp
)

X_test_scaled = scaler.transform(
    X_test_imp
)


# Converte novamente para DataFrame para manter os nomes
# das features no restante do pipeline.

X_train = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_val = pd.DataFrame(
    X_val_scaled,
    columns=X_val.columns,
    index=X_val.index
)

X_test = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': ['balanced'],
    'solver': ['lbfgs', 'saga']
}


best_f1 = 0.0
best_params = None
best_model = None


for params in ParameterGrid(param_grid):

    model_candidate = LogisticRegression(
        max_iter=2000,
        random_state=42,
        **params
    )

    # Treina somente no conjunto de treino.
    model_candidate.fit(
        X_train,
        y_train
    )

    # Avalia somente na validação.
    y_val_pred_candidate = (
        model_candidate.predict(X_val)
    )

    f1_val_candidate = f1_score(
        y_val,
        y_val_pred_candidate,
        average='macro'
    )

    print(
        f'Testando {params} -> '
        f'F1-Macro Val: {f1_val_candidate:.4f}'
    )

    if f1_val_candidate > best_f1:
        best_f1 = f1_val_candidate
        best_params = params
        best_model = model_candidate


print(
    f'\n[!] Melhor combinação encontrada: '
    f'{best_params}'
)

print(
    f'[!] F1-Macro na Validação: '
    f'{best_f1:.4f}'
)


model = best_model


# =============================================================
# 9. PREVISÕES RAW - VALIDAÇÃO E TESTE
# =============================================================

y_val_pred = model.predict(
    X_val
)

y_val_pred_proba_raw = model.predict_proba(
    X_val
)


y_pred = model.predict(
    X_test
)

y_pred_proba_raw = model.predict_proba(
    X_test
)


# =============================================================
# 10. MÉTRICAS DE CLASSIFICAÇÃO
# =============================================================

val_f1_macro = f1_score(
    y_val,
    y_val_pred,
    average='macro'
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_val_pred
)

val_precision_macro = precision_score(
    y_val,
    y_val_pred,
    average='macro',
    zero_division=0
)

val_recall_macro = recall_score(
    y_val,
    y_val_pred,
    average='macro',
    zero_division=0
)

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)


test_f1_macro = f1_score(
    y_test,
    y_pred,
    average='macro'
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

test_precision_macro = precision_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=0
)

test_recall_macro = recall_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=0
)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

calibrators = {}


for class_idx in [0, 1, 2]:

    iso = IsotonicRegression(
        out_of_bounds='clip'
    )

    iso.fit(
        y_val_pred_proba_raw[:, class_idx],
        (y_val == class_idx).astype(int)
    )

    calibrators[class_idx] = iso


def calibrate_proba(
    proba_raw,
    calibrators_dict
):
    """
    Aplica a calibração isotônica por classe e
    posteriormente renormaliza as probabilidades
    para que cada linha some 1.
    """

    calibrated = np.zeros_like(
        proba_raw
    )

    for class_idx, iso in calibrators_dict.items():

        calibrated[:, class_idx] = (
            iso.predict(
                proba_raw[:, class_idx]
            )
        )

    # A calibração independente por classe não garante
    # que as três probabilidades somem exatamente 1.
    row_sums = calibrated.sum(
        axis=1,
        keepdims=True
    )

    # Evita divisão por zero.
    row_sums[row_sums == 0] = 1.0

    return calibrated / row_sums


y_val_pred_proba = calibrate_proba(
    y_val_pred_proba_raw,
    calibrators
)

y_pred_proba = calibrate_proba(
    y_pred_proba_raw,
    calibrators
)


# =============================================================
# 12. LOG-LOSS RAW E CALIBRADO
# =============================================================

val_log_loss_raw = log_loss(
    y_val,
    y_val_pred_proba_raw,
    labels=[0, 1, 2]
)

val_log_loss = log_loss(
    y_val,
    y_val_pred_proba,
    labels=[0, 1, 2]
)


test_log_loss_raw = log_loss(
    y_test,
    y_pred_proba_raw,
    labels=[0, 1, 2]
)

test_log_loss = log_loss(
    y_test,
    y_pred_proba,
    labels=[0, 1, 2]
)


# =============================================================
# 13. RESULTADOS NA VALIDAÇÃO
# =============================================================

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE VALIDAÇÃO')
print('===================================')

print(
    f'F1-macro: {val_f1_macro:.4f}'
)

print(
    f'Log-loss (bruto / calibrado): '
    f'{val_log_loss_raw:.4f} / {val_log_loss:.4f}'
)

print(
    f'Balanced accuracy: '
    f'{val_balanced_accuracy:.4f}'
)

print(
    f'Precision macro: '
    f'{val_precision_macro:.4f}'
)

print(
    f'Recall macro: '
    f'{val_recall_macro:.4f}'
)

print(
    f'Accuracy: '
    f'{val_accuracy:.4f}'
)


# =============================================================
# 14. RESULTADOS NO TESTE
# =============================================================

print('\n===================================')
print('RESULTADOS NO CONJUNTO DE TESTE')
print('===================================')

print(
    f'F1-macro: {test_f1_macro:.4f}'
)

print(
    f'Log-loss (bruto / calibrado): '
    f'{test_log_loss_raw:.4f} / {test_log_loss:.4f}'
)

print(
    f'Balanced accuracy: '
    f'{test_balanced_accuracy:.4f}'
)

print(
    f'Precision macro: '
    f'{test_precision_macro:.4f}'
)

print(
    f'Recall macro: '
    f'{test_recall_macro:.4f}'
)

print(
    f'Accuracy: '
    f'{test_accuracy:.4f}'
)


# =============================================================
# 15. CLASSIFICATION REPORT
# =============================================================

print(
    '\nClassification report '
    '(0=away, 1=draw, 2=home):'
)

test_classification_report = classification_report(
    y_test,
    y_pred,
    target_names=[
        'away_win',
        'draw',
        'home_win'
    ]
)

print(test_classification_report)


# Também calculamos o report da validação para manter
# o relatório estruturado consistente entre os dois modelos.

val_classification_report_dict = classification_report(
    y_val,
    y_val_pred,
    target_names=[
        'away_win',
        'draw',
        'home_win'
    ],
    output_dict=True
)


test_classification_report_dict = classification_report(
    y_test,
    y_pred,
    target_names=[
        'away_win',
        'draw',
        'home_win'
    ],
    output_dict=True
)


# =============================================================
# 16. BASELINE
# =============================================================
#
# Baseline:
# sempre prever vitória do mandante (classe 2).
# =============================================================

baseline_pred = np.full_like(
    y_test,
    fill_value=2
)


baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    average='macro'
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision_macro = precision_score(
    y_test,
    baseline_pred,
    average='macro',
    zero_division=0
)

baseline_recall_macro = recall_score(
    y_test,
    baseline_pred,
    average='macro',
    zero_division=0
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)


print(
    f'\nBaseline "sempre mandante vence" '
    f'— F1-macro: {baseline_f1:.4f}'
)

print(
    f'Baseline Balanced accuracy: '
    f'{baseline_balanced_accuracy:.4f}'
)

print(
    f'Baseline Precision macro: '
    f'{baseline_precision_macro:.4f}'
)

print(
    f'Baseline Recall macro: '
    f'{baseline_recall_macro:.4f}'
)

print(
    f'Baseline Accuracy: '
    f'{baseline_accuracy:.4f}'
)


# =============================================================
# 17. RELATÓRIO DE MÉTRICAS
# =============================================================

evaluation_metrics = {

    'validation': {
        'split': 'validation',

        'num_samples': int(
            len(y_val)
        ),

        'accuracy': float(
            val_accuracy
        ),

        'f1_macro': float(
            val_f1_macro
        ),

        'log_loss_raw': float(
            val_log_loss_raw
        ),

        'log_loss_calibrated': float(
            val_log_loss
        ),

        'balanced_accuracy': float(
            val_balanced_accuracy
        ),

        'precision_macro': float(
            val_precision_macro
        ),

        'recall_macro': float(
            val_recall_macro
        ),

        'classification_report': (
            val_classification_report_dict
        )
    },

    'test': {
        'split': 'test',

        'num_samples': int(
            len(y_test)
        ),

        'accuracy': float(
            test_accuracy
        ),

        'f1_macro': float(
            test_f1_macro
        ),

        'log_loss_raw': float(
            test_log_loss_raw
        ),

        'log_loss_calibrated': float(
            test_log_loss
        ),

        'balanced_accuracy': float(
            test_balanced_accuracy
        ),

        'precision_macro': float(
            test_precision_macro
        ),

        'recall_macro': float(
            test_recall_macro
        ),

        'classification_report': (
            test_classification_report_dict
        )
    },

    'baseline_test': {
        'split': 'baseline_test',

        'num_samples': int(
            len(y_test)
        ),

        'accuracy': float(
            baseline_accuracy
        ),

        'f1_macro': float(
            baseline_f1
        ),

        'log_loss_raw': None,

        'log_loss_calibrated': None,

        'balanced_accuracy': float(
            baseline_balanced_accuracy
        ),

        'precision_macro': float(
            baseline_precision_macro
        ),

        'recall_macro': float(
            baseline_recall_macro
        ),

        'classification_report': None
    }
}


# JSON
with open(
    'evaluation_metrics.json',
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        evaluation_metrics,
        f,
        indent=2,
        ensure_ascii=False
    )


# =============================================================
# 18. MÉTRICAS EM CSV
# =============================================================

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


metrics_df.to_csv(
    'evaluation_metrics.csv',
    index=False
)


print(
    '\nRelatório de métricas salvo em '
    'evaluation_metrics.csv'
)

print(
    'Relatório de métricas estruturado salvo em '
    'evaluation_metrics.json'
)


# =============================================================
# 19. TABELA EXECUTIVA
# =============================================================

summary_table_path = export_summary_table(
    metrics_df,
    'resumo_executivo.png'
)

print(
    f'Tabela principal do relatório executivo '
    f'salva em {summary_table_path}'
)


# =============================================================
# 20. DESEMPENHO POR CLASSE
# =============================================================

class_report_df = (
    pd.DataFrame(
        evaluation_metrics['test']['classification_report']
    ).T
)


class_report_df = class_report_df.loc[
    ['away_win', 'draw', 'home_win'],
    ['precision', 'recall', 'f1-score']
].copy()


class_table_path = export_class_performance_table(
    class_report_df,
    'desempenho_por_classe.png'
)


print(
    f'Tabela de desempenho por classe '
    f'salva em {class_table_path}'
)


# =============================================================
# 21. MATRIZ DE CONFUSÃO
# =============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        'away_win',
        'draw',
        'home_win'
    ]
)


fig, ax = plt.subplots(
    figsize=(6, 5)
)


disp.plot(
    ax=ax,
    cmap='Blues',
    colorbar=False
)


ax.set_title(
    'Matriz de confusão — conjunto de teste'
)

fig.tight_layout()

fig.savefig(
    'confusion_matrix.png',
    dpi=150
)

plt.close(fig)


print(
    '\nMatriz de confusão salva em '
    'confusion_matrix.png'
)


importance_values = np.mean(
    np.abs(model.coef_),
    axis=0
)


importances = pd.Series(
    importance_values,
    index=feature_columns
).sort_values(
    ascending=False
)


fig, ax = plt.subplots(
    figsize=(8, 8)
)


importances.head(20).sort_values().plot(
    kind='barh',
    ax=ax,
    color='blue'
)


ax.set_title(
    'Top 20 features mais importantes (Regressão Logística)'
)

ax.set_xlabel(
    'Importância (Peso Absoluto Médio)'
)


fig.tight_layout()

fig.savefig(
    'feature_importance.png',
    dpi=150
)

plt.close(fig)


print(
    'Importância de features salva em '
    'feature_importance.png'
)


print('\nTop 15 features:')

print(
    importances
    .head(15)
    .round(4)
    .to_string()
)

# =============================================================
# 22. COEFICIENTES INDIVIDUAIS POR CLASSE
# =============================================================

class_labels = {
    0: 'away_win',
    1: 'draw',
    2: 'home_win'
}

# Obtém os coeficientes de cada classe diretamente do modelo.
# model.classes_ informa a ordem real das classes utilizada pelo sklearn.
coefficients_df = pd.DataFrame(
    model.coef_,
    index=[
        class_labels.get(int(class_id), str(class_id))
        for class_id in model.classes_
    ],
    columns=feature_columns
)

# -------------------------------------------------------------
# CSV em formato matricial
# -------------------------------------------------------------

coefficients_df.to_csv(
    'logistic_coefficients_by_class.csv',
    encoding='utf-8'
)

print(
    '\nCoeficientes individuais por classe salvos em '
    "'logistic_coefficients_by_class.csv'"
)

# -------------------------------------------------------------
# CSV em formato detalhado
# -------------------------------------------------------------

coefficients_long_df = (
    coefficients_df
    .reset_index()
    .rename(columns={'index': 'class'})
    .melt(
        id_vars='class',
        var_name='feature',
        value_name='coefficient'
    )
)

coefficients_long_df.to_csv(
    'logistic_coefficients_detailed.csv',
    index=False,
    encoding='utf-8'
)

print(
    'Coeficientes detalhados salvos em '
    "'logistic_coefficients_detailed.csv'"
)

# -------------------------------------------------------------
# Exibição no terminal
# -------------------------------------------------------------

for class_name in coefficients_df.index:

    print('\n===================================')
    print(f'COEFICIENTES — CLASSE: {class_name}')
    print('===================================')

    class_coefficients = (
        coefficients_df.loc[class_name]
        .sort_values(key=np.abs, ascending=False)
    )

    print(
        class_coefficients
        .round(4)
        .to_string()
    )

# -------------------------------------------------------------
# Geração de uma imagem para cada classe
# -------------------------------------------------------------

for class_name in coefficients_df.index:

    class_coefficients = (
        coefficients_df.loc[class_name]
        .sort_values(key=np.abs, ascending=True)
    )

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    class_coefficients.plot(
        kind='barh',
        ax=ax,
        color='steelblue'
    )

    ax.axvline(
        0,
        color='black',
        linewidth=0.8
    )

    ax.set_title(
        f'Coeficientes das features — classe {class_name}'
    )

    ax.set_xlabel(
        'Coeficiente'
    )

    ax.set_ylabel(
        'Feature'
    )

    fig.tight_layout()

    output_filename = (
        f'coefficients_{class_name}.png'
    )

    fig.savefig(
        output_filename,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f'Imagem de coeficientes salva em '
        f'{output_filename}'
    )

# -------------------------------------------------------------
# Top coeficientes positivos e negativos
# -------------------------------------------------------------

for class_name in coefficients_df.index:

    class_coefficients = coefficients_df.loc[class_name]

    positive_coefficients = (
        class_coefficients[
            class_coefficients > 0
        ]
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    negative_coefficients = (
        class_coefficients[
            class_coefficients < 0
        ]
        .sort_values(
            ascending=True
        )
        .head(10)
    )

    print('\n===================================')
    print(
        f'TOP 10 COEFICIENTES POSITIVOS — '
        f'{class_name}'
    )
    print('===================================')

    print(
        positive_coefficients
        .round(4)
        .to_string()
    )

    print('\n===================================')
    print(
        f'TOP 10 COEFICIENTES NEGATIVOS — '
        f'{class_name}'
    )
    print('===================================')

    print(
        negative_coefficients
        .round(4)
        .to_string()
    )


# =============================================================
# 23. SALVAR MODELO
# =============================================================

with open(
    'lr_match_result_model.pkl',
    'wb'
) as f:

    pickle.dump(
        model,
        f
    )


print(
    "\nModelo Regressão Logística otimizado salvo em "
    "'lr_match_result_model.pkl'"
)


# =============================================================
# 24. SALVAR IMPUTER
# =============================================================

with open(
    'lr_imputer.pkl',
    'wb'
) as f:

    pickle.dump(
        imputer,
        f
    )


print(
    "Imputer da mediana salvo em "
    "'lr_imputer.pkl'"
)


# =============================================================
# 25. SALVAR SCALER
# =============================================================

with open(
    'lr_scaler.pkl',
    'wb'
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    "Scaler salvo em "
    "'lr_scaler.pkl'"
)

preprocessing_artifacts = {

    'combined_team_freq': combined_team_freq,

    'known_teams': known_teams,

    'competition_bucket_columns': (
        competition_bucket_columns
    ),

    'feature_columns': feature_columns,

    'calibrators': calibrators
}


with open(
    'preprocessing_artifacts.pkl',
    'wb'
) as f:

    pickle.dump(
        preprocessing_artifacts,
        f
    )


print(
    'Artefatos de pré-processamento e calibração '
    'salvos em preprocessing_artifacts.pkl'
)


# =============================================================
# FINAL
# =============================================================

print(
    '\n==================================='
)

print(
    'PIPELINE DE REGRESSÃO LOGÍSTICA '
    'FINALIZADO COM SUCESSO'
)

print(
    '==================================='
)

print('\nArquivos gerados:')

print(' - evaluation_metrics.csv')
print(' - evaluation_metrics.json')
print(' - resumo_executivo.png')
print(' - desempenho_por_classe.png')
print(' - confusion_matrix.png')
print(' - feature_importance.png')
print(' - lr_match_result_model.pkl')
print(' - lr_imputer.pkl')
print(' - lr_scaler.pkl')
print(' - preprocessing_artifacts.pkl')