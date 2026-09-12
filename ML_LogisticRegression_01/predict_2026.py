import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


INPUT_FILE = '../football_matches_2026.csv'

MODEL_FILE = 'lr_match_result_model.pkl'
IMPUTER_FILE = 'lr_imputer.pkl'
SCALER_FILE = 'lr_scaler.pkl'
ARTIFACTS_FILE = 'preprocessing_artifacts.pkl'

OUTPUT_FILE = 'comparativo_real_vs_predito_2026_lr.csv'
METRICS_FILE = 'prediction_metrics_lr.json'
METRICS_CSV_FILE = 'prediction_metrics_2026_lr.csv'
CLASS_METRICS_FILE = 'class_metrics_2026_lr.csv'
CONFUSION_MATRIX_FILE = 'confusion_matrix_2026_lr.png'
CLASS_PERFORMANCE_FILE = 'desempenho_por_classe_2026_lr.png'

RESULT_LABELS = {
    0: 'away_win',
    1: 'draw',
    2: 'home_win'
}

TARGET_COLUMN = 'match_result'
CLASS_ORDER = [0, 1, 2]


print('Carregando modelo e artefatos da Regressão Logística...')

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

with open(IMPUTER_FILE, 'rb') as f:
    imputer = pickle.load(f)

with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)


combined_team_freq = artifacts.get('combined_team_freq')
known_teams = artifacts.get('known_teams')
competition_bucket_columns = artifacts.get(
    'competition_bucket_columns',
    []
)
feature_columns = artifacts['feature_columns']
calibrators = artifacts.get('calibrators')


df = pd.read_csv(INPUT_FILE)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )

has_real_result = TARGET_COLUMN in df.columns

print(f'Total de partidas em 2026: {len(df):,}')


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


df['competition_bucket'] = df['competition'].apply(
    bucket_competition
)


if combined_team_freq is not None:
    df['home_team_freq'] = (
        df['home_team']
        .map(combined_team_freq)
        .fillna(0.0)
    )

    df['away_team_freq'] = (
        df['away_team']
        .map(combined_team_freq)
        .fillna(0.0)
    )
else:
    df['home_team_freq'] = 0.0
    df['away_team_freq'] = 0.0


if known_teams is not None:
    df['home_team_seen'] = (
        df['home_team']
        .isin(known_teams)
        .astype(int)
    )

    df['away_team_seen'] = (
        df['away_team']
        .isin(known_teams)
        .astype(int)
    )
else:
    df['home_team_seen'] = 0
    df['away_team_seen'] = 0


num_new_home = int(
    (df['home_team_seen'] == 0).sum()
)

num_new_away = int(
    (df['away_team_seen'] == 0).sum()
)


if num_new_home or num_new_away:
    print(
        '\n[AVISO] Times não vistos no treinamento:'
    )

    print(
        f'  - Mandantes novos: {num_new_home}'
    )

    print(
        f'  - Visitantes novos: {num_new_away}'
    )

    print(
        '  Previsões envolvendo esses times podem ser '
        'menos confiáveis.'
    )


dummies = pd.get_dummies(
    df['competition_bucket'],
    prefix='comp'
)

for col in competition_bucket_columns:
    if col not in dummies.columns:
        dummies[col] = 0

if competition_bucket_columns:
    df[competition_bucket_columns] = (
        dummies[competition_bucket_columns]
    )


missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_features:
    print(
        '\n[AVISO] Features ausentes na base 2026, '
        'preenchidas com NaN:'
    )

    for col in missing_features:
        df[col] = float('nan')

        print(
            f'  - {col}: 100.0% das partidas sem esse dado'
        )


partially_missing = [
    col
    for col in feature_columns
    if col not in missing_features
    and df[col].isna().any()
]

if partially_missing:
    print(
        '\n[AVISO] Features com NaN parcial na base 2026:'
    )

    for col in partially_missing:
        pct = df[col].isna().mean() * 100

        print(
            f'  - {col}: {pct:.1f}% das partidas sem esse dado'
        )


X = df[feature_columns].copy()

X_imp = imputer.transform(X)

X_scaled = scaler.transform(X_imp)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=feature_columns,
    index=df.index
)


print('\nRealizando predições...')

y_pred = model.predict(X_scaled)

y_pred_proba_raw = model.predict_proba(X_scaled)


if calibrators:
    y_pred_proba = np.zeros_like(
        y_pred_proba_raw,
        dtype=float
    )

    for class_idx, iso in calibrators.items():
        y_pred_proba[:, class_idx] = (
            iso.predict(
                y_pred_proba_raw[:, class_idx]
            )
        )

    row_sums = y_pred_proba.sum(
        axis=1,
        keepdims=True
    )

    row_sums[row_sums == 0] = 1.0

    y_pred_proba = (
        y_pred_proba / row_sums
    )

    print(
        'Probabilidades calibradas com Isotonic Regression.'
    )
else:
    y_pred_proba = y_pred_proba_raw

    print(
        '[AVISO] Calibradores não encontrados. '
        'Usando probabilidades originais do modelo.'
    )


df['resultado_predito'] = (
    pd.Series(
        y_pred,
        index=df.index
    ).map(RESULT_LABELS)
)

df['prob_away_win'] = (
    y_pred_proba[:, 0]
)

df['prob_draw'] = (
    y_pred_proba[:, 1]
)

df['prob_home_win'] = (
    y_pred_proba[:, 2]
)


if has_real_result:
    df['resultado_real'] = (
        df[TARGET_COLUMN]
        .map(RESULT_LABELS)
    )

    df['acertou'] = (
        df['resultado_real']
        == df['resultado_predito']
    )
else:
    df['resultado_real'] = 'desconhecido'
    df['acertou'] = None

    print(
        '\n[AVISO] Coluna match_result não encontrada em 2026 — '
        'as métricas de desempenho não serão calculadas.'
    )


output_columns = []

if 'match_id' in df.columns:
    output_columns.append('match_id')

if 'date' in df.columns:
    output_columns.append('date')

output_columns += [
    'home_team',
    'away_team'
]

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

df_out.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f'\nCSV comparativo salvo em {OUTPUT_FILE}'
)

print('\nPrimeiras 10 previsões:')

print(
    df_out.head(10).to_string()
)


if has_real_result:

    y_true = df[TARGET_COLUMN].astype(int).to_numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average='macro',
        labels=CLASS_ORDER,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average='macro',
        labels=CLASS_ORDER,
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average='macro',
        labels=CLASS_ORDER,
        zero_division=0
    )

    print('\n===================================')
    print('RESULTADOS NA COPA DO MUNDO 2026')
    print('===================================')

    print(
        f'Accuracy: {accuracy:.4f}'
    )

    print(
        f'F1-macro: {f1_macro:.4f}'
    )

    print(
        f'Balanced accuracy: {balanced_accuracy:.4f}'
    )

    print(
        f'Precision macro: {precision_macro:.4f}'
    )

    print(
        f'Recall macro: {recall_macro:.4f}'
    )


    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=[
            'away_win',
            'draw',
            'home_win'
        ],
        output_dict=True,
        zero_division=0
    )


    class_metrics = []

    for class_idx, class_name in RESULT_LABELS.items():
        class_data = report_dict[class_name]

        class_metrics.append({
            'class': class_name,
            'class_id': class_idx,
            'precision': float(class_data['precision']),
            'recall': float(class_data['recall']),
            'f1_score': float(class_data['f1-score']),
            'support': int(class_data['support'])
        })


    class_metrics_df = pd.DataFrame(
        class_metrics
    )

    class_metrics_df.to_csv(
        CLASS_METRICS_FILE,
        index=False
    )


    print('\nDesempenho por classe:')

    print(
        class_metrics_df.to_string(
            index=False
        )
    )


    metrics_2026 = {
        'num_matches': int(len(df)),
        'has_real_result': True,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'balanced_accuracy': float(balanced_accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'new_home_teams': int(num_new_home),
        'new_away_teams': int(num_new_away),
        'classes': {
            class_name: {
                'precision': float(
                    report_dict[class_name]['precision']
                ),
                'recall': float(
                    report_dict[class_name]['recall']
                ),
                'f1_score': float(
                    report_dict[class_name]['f1-score']
                ),
                'support': int(
                    report_dict[class_name]['support']
                )
            }
            for class_name in RESULT_LABELS.values()
        }
    }


    metrics_df = pd.DataFrame([
        {
            'modelo': 'Regressão Logística',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'balanced_accuracy': balanced_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    ])

    metrics_df.to_csv(
        METRICS_CSV_FILE,
        index=False
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            'Vitória Visitante',
            'Empate',
            'Vitória Mandante'
        ]
    )

    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=False
    )

    ax.set_title(
        'Matriz de Confusão - Regressão Logística - Copa do Mundo 2026'
    )

    ax.set_xlabel(
        'Classe Predita'
    )

    ax.set_ylabel(
        'Classe Real'
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(len(class_metrics_df))
    width = 0.25

    ax.bar(
        x - width,
        class_metrics_df['precision'],
        width,
        label='Precisão'
    )

    ax.bar(
        x,
        class_metrics_df['recall'],
        width,
        label='Recall'
    )

    ax.bar(
        x + width,
        class_metrics_df['f1_score'],
        width,
        label='F1-score'
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        'Vitória\nVisitante',
        'Empate',
        'Vitória\nMandante'
    ])

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        'Valor'
    )

    ax.set_title(
        'Desempenho por Classe - Regressão Logística - Copa do Mundo 2026'
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        CLASS_PERFORMANCE_FILE,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


else:

    metrics_2026 = {
        'num_matches': int(len(df)),
        'has_real_result': False,
        'accuracy': None,
        'f1_macro': None,
        'balanced_accuracy': None,
        'precision_macro': None,
        'recall_macro': None,
        'new_home_teams': int(num_new_home),
        'new_away_teams': int(num_new_away)
    }


with open(
    METRICS_FILE,
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        metrics_2026,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f'\nMétricas de predição salvas em {METRICS_FILE}'
)

if has_real_result:
    print(
        f'Métricas globais salvas em {METRICS_CSV_FILE}'
    )

    print(
        f'Métricas por classe salvas em {CLASS_METRICS_FILE}'
    )

    print(
        f'Matriz de confusão salva em {CONFUSION_MATRIX_FILE}'
    )

    print(
        f'Desempenho por classe salvo em {CLASS_PERFORMANCE_FILE}'
    )

print(
    '\nPredição da Regressão Logística concluída.'
)