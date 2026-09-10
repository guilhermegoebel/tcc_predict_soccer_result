"""
predict_2026.py

Aplica o modelo XGBoost já treinado aos jogos de 2026 e gera:

1. CSV comparativo:
   - resultado real
   - resultado predito
   - probabilidades por classe

2. Métricas gerais:
   - Accuracy
   - F1-macro
   - Balanced Accuracy
   - Precision macro
   - Recall macro

3. Métricas por classe:
   - F1-score
   - Precision
   - Recall

4. Matriz de confusão:
   - exibida no terminal
   - salva como imagem PNG

5. Arquivo JSON com todas as métricas.

IMPORTANTE:
Este script NÃO utiliza frequency encoding.
Também NÃO utiliza home_team_seen ou away_team_seen.

O conjunto de features usado na predição é exatamente aquele
salvo em preprocessing_artifacts.pkl.

Pré-requisito:
    - xgb_match_result.json
    - preprocessing_artifacts.pkl
    - football_matches_2026.csv

Uso:
    python predict_2026.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =============================================================
# CONFIGURAÇÕES
# =============================================================

INPUT_FILE = 'football_matches_2026.csv'
MODEL_FILE = 'xgb_match_result.json'
ARTIFACTS_FILE = 'preprocessing_artifacts.pkl'

OUTPUT_FILE = 'comparativo_real_vs_predito_2026.csv'
METRICS_FILE = 'prediction_metrics.json'
CONFUSION_MATRIX_FILE = 'confusion_matrix_xgboost_2026.png'


# Mapeamento das classes utilizado pelo modelo
RESULT_LABELS = {
    0: 'away_win',
    1: 'draw',
    2: 'home_win'
}


# Nomes utilizados nas tabelas e na matriz de confusão
CLASS_NAMES = [
    'Vitória Visitante (0)',
    'Empate (1)',
    'Vitória Mandante (2)'
]


TARGET_COLUMN = 'match_result'


# =============================================================
# 1. CARREGAR MODELO E ARTEFATOS
# =============================================================

print('\nCarregando modelo e artefatos...')

model = XGBClassifier()
model.load_model(MODEL_FILE)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)


competition_bucket_columns = artifacts[
    'competition_bucket_columns'
]

feature_columns = artifacts[
    'feature_columns'
]

# Calibradores são opcionais para compatibilidade
# com modelos que não possuem calibração.
calibrators = artifacts.get('calibrators')


print('Modelo carregado com sucesso.')

print(
    f'Número de features utilizadas pelo modelo: '
    f'{len(feature_columns)}'
)


# =============================================================
# 2. VERIFICAR SE HÁ FEATURES DE FREQUENCY ENCODING
# =============================================================

frequency_features = {
    'home_team_freq',
    'away_team_freq',
    'home_team_seen',
    'away_team_seen'
}

frequency_features_found = (
    frequency_features.intersection(
        set(feature_columns)
    )
)

if frequency_features_found:

    raise ValueError(
        '\nERRO: O arquivo preprocessing_artifacts.pkl '
        'contém features de frequency encoding ou flags '
        f'de equipes: {sorted(frequency_features_found)}\n\n'
        'Este script foi configurado para NÃO utilizar '
        'frequency encoding.\n'
        'Verifique se o modelo final foi realmente treinado '
        'sem essas features.'
    )


# =============================================================
# 3. CARREGAR BASE DE 2026
# =============================================================

print('\nCarregando dados de 2026...')

df = pd.read_csv(INPUT_FILE)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )


has_real_result = TARGET_COLUMN in df.columns

print(
    f'Total de partidas em 2026: {len(df):,}'
)


if not has_real_result:

    print(
        '\n[AVISO] A coluna match_result não está '
        'presente na base.'
    )

    print(
        'As previsões serão geradas, mas as métricas '
        'não poderão ser calculadas.'
    )


# =============================================================
# 4. BUCKETIZAÇÃO DE COMPETITION
# =============================================================

def bucket_competition(comp):
    """
    Reproduz a categorização de competição utilizada
    no treinamento.
    """

    comp = str(comp).lower()

    if 'friendl' in comp:
        return 'friendly'

    if 'world cup' in comp and 'qualif' not in comp:
        return 'world_cup_final'

    if 'qualif' in comp:
        return 'qualifiers'

    if 'cup' in comp or 'championship' in comp:
        return 'cup_or_championship'

    return 'other'


df['competition_bucket'] = (
    df['competition']
    .apply(bucket_competition)
)


# =============================================================
# 5. CRIAR DUMMIES DE COMPETITION
# =============================================================

dummies = pd.get_dummies(
    df['competition_bucket'],
    prefix='comp'
)


# Garante que todas as colunas existentes no treinamento
# estejam presentes na base de 2026.
for col in competition_bucket_columns:

    if col not in dummies.columns:
        dummies[col] = 0


df[competition_bucket_columns] = (
    dummies[competition_bucket_columns]
)


# =============================================================
# 6. VERIFICAR FEATURES
# =============================================================

missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]


if missing_features:

    print(
        '\n[AVISO] Features esperadas pelo modelo '
        'não estão presentes na base de 2026:'
    )

    for col in missing_features:

        df[col] = np.nan

        print(
            f'  - {col}: preenchida com NaN'
        )


# Features existentes que possuem valores ausentes
partially_missing = [
    col
    for col in feature_columns
    if col not in missing_features
    and df[col].isna().any()
]


if partially_missing:

    print(
        '\n[AVISO] Features com valores NaN '
        'na base de 2026:'
    )

    for col in partially_missing:

        pct = (
            df[col].isna().mean()
            * 100
        )

        print(
            f'  - {col}: '
            f'{pct:.1f}% das partidas'
        )


# =============================================================
# 7. MONTAR MATRIZ X
# =============================================================

X = df[feature_columns].copy()


print(
    f'\nMatriz de predição: '
    f'{X.shape[0]} partidas x '
    f'{X.shape[1]} features'
)


# =============================================================
# 8. REALIZAR PREDIÇÃO
# =============================================================

print('\nRealizando predições...')

y_pred = model.predict(X)

y_pred_proba_raw = model.predict_proba(X)


# =============================================================
# 9. CALIBRAÇÃO DAS PROBABILIDADES
# =============================================================

if calibrators:

    print(
        'Aplicando calibração das probabilidades...'
    )

    y_pred_proba = np.zeros_like(
        y_pred_proba_raw
    )

    for class_idx, iso in calibrators.items():

        y_pred_proba[:, class_idx] = (
            iso.predict(
                y_pred_proba_raw[:, class_idx]
            )
        )

    # Renormaliza as probabilidades para que a soma
    # de cada linha seja igual a 1.
    row_sums = (
        y_pred_proba
        .sum(axis=1, keepdims=True)
    )

    row_sums[
        row_sums == 0
    ] = 1.0

    y_pred_proba = (
        y_pred_proba
        / row_sums
    )

else:

    print(
        'Nenhum calibrador encontrado. '
        'Utilizando probabilidades originais.'
    )

    y_pred_proba = y_pred_proba_raw


# =============================================================
# 10. ADICIONAR RESULTADOS AO DATAFRAME
# =============================================================

df['resultado_predito'] = (
    pd.Series(y_pred)
    .map(RESULT_LABELS)
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

    df['resultado_real'] = (
        'desconhecido'
    )

    df['acertou'] = None


# =============================================================
# 11. GERAR CSV COMPARATIVO
# =============================================================

output_columns = []


if 'match_id' in df.columns:
    output_columns.append(
        'match_id'
    )


if 'date' in df.columns:
    output_columns.append(
        'date'
    )


output_columns += [
    'home_team',
    'away_team'
]


if 'competition' in df.columns:
    output_columns.append(
        'competition'
    )


output_columns += [
    'resultado_real',
    'resultado_predito',
    'acertou',
    'prob_away_win',
    'prob_draw',
    'prob_home_win'
]


df_out = df[
    output_columns
].copy()


df_out.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f'\nCSV comparativo salvo em: '
    f'{OUTPUT_FILE}'
)


# =============================================================
# 12. MÉTRICAS
# =============================================================

if has_real_result:

    y_true = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    y_pred_metrics = (
        pd.Series(y_pred)
        .astype(int)
    )


    # =========================================================
    # 12.1 MÉTRICAS GERAIS
    # =========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred_metrics
    )


    f1_macro = f1_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average='macro',
        zero_division=0
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred_metrics
        )
    )


    precision_macro = precision_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average='macro',
        zero_division=0
    )


    recall_macro = recall_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average='macro',
        zero_division=0
    )


    # =========================================================
    # 12.2 F1-SCORE POR CLASSE
    # =========================================================

    f1_per_class = f1_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )


    f1_away_win = f1_per_class[0]
    f1_draw = f1_per_class[1]
    f1_home_win = f1_per_class[2]


    # =========================================================
    # 12.3 PRECISION POR CLASSE
    # =========================================================

    precision_per_class = precision_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )


    # =========================================================
    # 12.4 RECALL POR CLASSE
    # =========================================================

    recall_per_class = recall_score(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )


    # =========================================================
    # 13. MATRIZ DE CONFUSÃO
    # =========================================================

    cm = confusion_matrix(
        y_true,
        y_pred_metrics,
        labels=[0, 1, 2]
    )


    # =========================================================
    # 14. EXIBIR RESULTADOS NO TERMINAL
    # =========================================================

    print('\n')
    print('=' * 75)
    print(
        'RESULTADOS XGBOOST - COPA DO MUNDO 2026'
    )
    print('=' * 75)


    # ---------------------------------------------------------
    # Métricas gerais
    # ---------------------------------------------------------

    metrics_table = pd.DataFrame({
        'Métrica': [
            'Accuracy',
            'F1-macro',
            'Balanced Accuracy',
            'Precision macro',
            'Recall macro'
        ],
        'Resultado': [
            accuracy,
            f1_macro,
            balanced_accuracy,
            precision_macro,
            recall_macro
        ]
    })


    print('\nMÉTRICAS GERAIS')
    print('-' * 75)

    print(
        metrics_table.to_string(
            index=False,
            formatters={
                'Resultado':
                    '{:.4f}'.format
            }
        )
    )


    # ---------------------------------------------------------
    # Métricas por classe
    # ---------------------------------------------------------

    class_table = pd.DataFrame({

        'Classe': [
            'Vitória Visitante (0)',
            'Empate (1)',
            'Vitória Mandante (2)'
        ],

        'F1-score': [
            f1_away_win,
            f1_draw,
            f1_home_win
        ],

        'Precision': [
            precision_per_class[0],
            precision_per_class[1],
            precision_per_class[2]
        ],

        'Recall': [
            recall_per_class[0],
            recall_per_class[1],
            recall_per_class[2]
        ]
    })


    print('\nMÉTRICAS POR CLASSE')
    print('-' * 75)

    print(
        class_table.to_string(
            index=False,
            formatters={
                'F1-score':
                    '{:.4f}'.format,
                'Precision':
                    '{:.4f}'.format,
                'Recall':
                    '{:.4f}'.format
            }
        )
    )


    # =========================================================
    # 15. MATRIZ DE CONFUSÃO NO TERMINAL
    # =========================================================

    cm_table = pd.DataFrame(

        cm,

        index=[
            'Real: Visitante (0)',
            'Real: Empate (1)',
            'Real: Mandante (2)'
        ],

        columns=[
            'Pred: Visitante (0)',
            'Pred: Empate (1)',
            'Pred: Mandante (2)'
        ]
    )


    print('\nMATRIZ DE CONFUSÃO')
    print('-' * 75)

    print(
        cm_table.to_string()
    )


    # =========================================================
    # 16. GERAR IMAGEM DA MATRIZ DE CONFUSÃO
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            'Vitória\nVisitante',
            'Empate',
            'Vitória\nMandante'
        ]
    )


    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=False
    )


    ax.set_title(
        'Matriz de Confusão - XGBoost - '
        'Copa do Mundo 2026'
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


    print(
        f'\nImagem da matriz de confusão salva em: '
        f'{CONFUSION_MATRIX_FILE}'
    )


    # =========================================================
    # 17. SALVAR TODAS AS MÉTRICAS EM JSON
    # =========================================================

    prediction_metrics = {

        'model': 'XGBoost',

        'year': 2026,

        'num_matches': int(
            len(df_out)
        ),

        'has_real_result': True,


        # ---------------------------------------------
        # Métricas gerais
        # ---------------------------------------------

        'accuracy': float(
            accuracy
        ),

        'f1_macro': float(
            f1_macro
        ),

        'balanced_accuracy': float(
            balanced_accuracy
        ),

        'precision_macro': float(
            precision_macro
        ),

        'recall_macro': float(
            recall_macro
        ),


        # ---------------------------------------------
        # F1 por classe
        # ---------------------------------------------

        'f1_per_class': {

            'away_win_0': float(
                f1_away_win
            ),

            'draw_1': float(
                f1_draw
            ),

            'home_win_2': float(
                f1_home_win
            )
        },


        # ---------------------------------------------
        # Precision por classe
        # ---------------------------------------------

        'precision_per_class': {

            'away_win_0': float(
                precision_per_class[0]
            ),

            'draw_1': float(
                precision_per_class[1]
            ),

            'home_win_2': float(
                precision_per_class[2]
            )
        },


        # ---------------------------------------------
        # Recall por classe
        # ---------------------------------------------

        'recall_per_class': {

            'away_win_0': float(
                recall_per_class[0]
            ),

            'draw_1': float(
                recall_per_class[1]
            ),

            'home_win_2': float(
                recall_per_class[2]
            )
        },


        # ---------------------------------------------
        # Matriz de confusão
        # ---------------------------------------------

        'confusion_matrix': (
            cm.tolist()
        )
    }


else:

    # =========================================================
    # SEM RESULTADO REAL
    # =========================================================

    prediction_metrics = {

        'model': 'XGBoost',

        'year': 2026,

        'num_matches': int(
            len(df_out)
        ),

        'has_real_result': False,

        'accuracy': None,

        'f1_macro': None,

        'balanced_accuracy': None,

        'precision_macro': None,

        'recall_macro': None,

        'f1_per_class': None,

        'precision_per_class': None,

        'recall_per_class': None,

        'confusion_matrix': None
    }


# =============================================================
# 18. SALVAR JSON
# =============================================================

with open(
    METRICS_FILE,
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        prediction_metrics,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f'Métricas salvas em: '
    f'{METRICS_FILE}'
)


# =============================================================
# 19. FINAL
# =============================================================

print('\n')
print('=' * 75)
print('PROCESSAMENTO CONCLUÍDO')
print('=' * 75)

print(
    f'CSV: {OUTPUT_FILE}'
)

print(
    f'JSON: {METRICS_FILE}'
)

if has_real_result:

    print(
        f'Matriz de confusão: '
        f'{CONFUSION_MATRIX_FILE}'
    )

print('=' * 75)