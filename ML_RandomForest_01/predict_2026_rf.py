"""
predict_2026_rf.py

Aplica o modelo Random Forest já treinado aos jogos de 2026 e gera:

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
salvo em rf_preprocessing_artifacts.pkl (gerado pelo próprio
train_rf.py — este script não depende dos artefatos do XGBoost).

Pré-requisito: ter rodado train_rf.py antes, na mesma pasta, para gerar:
    - rf_match_result_model.pkl
    - rf_imputer.pkl
    - rf_calibrators.pkl
    - rf_preprocessing_artifacts.pkl
    - football_matches_2026.csv (na raiz do projeto)

Uso:
    python predict_2026_rf.py

-----------------------------------------------------------------
CHANGELOG (alinhado com as melhorias feitas em predict_2026.py):
1. Não depende mais do preprocessing_artifacts.pkl do XGBoost — a
   lista de features vem de rf_preprocessing_artifacts.pkl, gerado
   pelo próprio train_rf.py.
2. Removidos por completo o bucket_competition, o frequency encoding
   (home_team_freq/away_team_freq) e as flags home_team_seen/
   away_team_seen: o RF nunca usou essas colunas como feature.
3. As probabilidades passam pela calibração isotônica salva em
   rf_calibrators.pkl antes de ir para o CSV.
4. Aviso de features ausentes mostra o percentual de partidas
   afetadas, com aviso separado para NaN parcial.
5. NOVO — Cálculo completo de métricas na base 2026 (accuracy,
   F1-macro, balanced accuracy, precision/recall macro), métricas
   por classe (F1/precision/recall) e matriz de confusão, com
   tabelas formatadas no terminal e imagem PNG salva — mesmo nível
   de detalhe que o predict_2026.py já produzia. Antes, o
   predict_2026_rf.py só calculava a acurácia simples.
6. NOVO — JSON de métricas (rf_prediction_metrics.json) passou a
   seguir o mesmo esquema do prediction_metrics.json do XGBoost
   (model, year, num_matches, has_real_result, métricas gerais,
   métricas por classe, matriz de confusão), em vez de só
   num_matches/has_real_result/accuracy — necessário para comparar
   os dois modelos lado a lado na Seção 4.4 do TCC.
7. NOVO — Checagem defensiva: se feature_columns por algum motivo
   contiver colunas de frequency encoding ou de "time visto no
   treino", o script interrompe com erro, em vez de seguir em
   frente silenciosamente com um comportamento não documentado
   (mesmo princípio de segurança adotado no predict_2026.py).
-----------------------------------------------------------------
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Volta uma pasta para ler a base de 2026 na raiz do projeto
INPUT_FILE = '../football_matches_2026.csv'
MODEL_FILE = 'rf_match_result_model.pkl'
IMPUTER_FILE = 'rf_imputer.pkl'
CALIBRATORS_FILE = 'rf_calibrators.pkl'
ARTIFACTS_FILE = 'rf_preprocessing_artifacts.pkl'

OUTPUT_FILE = 'comparativo_real_vs_predito_2026_rf.csv'
METRICS_FILE = 'rf_prediction_metrics.json'
CONFUSION_MATRIX_FILE = 'confusion_matrix_rf_2026.png'

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
# 1. CARREGAR MODELO E ARTEFATOS (TUDO DO PRÓPRIO RF)
# =============================================================

print('\nCarregando modelo e artefatos...')

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

print('Modelo carregado com sucesso.')
print(f'Número de features utilizadas pelo modelo: {len(feature_columns)}')


# =============================================================
# 2. VERIFICAR SE HÁ FEATURES DE FREQUENCY ENCODING
# =============================================================
# O Random Forest nunca usou frequency encoding nem as flags de
# "time visto no treino" como feature (isso ficou só no XGBoost).
# Essa checagem é defensiva: se um dia train_rf.py voltar a incluir
# essas colunas em feature_columns, o script para em vez de seguir
# em frente silenciosamente com um comportamento não documentado.

frequency_features = {
    'home_team_freq',
    'away_team_freq',
    'home_team_seen',
    'away_team_seen'
}

frequency_features_found = frequency_features.intersection(set(feature_columns))

if frequency_features_found:
    raise ValueError(
        '\nERRO: o arquivo rf_preprocessing_artifacts.pkl contém '
        f'features de frequency encoding ou flags de equipes: '
        f'{sorted(frequency_features_found)}\n\n'
        'Este script foi configurado para NÃO utilizar frequency '
        'encoding no Random Forest.\n'
        'Verifique se train_rf.py foi alterado para incluir essas '
        'features novamente.'
    )


# =============================================================
# 3. CARREGAR BASE DE 2026
# =============================================================

print('\nCarregando dados de 2026...')

df = pd.read_csv(INPUT_FILE)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

has_real_result = TARGET_COLUMN in df.columns

print(f'Total de partidas em 2026: {len(df):,}')

if not has_real_result:
    print('\n[AVISO] A coluna match_result não está presente na base.')
    print('As previsões serão geradas, mas as métricas não poderão ser calculadas.')


# =============================================================
# 4. VERIFICAR FEATURES
# =============================================================

missing_features = [c for c in feature_columns if c not in df.columns]
if missing_features:
    print('\n[AVISO] Features esperadas pelo modelo não estão presentes na base de 2026:')
    for col in missing_features:
        df[col] = np.nan
        print(f'  - {col}: 100.0% das partidas sem esse dado')

partially_missing = [
    c for c in feature_columns
    if c not in missing_features and df[c].isna().any()
]
if partially_missing:
    print('\n[AVISO] Features com valores NaN na base de 2026:')
    for col in partially_missing:
        pct = df[col].isna().mean() * 100
        print(f'  - {col}: {pct:.1f}% das partidas')


# =============================================================
# 5. MONTAR MATRIZ X E IMPUTAR NULOS
# =============================================================
# Diferente do XGBoost, o Random Forest não trata NaN nativamente:
# usa-se o imputer (mediana do treino) salvo em rf_imputer.pkl.

X = df[feature_columns].copy()
X_imp = imputer.transform(X)

print(f'\nMatriz de predição: {X.shape[0]} partidas x {X.shape[1]} features')


# =============================================================
# 6. REALIZAR PREDIÇÃO
# =============================================================

print('\nRealizando predições...')

y_pred = model.predict(X_imp)
y_pred_proba_raw = model.predict_proba(X_imp)


# =============================================================
# 7. CALIBRAÇÃO DAS PROBABILIDADES
# =============================================================

if calibrators:
    print('Aplicando calibração das probabilidades...')

    y_pred_proba = np.zeros_like(y_pred_proba_raw)
    for class_idx, iso in calibrators.items():
        y_pred_proba[:, class_idx] = iso.predict(y_pred_proba_raw[:, class_idx])

    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_proba = y_pred_proba / row_sums
else:
    print('Nenhum calibrador encontrado. Utilizando probabilidades originais.')
    y_pred_proba = y_pred_proba_raw


# =============================================================
# 8. ADICIONAR RESULTADOS AO DATAFRAME
# =============================================================

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


# =============================================================
# 9. GERAR CSV COMPARATIVO
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
    'prob_home_win'
]

df_out = df[output_columns].copy()
df_out.to_csv(OUTPUT_FILE, index=False)

print(f'\nCSV comparativo salvo em: {OUTPUT_FILE}')


# =============================================================
# 10. MÉTRICAS
# =============================================================

if has_real_result:

    y_true = df[TARGET_COLUMN].astype(int)
    y_pred_metrics = pd.Series(y_pred).astype(int)

    # ---------------------------------------------------------
    # 10.1 MÉTRICAS GERAIS
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred_metrics)
    f1_macro = f1_score(y_true, y_pred_metrics, labels=[0, 1, 2], average='macro', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred_metrics)
    precision_macro = precision_score(y_true, y_pred_metrics, labels=[0, 1, 2], average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_metrics, labels=[0, 1, 2], average='macro', zero_division=0)

    # ---------------------------------------------------------
    # 10.2 F1-SCORE, PRECISION E RECALL POR CLASSE
    # ---------------------------------------------------------
    f1_per_class = f1_score(y_true, y_pred_metrics, labels=[0, 1, 2], average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred_metrics, labels=[0, 1, 2], average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred_metrics, labels=[0, 1, 2], average=None, zero_division=0)

    f1_away_win, f1_draw, f1_home_win = f1_per_class

    # ---------------------------------------------------------
    # 10.3 MATRIZ DE CONFUSÃO
    # ---------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred_metrics, labels=[0, 1, 2])

    # ---------------------------------------------------------
    # 10.4 EXIBIR RESULTADOS NO TERMINAL
    # ---------------------------------------------------------
    print('\n')
    print('=' * 75)
    print('RESULTADOS RANDOM FOREST - COPA DO MUNDO 2026')
    print('=' * 75)

    metrics_table = pd.DataFrame({
        'Métrica': ['Accuracy', 'F1-macro', 'Balanced Accuracy', 'Precision macro', 'Recall macro'],
        'Resultado': [accuracy, f1_macro, balanced_accuracy, precision_macro, recall_macro]
    })

    print('\nMÉTRICAS GERAIS')
    print('-' * 75)
    print(metrics_table.to_string(index=False, formatters={'Resultado': '{:.4f}'.format}))

    class_table = pd.DataFrame({
        'Classe': CLASS_NAMES,
        'F1-score': f1_per_class,
        'Precision': precision_per_class,
        'Recall': recall_per_class
    })

    print('\nMÉTRICAS POR CLASSE')
    print('-' * 75)
    print(class_table.to_string(
        index=False,
        formatters={'F1-score': '{:.4f}'.format, 'Precision': '{:.4f}'.format, 'Recall': '{:.4f}'.format}
    ))

    cm_table = pd.DataFrame(
        cm,
        index=['Real: Visitante (0)', 'Real: Empate (1)', 'Real: Mandante (2)'],
        columns=['Pred: Visitante (0)', 'Pred: Empate (1)', 'Pred: Mandante (2)']
    )

    print('\nMATRIZ DE CONFUSÃO')
    print('-' * 75)
    print(cm_table.to_string())

    # ---------------------------------------------------------
    # 10.5 GERAR IMAGEM DA MATRIZ DE CONFUSÃO
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Vitória\nVisitante', 'Empate', 'Vitória\nMandante']
    )
    disp.plot(ax=ax, cmap='Greens', values_format='d', colorbar=False)

    ax.set_title('Matriz de Confusão - Random Forest - Copa do Mundo 2026')
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\nImagem da matriz de confusão salva em: {CONFUSION_MATRIX_FILE}')

    # ---------------------------------------------------------
    # 10.6 SALVAR TODAS AS MÉTRICAS EM JSON
    # ---------------------------------------------------------
    prediction_metrics = {
        'model': 'Random Forest',
        'year': 2026,
        'num_matches': int(len(df_out)),
        'has_real_result': True,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'balanced_accuracy': float(balanced_accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_per_class': {
            'away_win_0': float(f1_away_win),
            'draw_1': float(f1_draw),
            'home_win_2': float(f1_home_win)
        },
        'precision_per_class': {
            'away_win_0': float(precision_per_class[0]),
            'draw_1': float(precision_per_class[1]),
            'home_win_2': float(precision_per_class[2])
        },
        'recall_per_class': {
            'away_win_0': float(recall_per_class[0]),
            'draw_1': float(recall_per_class[1]),
            'home_win_2': float(recall_per_class[2])
        },
        'confusion_matrix': cm.tolist()
    }

else:
    prediction_metrics = {
        'model': 'Random Forest',
        'year': 2026,
        'num_matches': int(len(df_out)),
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
# 11. SALVAR JSON
# =============================================================

with open(METRICS_FILE, 'w', encoding='utf-8') as f:
    json.dump(prediction_metrics, f, indent=2, ensure_ascii=False)

print(f'Métricas salvas em: {METRICS_FILE}')


# =============================================================
# 12. FINAL
# =============================================================

print('\n')
print('=' * 75)
print('PROCESSAMENTO CONCLUÍDO')
print('=' * 75)
print(f'CSV: {OUTPUT_FILE}')
print(f'JSON: {METRICS_FILE}')
if has_real_result:
    print(f'Matriz de confusão: {CONFUSION_MATRIX_FILE}')
print('=' * 75)