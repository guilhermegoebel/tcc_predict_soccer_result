"""
Avaliação de importância de features do modelo Random Forest já
treinado, usando dois métodos AGNÓSTICOS ao modelo (calculados sobre
o comportamento do modelo já treinado em dados de teste, não sobre a
estrutura interna das árvores):

1. Permutation Importance (sklearn)
   Embaralha os valores de uma feature por vez no conjunto de teste e
   mede o quanto a métrica de desempenho (F1-macro) piora. Quanto
   maior a queda, mais o modelo realmente depende daquela feature
   para prever corretamente — mede impacto na predição, não só
   frequência/estrutura de uso nas árvores.

2. SHAP (SHapley Additive exPlanations)
   Atribui a cada feature, para cada previsão individual, uma
   contribuição baseada em valores de Shapley (teoria dos jogos), que
   somam exatamente à diferença entre a predição do modelo naquela
   instância e a predição média do modelo. Mais robusto a variáveis
   correlacionadas e ao viés de cardinalidade do que a importância
   nativa (mean decrease in impurity) do Random Forest, e ainda
   permite ver a DIREÇÃO do efeito de cada variável.

Por que complementar a importância nativa (usada em train_rf.py,
rf_feature_importance.png) com esses métodos: a importância nativa do
Random Forest é calculada olhando só para a estrutura das árvores já
construídas durante o treino, o que a torna enviesada a favor de
variáveis contínuas/de alta cardinalidade (mais oportunidades de
split) e sensível a correlação entre features. Permutation importance
e SHAP avaliam o efeito real da feature no desempenho/predição do
modelo já treinado, em dados fora do treino.

Espelha feature_importance_analysis.py (XGBoost), com 3 diferenças
justificadas pela natureza do Random Forest:
1. Carrega o modelo via pickle (rf_match_result_model.pkl), não JSON.
2. Aplica o imputer de mediana (rf_imputer.pkl) sobre X_test antes de
   qualquer coisa — diferente do XGBoost, o RF não lida nativamente
   com NaN, então permutation importance e SHAP precisam rodar sobre
   os dados já imputados (os mesmos valores que o modelo de fato usa
   para prever).
3. Não recalcula competition_bucket — o RF nunca usou essa feature
   (frequency encoding e dummies de competição ficaram só no
   XGBoost, ver changelog de train_rf.py).

Requisitos:
    pip install shap --break-system-packages
    (scikit-learn, pandas, numpy, matplotlib já são dependências
    do train_rf.py)

Uso:
    python feature_importance_analysis_rf.py

Pré-requisito: já ter rodado train_rf.py neste diretório, pois este
script carrega os artefatos gerados por ele:
    - rf_match_result_model.pkl      (modelo treinado)
    - rf_imputer.pkl                 (imputer de mediana do treino)
    - rf_preprocessing_artifacts.pkl (feature_columns usadas no treino)

Saída:
    permutation_importance_rf.png
    shap_importance_rf.png
    shap_summary_home_win_rf.png   (direção do efeito, classe "vitória mandante")
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score

# =============================================================
# Config (mesmos valores de train_rf.py, para reproduzir o mesmo
# split temporal e as mesmas partidas de teste)
# =============================================================

INPUT_FILE = '../football_matches_ml.csv'
TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023
TARGET_COLUMN = 'match_result'

MODEL_FILE = Path('rf_match_result_model.pkl')
IMPUTER_FILE = Path('rf_imputer.pkl')
ARTIFACTS_FILE = Path('rf_preprocessing_artifacts.pkl')

N_REPEATS = 30          # repetições do embaralhamento por feature (permutation importance)
RANDOM_STATE = 42
TOP_N = 20              # quantas features mostrar nos gráficos
CLASS_NAMES = ['away_win', 'draw', 'home_win']


# =============================================================
# 1. CARREGAR MODELO E ARTEFATOS JÁ TREINADOS
# =============================================================

if not MODEL_FILE.exists() or not IMPUTER_FILE.exists() or not ARTIFACTS_FILE.exists():
    raise FileNotFoundError(
        f'{MODEL_FILE}, {IMPUTER_FILE} e/ou {ARTIFACTS_FILE} não encontrados. '
        f'Rode train_rf.py primeiro neste mesmo diretório.'
    )

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

with open(IMPUTER_FILE, 'rb') as f:
    imputer = pickle.load(f)

with open(ARTIFACTS_FILE, 'rb') as f:
    artifacts = pickle.load(f)

feature_columns = artifacts['feature_columns']
print(f'Modelo e artefatos carregados. {len(feature_columns)} features.')


# =============================================================
# 2. RECONSTRUIR O CONJUNTO DE TESTE (idêntico ao train_rf.py)
# =============================================================
# Reconstruímos aqui em vez de importar train_rf.py porque esse
# script roda o treino inteiro como efeito colateral do import — não
# é reaproveitável como módulo. Mantemos a lógica idêntica para o
# X_test bater exatamente com o que o modelo já viu na avaliação
# original. Diferente do XGBoost, aplicamos o imputer de mediana logo
# em seguida — permutation importance e SHAP precisam ver os dados
# exatamente como o modelo os recebeu no treino/teste (sem NaN).

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

test_mask = df['year'] > VAL_END_YEAR
df_test = df.loc[test_mask].copy()

X_test_raw = df_test[feature_columns]
y_test = df_test[TARGET_COLUMN]

X_test_imp = imputer.transform(X_test_raw)
X_test = pd.DataFrame(X_test_imp, columns=feature_columns, index=X_test_raw.index)

print(f'Conjunto de teste reconstruído: {len(X_test):,} partidas '
      f'(a partir de {VAL_END_YEAR + 1})')


# =============================================================
# 3. PERMUTATION IMPORTANCE
# =============================================================
# Métrica igual à usada para selecionar hiperparâmetros e reportar
# desempenho no restante do trabalho (F1-macro), para manter a
# análise de importância consistente com o resto do texto.

f1_macro_scorer = make_scorer(f1_score, average='macro')

print(f'\nCalculando permutation importance ({N_REPEATS} repetições '
      f'por feature, isso pode demorar)...')

perm_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring=f1_macro_scorer,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

perm_importances = pd.Series(
    perm_result.importances_mean, index=feature_columns
).sort_values(ascending=False)
perm_std = pd.Series(
    perm_result.importances_std, index=feature_columns
)

fig, ax = plt.subplots(figsize=(8, 8))
top_perm = perm_importances.head(TOP_N).sort_values()
ax.barh(
    top_perm.index, top_perm.values,
    xerr=perm_std.loc[top_perm.index].values,
    color='#2e7d32', ecolor='#a5a5a5', capsize=3,
)
ax.set_title(f'Top {TOP_N} features mais importantes (Permutation Importance) - Random Forest')
ax.set_xlabel('Queda média no F1-macro ao embaralhar a feature')
fig.tight_layout()
fig.savefig('permutation_importance_rf.png', dpi=150)
print('Permutation importance salva em permutation_importance_rf.png')

print(f'\nTop 15 features (permutation importance, queda de F1-macro):')
print(perm_importances.head(15).round(4).to_string())


# =============================================================
# 4. SHAP (TreeExplainer)
# =============================================================

try:
    import shap
except ImportError as exc:
    raise ImportError(
        'Biblioteca shap não instalada. Rode: '
        'pip install shap --break-system-packages'
    ) from exc

print('\nCalculando valores SHAP (TreeExplainer)...')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Diferentes versões da lib retornam formatos distintos para
# multiclasse: lista de arrays (um por classe) OU um único array 3D
# (n_amostras, n_features, n_classes). Normalizamos para uma lista.
if isinstance(shap_values, list):
    shap_values_per_class = shap_values
else:
    shap_values_per_class = [
        shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[-1])
    ]

# Importância global = média de |SHAP| por feature, promediada entre
# as três classes (equivalente ao que o summary_plot faz por trás).
mean_abs_shap_per_class = np.stack([
    np.abs(class_values).mean(axis=0) for class_values in shap_values_per_class
])
global_shap_importance = pd.Series(
    mean_abs_shap_per_class.mean(axis=0), index=feature_columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
top_shap = global_shap_importance.head(TOP_N).sort_values()
ax.barh(top_shap.index, top_shap.values, color='#1565c0')
ax.set_title(f'Top {TOP_N} features mais importantes (SHAP, média |valor| entre classes) - Random Forest')
ax.set_xlabel('Média de |valor SHAP|')
fig.tight_layout()
fig.savefig('shap_importance_rf.png', dpi=150)
print('Importância SHAP salva em shap_importance_rf.png')

print(f'\nTop 15 features (SHAP, média |valor| entre as 3 classes):')
print(global_shap_importance.head(15).round(4).to_string())

# Gráfico de resumo (direção do efeito) para a classe "vitória
# mandante" (índice 2) — mostra não só a magnitude, mas se valores
# altos/baixos da feature empurram a predição para essa classe.
home_win_idx = CLASS_NAMES.index('home_win')
fig = plt.figure(figsize=(9, 8))
shap.summary_plot(
    shap_values_per_class[home_win_idx],
    X_test,
    feature_names=feature_columns,
    show=False,
    max_display=TOP_N,
)
plt.title('SHAP summary — classe "vitória mandante" (direção do efeito) - Random Forest')
plt.tight_layout()
plt.savefig('shap_summary_home_win_rf.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Resumo SHAP (direção do efeito) salvo em shap_summary_home_win_rf.png')

print(
    '\nConcluído. Compare estes rankings com o gráfico de "importância nativa" '
    'gerado por train_rf.py (rf_feature_importance.png): se a ordem das '
    'variáveis principais se mantiver estável entre os três métodos, '
    'isso reforça a confiabilidade da importância nativa reportada.'
)