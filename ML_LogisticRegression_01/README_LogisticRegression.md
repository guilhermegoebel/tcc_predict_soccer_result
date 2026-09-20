# Regressão Logística para previsão de resultados de partidas

## Visão geral

Este diretório contém o pipeline de ajuste de hiperparâmetros, treinamento, avaliação, explicabilidade e inferência de um modelo de Regressão Logística multinomial para prever resultados de partidas internacionais de futebol em três classes:

- `0` = vitória visitante (`away_win`)
- `1` = empate (`draw`)
- `2` = vitória mandante (`home_win`)

A implementação está dividida em quatro scripts:

- `tune_logistic_regression_hyperparams.py` — busca de hiperparâmetros com validação temporal;
- `train_logistic_regression.py` — treinamento, calibração e avaliação do modelo;
- `rl_feature_importance_analysis.py` — análise por coeficientes, permutation importance e SHAP;
- `predict_2026.py` — aplicação do modelo treinado às partidas de 2026.

---

## Base de dados

O treinamento utiliza o arquivo localizado na raiz do projeto:

- `../football_matches_ml.csv`

A inferência de 2026 utiliza:

- `../football_matches_2026.csv`

Os arquivos devem ser produzidos pelo pipeline de preparação dos dados antes da execução dos scripts deste diretório.

### Variável alvo

A coluna `match_result` segue o mapeamento:

| Valor | Classe |
| --- | --- |
| `0` | Vitória visitante |
| `1` | Empate |
| `2` | Vitória mandante |

---

## Separação temporal

Os registros são ordenados cronologicamente e separados sem embaralhamento:

- treino: partidas realizadas até 2021;
- validação: partidas de 2022 e 2023;
- teste: partidas a partir de 2024.

A busca de hiperparâmetros utiliza os dados disponíveis até 2023 e constrói cinco partições temporais crescentes, agrupadas por datas únicas. Em cada partição, todas as partidas de treinamento são anteriores às partidas de validação.

---

## Features utilizadas

O modelo utiliza 24 variáveis numéricas:

- `home_rank`, `away_rank`;
- `home_points`, `away_points`;
- `rank_diff`, `points_diff`;
- `home_market_value_avg`, `away_market_value_avg`;
- `market_value_avg_diff`;
- `home_found_count`, `away_found_count`;
- `home_recent_win_rate`, `away_recent_win_rate`;
- `home_recent_goals_scored`, `away_recent_goals_scored`;
- `home_recent_goals_conceded`, `away_recent_goals_conceded`;
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_goal_diff`;
- `year`, `month`, `world_cup`.

São excluídas as colunas identificadoras, o placar observado e a variável alvo:

- `match_id`, `date`;
- `home_goals`, `away_goals`;
- `match_result`.

Também não são utilizadas como preditoras:

- `home_team`, `away_team`;
- `competition`, `competition_bucket`;
- `home_team_freq`, `away_team_freq`;
- `home_team_seen`, `away_team_seen`.

Assim, o modelo não cria dummies de competição nem atributos de frequência ou de seleção conhecida.

---

## Pré-processamento

O pré-processamento é ajustado exclusivamente sobre o conjunto de treino e depois reaplicado aos conjuntos posteriores:

1. valores ausentes são imputados pela mediana com `SimpleImputer`;
2. as variáveis são padronizadas com `StandardScaler`;
3. a Regressão Logística utiliza `class_weight="balanced"`;
4. a semente aleatória é fixada em `42`;
5. o limite de iterações do estimador é `5000`.

Na busca de hiperparâmetros, imputação e padronização ficam dentro de um `Pipeline`, sendo reajustadas em cada partição temporal somente com os dados de treinamento daquela partição.

---

## Busca de hiperparâmetros

O script `tune_logistic_regression_hyperparams.py` utiliza:

- `RandomizedSearchCV`;
- 40 combinações de hiperparâmetros;
- cinco partições temporais;
- `f1_macro` como métrica de seleção;
- `random_state=42`;
- `refit=False`.

O espaço de busca contempla:

- penalização L2 com solver `lbfgs`;
- penalizações L1 e L2 com solver `saga`;
- penalização elastic net com solver `saga`;
- parâmetro `C` entre `1e-3` e `1e2` em escala logarítmica;
- `l1_ratio` entre 0 e 1 para elastic net.

Os arquivos gerados são:

- `rl_best_hyperparams.json`;
- `rl_cv_results_hyperparam_search.csv`.

Na execução registrada no projeto, a configuração selecionada foi:

- `C`: `0.007359075652019385`;
- `solver`: `saga`;
- `penalty`: `elasticnet`;
- `l1_ratio`: `0.3910606075732408`.

O treinamento usa esses valores quando `rl_best_hyperparams.json` está disponível. Caso contrário, o script emite um aviso e utiliza a configuração padrão definida no código.

---

## Treinamento e calibração

O script `train_logistic_regression.py` ajusta o modelo no conjunto de treino e usa o conjunto de validação para ajustar uma calibração isotônica separada por classe.

As previsões de classe são obtidas diretamente pelo classificador. A calibração é aplicada às probabilidades utilizadas nos arquivos de saída e no cálculo do log-loss, sem substituir as classes previstas pelo modelo.

O conjunto de teste permanece fora do ajuste do modelo, do imputer, do scaler e dos calibradores.

### Artefatos principais do treinamento

- `rl_match_result_model.pkl` — modelo treinado;
- `rl_preprocessing_artifacts.pkl` — lista e ordem das features, imputer, scaler, calibradores, classes e parâmetros;
- `rl_imputer.pkl` e `rl_scaler.pkl` — objetos de pré-processamento separados;
- `rl_evaluation_metrics.json` e `rl_evaluation_metrics.csv` — métricas de validação, teste e baseline;
- `rl_confusion_matrix.png` — matriz de confusão do teste;
- `rl_resumo_executivo.png` — resumo das métricas;
- `rl_desempenho_por_classe.png` — desempenho por classe;
- `rl_feature_importance.csv` e `rl_feature_importance.png` — média dos valores absolutos dos coeficientes;
- `rl_coefficients_by_class.csv` — matriz de coeficientes por classe;
- `rl_coefficients_detailed.csv` — coeficientes no formato longo;
- `rl_coefficients_away_win.png`, `rl_coefficients_draw.png` e `rl_coefficients_home_win.png` — coeficientes de maior magnitude por classe.

---

## Métricas de avaliação

O treinamento calcula:

- accuracy;
- F1-macro;
- balanced accuracy;
- precision macro;
- recall macro;
- log-loss bruto;
- log-loss após calibração.

Também é calculado o baseline que prevê vitória do mandante em todas as partidas. Esse resultado serve como referência mínima para verificar se o classificador supera a regra baseada apenas na classe majoritária.

---

## Explicabilidade

O script `rl_feature_importance_analysis.py` reconstrói o conjunto de teste usando os artefatos salvos e verifica inicialmente as métricas do modelo.

São utilizadas três perspectivas complementares:

### Coeficientes

Os coeficientes da Regressão Logística preservam o sinal da associação com cada classe. Como as features são padronizadas, suas magnitudes podem ser comparadas dentro do modelo ajustado. Os coeficientes não devem ser interpretados como efeitos causais.

### Permutation importance

Cada feature é embaralhada 30 vezes no conjunto de teste, e a variação do F1-macro é utilizada para estimar sua contribuição preditiva.

Arquivos gerados:

- `rl_permutation_importance.csv`;
- `rl_permutation_importance.png`.

### SHAP

O `LinearExplainer` é aplicado aos escores lineares do modelo antes da calibração isotônica. A análise produz uma importância global entre as classes e um resumo direcional para `home_win`.

Arquivos gerados:

- `rl_shap_importance.csv`;
- `rl_shap_importance.png`;
- `rl_shap_summary_home_win.png`.

---

## Inferência nas partidas de 2026

O script `predict_2026.py` carrega:

- `rl_match_result_model.pkl`;
- `rl_preprocessing_artifacts.pkl`;
- `../football_matches_2026.csv`.

O script valida a presença e a ordem das features, reaplica o imputer e o scaler e gera probabilidades brutas e calibradas.

### Saídas de 2026

- `rl_comparativo_real_vs_predito_2026.csv`;
- `rl_prediction_metrics_2026.json`;
- `rl_prediction_metrics_2026.csv`;
- `rl_class_metrics_2026.csv`;
- `rl_confusion_matrix_2026.png`;
- `rl_desempenho_por_classe_2026.png`.

Quando `match_result` está disponível, o script compara as previsões com os resultados observados. Sem essa coluna, as previsões são geradas, mas as métricas supervisionadas não são calculadas.

---

## Ordem de execução

Execute os comandos a partir da raiz do repositório:

```bash
python ML_LogisticRegression_01/tune_logistic_regression_hyperparams.py
python ML_LogisticRegression_01/train_logistic_regression.py
python ML_LogisticRegression_01/rl_feature_importance_analysis.py
python ML_LogisticRegression_01/predict_2026.py
```

O ajuste de hiperparâmetros deve preceder o treinamento para que `rl_best_hyperparams.json` seja utilizado. O treinamento deve preceder a explicabilidade e a inferência, pois essas etapas dependem do modelo e dos artefatos salvos.

---

## Reprodutibilidade

O ambiente utilizado no estudo está registrado no `requirements.txt` da raiz. As operações estocásticas usam `random_state=42`, e as divisões dos dados preservam a ordem temporal.

Resultados numericamente idênticos ainda podem depender do sistema operacional, do paralelismo e das versões instaladas. Para reproduzir o ambiente, recomenda-se criar um ambiente virtual e instalar as dependências fixadas no projeto.

