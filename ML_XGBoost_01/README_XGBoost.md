# XGBoost para previsão de resultados de partidas

## Visão geral

Este diretório concentra o pipeline de treinamento, validação, explicabilidade e inferência de um modelo de classificação multinomial baseado em XGBoost para prever o resultado de partidas de futebol em três classes:

- 0 = vitória visitante
- 1 = empate
- 2 = vitória mandante

O objetivo do modelo é estimar, a partir de dados históricos, a probabilidade de cada resultado possível antes da partida, com foco em uso robusto para análise de desempenho esportivo, comparação de cenários e suporte à interpretação do fator que mais influencia o resultado.

A implementação atual está organizada em quatro blocos principais:

- `train_xgboost.py` — treinamento e validação do modelo
- `tune_hyperparams.py` — busca de hiperparâmetros via RandomizedSearchCV com validação temporal
- `predict_2026.py` — aplicação do modelo em dados de 2026
- `feature_importance_analysis.py` — análise de importância por SHAP e permutation importance

---

## Objetivo do modelo

O modelo classifica partidas em:

- vitória do time visitante
- empate
- vitória do time mandante

A classificação é feita utilizando informações pré-jogo, como:

- histórico recente de resultados
- diferença de ranking
- diferença de pontos
- valor de mercado médio
- desempenho em confrontos diretos
- contexto de competição
- ano e mês da partida

A intenção é produzir um modelo útil para previsão e também para análise interpretativa, permitindo responder perguntas como:

- Quais variáveis influenciam mais o resultado?
- A diferença de ranking afeta mais do que o histórico recente?
- O modelo descobre o viés do mandante mesmo sem depender apenas da distribuição marginal?

---

## Base de dados

A base principal é o arquivo:

- `../football_matches_ml.csv`

Esse dataset é gerado pelo fluxo principal do projeto e já contém as variáveis derivadas necessárias para o treinamento. O modelo nunca deve ser treinado em dados que ainda não tenham sido convertidos para o formato de features do pipeline, e a ordem temporal deve ser preservada para evitar vazamento de informação.

### Colunas importantes da base

Algumas variáveis centrais do modelo incluem:

- `home_team`, `away_team`
- `competition`
- `year`, `month`
- `home_rank`, `away_rank`
- `home_points`, `away_points`
- `home_market_value_avg`, `away_market_value_avg`
- `home_recent_goals_scored`, `away_recent_goals_scored`
- `home_recent_goals_conceded`, `away_recent_goals_conceded`
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`
- `rank_diff`, `points_diff`
- `match_result`

A variável alvo é:

- `match_result` com mapeamento:
  - 0 = away_win
  - 1 = draw
  - 2 = home_win

---

## Estratégia de pré-processamento

### 1. Separação temporal

O modelo usa divisão temporal e não aleatória:

- treino: até `TRAIN_END_YEAR`
- validação: entre `TRAIN_END_YEAR + 1` e `VAL_END_YEAR`
- teste: depois de `VAL_END_YEAR`

Essa convenção é essencial porque variáveis como histórico recente, H2H e forma recente dependem da ordem cronológica dos jogos. O uso de validação aleatória pode causar vazamento temporal ao permitir que o modelo veja o futuro em dados de treino.

### 2. Peso por classe para corrigir viés do mandante

O conjunto de dados apresenta forte tendência para vitórias do mandante. Para mitigar esse efeito, o treino usa `sample_weight` calculado com `class_weight='balanced'`.

Esse ajuste:

- reduz o favorecimento artificial da classe mais frequente
- melhora o desempenho em classes raras como empate e vitória visitante
- é especialmente importante para classificação multinomial com classes desbalanceadas

### 3. Early stopping

O modelo é treinado com `early_stopping_rounds=30` e monitoramento do conjunto de validação. Isso reduz o risco de overfitting e permite parar a otimização assim que a performance deixa de melhorar.

### 4. Calibração de probabilidades

Mesmo com pesos por classe, as probabilidades brutas do XGBoost podem ficar menos confiáveis quando o modelo é treinado com sample weights. Por isso, o pipeline inclui calibração por classe usando `IsotonicRegression`, ajustada apenas no conjunto de validação.

Essa etapa:

- não altera a decisão de classe final
- melhora a qualidade das probabilidades emitidas
- reduz distorções no `log_loss` calibrado

---

## Modelo e hiperparâmetros

O modelo principal é o `XGBClassifier` configurado com abordagem multinomial:

- `objective='multi:softprob'`
- `num_class=3`
- `eval_metric='mlogloss'`
- `tree_method='hist'`
- `random_state=42`
- `early_stopping_rounds=30`

Os hiperparâmetros usados estão definidos em `train_xgboost.py` e podem ser sobrescritos automaticamente por `best_hyperparams.json`, quando gerado por `tune_hyperparams.py`.

### Hiperparâmetros padrão

- `n_estimators`: 500
- `learning_rate`: 0.05
- `max_depth`: 5
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 5
- `reg_lambda`: 1.0

### Busca de hiperparâmetros

A busca é realizada em `tune_hyperparams.py` com:

- `RandomizedSearchCV`
- `TimeSeriesSplit` agrupado por data
- métrica `f1_macro`
- validação temporal para evitar contaminação por vazamento

Esse processo gera:

- `best_hyperparams.json`
- `cv_results_hyperparam_search.csv`

---

## Features usadas no modelo

As colunas identificadoras e vazadas são removidas antes do treino:

- `match_id`
- `date`
- `home_goals`
- `away_goals`
- `match_result`

A lista final das features inclui variáveis de:

- ranking e pontos
- valor de mercado
- recente desempenho
- histórico direto
- contexto de competição
- informação temporal
- diferenciais calculados antes do jogo

---

## Métricas de avaliação

O treinamento gera avaliação em três conjuntos:

- treino
- validação
- teste

As principais métricas calculadas são:

- `accuracy`
- `f1_macro`
- `balanced_accuracy`
- `precision_macro`
- `recall_macro`
- `log_loss` bruto e calibrado

Além disso, o pipeline compara com um baseline simples:

- prever sempre vitória do mandante

Esse baseline serve para verificar se o modelo realmente aprendeu algo além do viés dominante da distribuição dos resultados.

### Artefatos de avaliação gerados

- `evaluation_metrics.json`
- `evaluation_metrics.csv`
- `resumo_executivo.png`
- `desempenho_por_classe.png`
- `confusion_matrix.png`

---

## Explicabilidade do modelo

O diretório também inclui análise de explicabilidade para responder o que o modelo está usando na decisão. Existem duas abordagens complementares:

### 1. Permutation importance

Testa o impacto de cada feature ao embaralhar seus valores no conjunto de teste e medir a perda de desempenho. É útil para responder:

- qual feature mais afeta o F1-macro?
- a remoção de uma variável compromete muito a qualidade do modelo?

### 2. SHAP

O SHAP explica a predição de cada instância em termos de contribuição de cada feature. Isso permite:

- entender sinais positivos e negativos para cada classe
- ver a direção do efeito da feature
- analisar se a variável empurra a previsão para vitória do mandante, empate ou vitória visitante

Artefatos produzidos:

- `permutation_importance.png`
- `shap_importance.png`
- `shap_summary_home_win.png`

A combinação de SHAP + permutation importance é recomendada porque:

- a importância nativa do XGBoost mede uso dentro da árvore
- SHAP mede impacto real da feature na previsão
- permutation importance mede perda de desempenho ao remover o sinal da feature

---

## Inferência em 2026

O script `predict_2026.py` aplica o modelo treinado ao arquivo de 2026.

### Objetivo

Carregar:

- `xgb_match_result.json`
- `preprocessing_artifacts.pkl`

E gerar:

- probabilidades por classe
- resultado previsto
- comparação com o valor real
- métricas de desempenho
- matriz de confusão

### Arquivo de saída principal

- `comparativo_real_vs_predito_2026.csv`

Também são produzidos:

- `prediction_metrics.json`
- `confusion_matrix_xgboost_2026.png`

---

## Fluxo recomendado para retrain

Para treinar novamente o modelo de forma consistente, siga este fluxo:

1. Verificar que a base `football_matches_ml.csv` esteja atualizada.
2. Valer a integridade das colunas necessárias e o alvo `match_result`.
3. Ajustar os limites temporais de treino/validação/teste, se necessário.
4. Executar `tune_hyperparams.py` para otimizar os parâmetros do modelo.
5. Executar `train_xgboost.py` para treinar o modelo final com early stopping.
6. Validar métricas em `evaluation_metrics.json`.
7. Rodar `feature_importance_analysis.py` para interpretar o comportamento do modelo.
8. Executar `predict_2026.py` para inferência em dados novos.

---

## Conclusão

Este pipeline foi desenhado para equilibrar três objetivos principais:

1. desempenho preditivo robusto
2. consistência temporal e minimização de vazamento
3. explicabilidade e rastreabilidade do modelo

O XGBoost é uma escolha adequada para esse problema por combinar alta capacidade preditiva, boa resposta a padrões tabulares e suporte à interpretação por métodos como SHAP. A qualidade do modelo, no entanto, depende diretamente da rigorosidade do pipeline de treino, da manutenção da ordem temporal e da reprodutibilidade do pré-processamento.

A documentação do projeto deve ser revisada sempre que houver:

- mudança na base de dados
- inclusão de novas features
- alteração dos limites temporais
- alteração de hiperparâmetros
- mudança do alvo ou da forma de classificar resultado

A manutenção dessa rastreabilidade é essencial para garantir que o modelo continue validado, interpretável e reprodutível.
