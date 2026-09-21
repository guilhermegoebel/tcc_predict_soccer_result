# Random Forest para previsão de resultados de partidas

## Visão geral

Este diretório concentra o pipeline de treinamento, validação, explicabilidade e inferência de um modelo de classificação multinomial baseado em Random Forest para prever o resultado de partidas de futebol em três classes:

- 0 = vitória visitante
- 1 = empate
- 2 = vitória mandante

O objetivo do modelo é estimar, a partir de dados históricos, a probabilidade de cada resultado possível antes da partida, com foco em uso robusto para análise de desempenho esportivo, comparação de cenários e suporte à interpretação do fator que mais influencia o resultado.

A implementação atual está organizada em quatro blocos principais:

- `train_rf.py` — treinamento e validação do modelo
- `tune_rf_hyperparams.py` — busca de hiperparâmetros via RandomizedSearchCV com validação temporal
- `predict_2026_rf.py` — aplicação do modelo em dados de 2026
- `feature_importance_analysis_rf.py` — análise de importância por SHAP e permutation importance

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

- `home_team`, `away_team` (identificadoras — não entram como feature, ver "Features usadas no modelo")
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

O conjunto de dados apresenta forte tendência para vitórias do mandante. Para mitigar esse efeito, o `RandomForestClassifier` é instanciado com `class_weight='balanced'`.

Esse ajuste:

- reduz o favorecimento artificial da classe mais frequente
- melhora o desempenho em classes raras como empate e vitória visitante
- é especialmente importante para classificação multinomial com classes desbalanceadas

### 3. Tratamento de valores nulos (Imputação)

O `RandomForestClassifier` do scikit-learn não lida nativamente com valores ausentes (`NaN`) nas features. Por isso, há uma etapa de imputação: `SimpleImputer(strategy='median')`, ajustado **apenas** no conjunto de treino e aplicado, sem reajuste, aos conjuntos de validação e teste — garantindo ausência de vazamento também nessa etapa.

### 4. Controle de overfitting e otimização de hiperparâmetros

O controle de overfitting é feito por meio da própria escolha de hiperparâmetros (profundidade máxima, mínimo de amostras por folha), validada em `2022–2023`.

Duas formas de obter esses hiperparâmetros, escolhidas automaticamente por `train_rf.py`:

- **Se `rf_best_hyperparams.json` existir** (gerado por `tune_rf_hyperparams.py`), ele é carregado diretamente.
- **Caso contrário**, o script busca internamente via `RandomizedSearchCV` com `PredefinedSplit` — uma única divisão fixa treino/validação, testando uma grade simplificada de combinações.

Em ambos os casos, o modelo final é treinado **apenas com o conjunto de treino** — nunca treino+validação combinados —, para manter a validação livre de contaminação, necessária para a calibração a seguir.

### 5. Calibração de probabilidades

Mesmo com pesos por classe, as probabilidades brutas do Random Forest podem ficar menos confiáveis quando o modelo é treinado com balanceamento de classes. Por isso, o pipeline inclui calibração por classe usando `IsotonicRegression`, ajustada apenas no conjunto de validação.

Essa etapa:

- não altera a decisão de classe final
- melhora a qualidade das probabilidades emitidas
- reduz distorções no `log_loss` calibrado

---

## Modelo e hiperparâmetros

O modelo principal é o `RandomForestClassifier` configurado com:

- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1`

Os demais hiperparâmetros (`n_estimators`, `max_depth`, `min_samples_leaf` e, quando obtidos via `tune_rf_hyperparams.py`, também `min_samples_split` e `max_features`) não têm um valor fixo declarado no código — são o resultado da busca (interna ou externa, ver seção anterior) e variam conforme a execução. Para conferir os hiperparâmetros do modelo efetivamente salvo, use:

```python
import pickle
with open('rf_match_result_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.get_params())
```

### Grade interna (padrão, sem arquivo externo)

Usada por `train_rf.py` quando `rf_best_hyperparams.json` não existe:

- `n_estimators`: [100, 200, 300]
- `max_depth`: [5, 8, 10]
- `min_samples_leaf`: [5, 10]

Busca: `RandomizedSearchCV` com `PredefinedSplit` (uma única divisão fixa treino/validação), 10 combinações testadas, métrica `f1_macro`.

### Busca de hiperparâmetros (`tune_rf_hyperparams.py`, opcional)

Espaço de busca mais amplo, e validação mais rigorosa:

- `RandomizedSearchCV`
- `TimeSeriesSplit` agrupado por data (5 folds)
- métrica `f1_macro`
- 40 combinações testadas
- validação temporal para evitar contaminação por vazamento

Esse processo gera:

- `rf_best_hyperparams.json`
- `cv_results_hyperparam_search_rf.csv`

**Nota:** essa busca externa ainda não foi executada de forma definitiva neste projeto — o modelo atual usa os hiperparâmetros da grade interna.

---

## Features usadas no modelo

As colunas identificadoras e vazadas são removidas antes do treino:

- `match_id`
- `date`
- `home_goals`
- `away_goals`
- `match_result`

Adicionalmente, também são excluídas (calculadas apenas para servir de identificador, não como feature numérica direta):

- `home_team`, `away_team`, `competition`, `competition_bucket`

Nem frequency encoding de time (`home_team_freq`/`away_team_freq`), nem as flags de "time visto no treino" (`home_team_seen`/`away_team_seen`), nem dummies de competição (`comp_*`) são usadas como feature — nenhuma dessas colunas chega a ser calculada.

`rank_diff` e `points_diff` são mantidos como features — são diferenças pré-calculadas de informação disponível antes do jogo (ranking e pontuação FIFA), não do resultado da partida.

A lista final das features inclui variáveis de:

- ranking e pontos
- valor de mercado
- recente desempenho
- histórico direto
- contexto de competição (indicador `world_cup`)
- informação temporal
- diferenciais calculados antes do jogo

A lista completa é calculada localmente em `train_rf.py` e persistida em `rf_preprocessing_artifacts.pkl`, para reaplicação idêntica em `predict_2026_rf.py` e `feature_importance_analysis_rf.py`.

---

## Métricas de avaliação

O treinamento gera avaliação em dois conjuntos, mais um baseline de referência:

- validação
- teste
- baseline ("sempre vitória do mandante")

As principais métricas calculadas são:

- `accuracy`
- `f1_macro`
- `balanced_accuracy`
- `precision_macro`
- `recall_macro`
- `log_loss` bruto e calibrado

O baseline serve para verificar se o modelo realmente aprendeu algo além do viés dominante da distribuição dos resultados.

### Artefatos de avaliação gerados

- `rf_evaluation_metrics.json`
- `rf_evaluation_metrics.csv`
- `rf_resumo_executivo.png`
- `rf_desempenho_por_classe.png`
- `rf_confusion_matrix.png`

---

## Explicabilidade do modelo

O diretório também inclui análise de explicabilidade para responder o que o modelo está usando na decisão. Existem duas abordagens complementares, calculadas em `feature_importance_analysis_rf.py`:

### 1. Permutation importance

Testa o impacto de cada feature ao embaralhar seus valores no conjunto de teste (já imputado) e medir a perda de desempenho. É útil para responder:

- qual feature mais afeta o F1-macro?
- a remoção de uma variável compromete muito a qualidade do modelo?

### 2. SHAP

O SHAP explica a predição de cada instância em termos de contribuição de cada feature. Isso permite:

- entender sinais positivos e negativos para cada classe
- ver a direção do efeito da feature
- analisar se a variável empurra a previsão para vitória do mandante, empate ou vitória visitante

Artefatos produzidos:

- `permutation_importance_rf.png`
- `shap_importance_rf.png`
- `shap_summary_home_win_rf.png`

A combinação de SHAP + permutation importance é recomendada porque:

- a importância nativa do Random Forest mede uso dentro da árvore (sensível a correlação entre features e a variáveis de alta cardinalidade)
- SHAP mede impacto real da feature na previsão
- permutation importance mede perda de desempenho ao remover o sinal da feature

Requer a biblioteca `shap` (`pip install shap`), que não é dependência dos demais scripts.

---

## Inferência em 2026

O script `predict_2026_rf.py` aplica o modelo treinado ao arquivo de 2026.

### Objetivo

Carregar:

- `rf_match_result_model.pkl`
- `rf_imputer.pkl`
- `rf_calibrators.pkl`
- `rf_preprocessing_artifacts.pkl`

E gerar:

- probabilidades por classe (calibradas)
- resultado previsto
- comparação com o valor real
- métricas de desempenho
- matriz de confusão

### Arquivo de saída principal

- `comparativo_real_vs_predito_2026_rf.csv`

Também são produzidos:

- `rf_prediction_metrics.json`
- `confusion_matrix_rf_2026.png`

---

## Fluxo recomendado para retrain

Para treinar novamente o modelo de forma consistente, siga este fluxo:

1. Verificar que a base `football_matches_ml.csv` esteja atualizada.
2. Valer a integridade das colunas necessárias e o alvo `match_result`.
3. Ajustar os limites temporais de treino/validação/teste, se necessário.
4. Executar `tune_rf_hyperparams.py` para otimizar os parâmetros do modelo (opcional — sem esse passo, `train_rf.py` busca internamente uma grade menor).
5. Executar `train_rf.py` para treinar o modelo final.
6. Validar métricas em `rf_evaluation_metrics.json`.
7. Rodar `feature_importance_analysis_rf.py` para interpretar o comportamento do modelo.
8. Executar `predict_2026_rf.py` para inferência em dados novos.

---

## Conclusão

Este pipeline foi desenhado para equilibrar três objetivos principais:

1. desempenho preditivo robusto
2. consistência temporal e minimização de vazamento
3. explicabilidade e rastreabilidade do modelo

O Random Forest é uma escolha adequada para esse problema por combinar boa capacidade preditiva com relativa robustez a overfitting (via bagging), além de suporte à interpretação por métodos como SHAP. A qualidade do modelo, no entanto, depende diretamente da rigorosidade do pipeline de treino, da manutenção da ordem temporal e da reprodutibilidade do pré-processamento.

A documentação do projeto deve ser revisada sempre que houver:

- mudança na base de dados
- inclusão de novas features
- alteração dos limites temporais
- alteração de hiperparâmetros
- mudança do alvo ou da forma de classificar resultado

A manutenção dessa rastreabilidade é essencial para garantir que o modelo continue validado, interpretável e reprodutível.