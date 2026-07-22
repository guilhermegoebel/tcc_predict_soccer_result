# README — XGBoost para previsão de resultados de partidas

Este diretório contém o pipeline de treinamento e aplicação de um modelo XGBoost para prever o resultado de partidas de futebol.

## Objetivo

O objetivo é prever o resultado de uma partida em três classes:

- `0` = vitória visitante
- `1` = empate
- `2` = vitória mandante

A base de entrada é o arquivo `football_matches_ml.csv`, gerado pelo script principal do projeto.

## Pipeline utilizado

### 1. Bucketização da coluna `competition`

A coluna `competition` possui texto livre e foi convertida para categorias mais simples para reduzir ruído e facilitar o aprendizado do modelo.

Exemplos de categorias criadas:

- `friendly`
- `world_cup_final`
- `qualifiers`
- `cup_or_championship`
- `other`

Essa etapa é feita com a função `bucket_competition()`.

### 2. Frequency encoding para times

Em vez de usar one-hot encoding para nomes das seleções, o modelo aplica `frequency encoding` aos times.

Benefícios:

- evita explosão de dimensionalidade
- mantém um sinal compacto da frequência relativa das seleções no conjunto de treino
- reduz a complexidade do espaço de features

A codificação é calculada somente com os dados de treino, e depois reaplicada nos splits de validação e teste, evitando vazamento de informação.

### 3. Split temporal para evitar data leakage

A divisão entre treino, validação e teste é feita de forma temporal:

- treino: dados até um limite de ano
- validação: intervalo seguinte
- teste: dados mais recentes

Isso é importante porque features como histórico direto e forma recente dependem da ordem cronológica dos jogos. O modelo não deve aprender com o futuro para prever o passado.

### 4. Correção do viés de mandante com sample weights

Para mitigar o efeito de classe desbalanceada e o viés histórico de que o mandante vence com maior frequência, o treinamento usa `sample_weight` calculados com `class_weight='balanced'`.

Em prática:

- cada exemplo recebe um peso inversamente proporcional à frequência da classe
- isso ajuda o modelo a aprender também as classes menos frequentes, como empate e derrota do mandante

### 5. Early stopping para evitar overfitting

O modelo é treinado com `early_stopping_rounds=30`.

Esse mecanismo monitora a performance no conjunto de validação e interrompe o treinamento quando a melhoria deixa de acontecer, reduzindo o risco de overfitting.

## Modelo principal

O modelo utilizado é o `XGBClassifier` com configuração multinomial:

- `objective='multi:softprob'`
- `num_class=3`
- `eval_metric='mlogloss'`
- `tree_method='hist'`
- `n_estimators=500`
- `learning_rate=0.05`
- `max_depth=5`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=5`
- `reg_lambda=1.0`
- `random_state=42`

## Features usadas

As colunas de identificação e do alvo são removidas antes do treinamento, como:

- `match_id`
- `date`
- `home_goals`
- `away_goals`
- `rank_diff`
- `points_diff`
- `match_result`

As features restantes são compostas por variáveis como:

- rankings e pontos de FIFA
- valor de mercado médio
- forma recente
- histórico direto (H2H)
- ano e mês
- indicador de ano de Copa do Mundo
- features derivadas de encoding de time e competição

## Métricas de avaliação

No script de treino, o modelo é avaliado com:

- `F1-macro`
- `log_loss`
- `classification_report`
- matriz de confusão

Também é calculado um baseline simples:

- prever sempre vitória do mandante

Esse baseline serve para confirmar se o modelo realmente aprende algo além do viés natural da distribuição.

## Arquivos gerados no treino

Após a execução do treinamento, são produzidos os seguintes artefatos:

- `xgb_match_result.json` — modelo treinado em formato XGBoost
- `preprocessing_artifacts.pkl` — artefatos de pré-processamento necessários para reaplicar o mesmo pipeline em novos dados
- `confusion_matrix.png` — matriz de confusão no conjunto de teste
- `feature_importance.png` — gráfico da importância das variáveis

## Predição em 2026

O script `predict_2026.py` carrega:

- o modelo salvo em `xgb_match_result.json`
- os artefatos de pré-processamento em `preprocessing_artifacts.pkl`

Em seguida, ele reaplica o mesmo tratamento de:

- `competition`
- `home_team` e `away_team`
- dummies de bucketização

E gera um CSV comparativo com:

- resultado real
- resultado predito
- acerto/erro
- probabilidades por classe

Arquivo produzido:

- `comparativo_real_vs_predito_2026.csv`

## Resumo executivo

O modelo foi construído com uma abordagem robusta para evitar vazamento temporal e manter o aprendizado alinhado com a dinâmica real de partidas ao longo do tempo.

Os principais pontos do desenho são:

- bucketização one-hot de `competition`
- frequency encoding para os nomes das seleções em vez de one-hot
- separação treino/validação/teste com ordem temporal
- correção do viés do mandante através de pesos por classe
- uso de early stopping para reduzir overfitting

## Como executar

1. Gere o dataset principal com `script.py`
2. Treine o modelo com `train_xgboost.py`
3. Faça a predição com `predict_2026.py`

## Observação

A lógica de pré-processamento precisa ser reaplicada exatamente como no treino para que as features fiquem compatíveis com o modelo já aprendido.