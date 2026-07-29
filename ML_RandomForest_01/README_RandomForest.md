# README — Random Forest para previsão de resultados de partidas

Este diretório contém o pipeline de treinamento e aplicação de um modelo Random Forest para prever o resultado de partidas de futebol, atuando como um comparativo ao modelo XGBoost desenvolvido no escopo do TCC.

## Objetivo

O objetivo é prever o resultado de uma partida em três classes:

- `0` = vitória visitante
- `1` = empate
- `2` = vitória mandante

A base de entrada é o arquivo `football_matches_ml.csv`, gerado pelo script principal do projeto.

## Pipeline utilizado

Para garantir a validade científica e a justa comparação entre os algoritmos no trabalho final, o pré-processamento deste modelo consome diretamente os artefatos (`preprocessing_artifacts.pkl`) gerados pelo script do XGBoost.

### 1. Bucketização da coluna `competition`

As regras e categorias criadas para simplificar o texto livre das competições (como `friendly`, `world_cup_final`, `qualifiers`) são idênticas e importadas dos artefatos originais. 

### 2. Frequency encoding para times

Para evitar a explosão de dimensionalidade e manter a paridade com o modelo baseline, aplicamos o dicionário de frequências já calculado no treino do XGBoost aos nomes das seleções.

### 3. Split temporal para evitar data leakage

A divisão entre treino, validação e teste é feita de forma temporal rigorosa, evitando que o modelo aprenda com o futuro para prever o passado.
- Treino: dados até 2021
- Validação (usada na validação cruzada interna): 2022 a 2023
- Teste: dados a partir de 2024

### 4. Tratamento de valores nulos (Imputação)

Como o algoritmo `RandomForestClassifier` do `scikit-learn` não lida nativamente com valores ausentes (`NaN`) nas *features* — diferentemente do XGBoost —, foi adicionada uma etapa de imputação. Utilizou-se o `SimpleImputer(strategy='median')`, ajustado apenas no conjunto de treino e aplicado aos demais conjuntos, garantindo a ausência de *data leakage*.

### 5. Correção do viés de mandante

Para mitigar o efeito de classe desbalanceada e o viés histórico de vitórias do mandante, o modelo utiliza o parâmetro interno `class_weight='balanced'`. Na prática, o algoritmo pondera as classes menos frequentes (empate e vitória visitante) com maior peso durante a construção das árvores.

### 6. Otimização de hiperparâmetros (Hyperparameter Tuning)

No lugar do *early stopping* (utilizado no XGBoost), o controle de *overfitting* e a busca pela melhor arquitetura foram realizados através do `RandomizedSearchCV`. O processo iterou sobre diferentes combinações utilizando um *split* temporal fixo (`PredefinedSplit`). Isso garantiu que o Random Forest utilizasse exatamente a mesma base de validação (2022-2023) empregada pelo modelo de referência, assegurando uma comparação metodologicamente justa.

## Modelo principal

O modelo final assumiu a melhor configuração determinística encontrada durante a busca (fixada através do `random_state=42`):

- `n_estimators=300` (quantidade de árvores)
- `max_depth=5` (limite raso para generalização e controle de overfitting)
- `min_samples_leaf=5`
- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1` (processamento paralelo)

## Features usadas

As colunas de identificação e as que causam vazamento do target (`match_id`, `date`, `home_goals`, `away_goals`, `rank_diff`, `points_diff`, `match_result`) foram removidas. 

O conjunto exato de *features* utilizado pelo modelo foi importado do arquivo de artefatos do XGBoost (`feature_columns`), nivelando a base de aprendizado para a comparação acadêmica.

## Métricas de avaliação

O modelo é avaliado no conjunto de teste com as seguintes métricas:

- `F1-macro`
- `log_loss`
- `classification_report`
- Matriz de confusão

## Arquivos gerados no treino

Após a execução do treinamento (`train_rf.py`), são produzidos os seguintes artefatos:

- `rf_match_result_model.pkl` — modelo treinado e otimizado em formato pickle.
- `rf_imputer.pkl` — objeto de imputação da mediana, ajustado exclusivamente com os dados de treino para evitar *data leakage* em predições futuras.
- `rf_confusion_matrix.png` — matriz de confusão no conjunto de teste.
- `rf_feature_importance.png` — gráfico listando o impacto de cada variável na decisão das árvores.

## Predição em 2026

O script `predict_2026_rf.py` carrega:

- O modelo Random Forest salvo em `rf_match_result_model.pkl`
- O preenchedor de nulos salvo em `rf_imputer.pkl`
- Os artefatos base em `preprocessing_artifacts.pkl`

O pipeline reaplica o tratamento exato aos dados (incluindo o `SimpleImputer` para preencher os valores nulos com a mediana do treino) e gera um CSV comparativo com:

- resultado real
- resultado predito
- acerto/erro
- probabilidades por classe

Arquivo produzido:

- `comparativo_real_vs_predito_2026_rf.csv`

## Resumo executivo

Este desenvolvimento assegurou uma comparação metodologicamente sólida com o classificador XGBoost. Através do consumo dos mesmos artefatos de pré-processamento e da aplicação de validação cruzada para seleção de hiperparâmetros, o Random Forest atingiu uma acurácia comparável, exibindo um viés probabilístico mais conservador que auxilia na análise das incertezas inerentes aos confrontos de futebol.

## Como executar

1. Certifique-se de que o script `train_xgboost.py` (na pasta vizinha) já foi executado para gerar os artefatos de pré-processamento.
2. Treine e avalie o modelo executando `python train_rf.py`.
3. Faça a predição para a Copa de 2026 executando `python predict_2026_rf.py`.