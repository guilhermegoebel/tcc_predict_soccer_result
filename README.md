# Projeto de ML para previsão de resultados de partidas

Este diretório reúne o pipeline de preparação de dados e treino de um classificador para prever o resultado de partidas de futebol com base em características históricas e de desempenho recente.

## Estrutura

- `script.py`: gera o dataset principal `football_matches_ml.csv` a partir de:
  - `matches.csv`
  - `rankings_fifa.csv`
  - `valor_mercado_jogadores.csv`
  - `paises_siglas_relacao.csv`
- `dataset_copa2026.py`: filtra partidas envolvendo seleções classificadas para a Copa do Mundo de 2026.
- `ML_XGBoost_01/train_xgboost.py`: treina um modelo XGBoost com split temporal e avaliação em validação/teste.
- `ML_XGBoost_01/predict_2026.py`: aplica o modelo treinado a partidas de 2026 e gera um comparativo real vs predito.

## Dataset principal

O arquivo gerado é `football_matches_ml.csv`.

### Dicionário de colunas

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `match_id` | inteiro | Identificador único da partida. |
| `date` | data | Data da partida. |
| `competition` | texto | Competição da partida. |
| `home_team` | texto | Seleção mandante. |
| `away_team` | texto | Seleção visitante. |
| `home_goals` | inteiro | Gols do mandante na partida. |
| `away_goals` | inteiro | Gols do visitante na partida. |
| `home_rank` | numérico | Ranking FIFA da seleção mandante antes do jogo. |
| `away_rank` | numérico | Ranking FIFA da seleção visitante antes do jogo. |
| `home_points` | numérico | Pontos FIFA do mandante antes da partida. |
| `away_points` | numérico | Pontos FIFA do visitante antes da partida. |
| `rank_diff` | numérico | Diferença `away_rank - home_rank`. |
| `points_diff` | numérico | Diferença `home_points - away_points`. |
| `home_market_value_avg` | numérico | Média do valor de mercado dos jogadores do mandante encontrados para a partida. |
| `away_market_value_avg` | numérico | Média do valor de mercado dos jogadores do visitante encontrados para a partida. |
| `market_value_avg_diff` | numérico | Diferença `home_market_value_avg - away_market_value_avg`. |
| `home_found_count` | inteiro | Quantidade de jogadores com valor de mercado encontrado para o mandante. |
| `away_found_count` | inteiro | Quantidade de jogadores com valor de mercado encontrado para o visitante. |
| `home_recent_win_rate` | numérico | Taxa de vitórias recentes do mandante. |
| `away_recent_win_rate` | numérico | Taxa de vitórias recentes do visitante. |
| `home_recent_goals_scored` | numérico | Média de gols marcados nos últimos jogos do mandante. |
| `away_recent_goals_scored` | numérico | Média de gols marcados nos últimos jogos do visitante. |
| `home_recent_goals_conceded` | numérico | Média de gols sofridos nos últimos jogos do mandante. |
| `away_recent_goals_conceded` | numérico | Média de gols sofridos nos últimos jogos do visitante. |
| `h2h_home_wins` | inteiro | Vitórias históricas do mandante no histórico direto. |
| `h2h_away_wins` | inteiro | Vitórias históricas do visitante no histórico direto. |
| `h2h_draws` | inteiro | Empates históricos no confronto direto. |
| `h2h_goal_diff` | numérico | Saldo de gols histórico do mandante no confronto direto. |
| `year` | inteiro | Ano da partida. |
| `month` | inteiro | Mês da partida. |
| `world_cup` | binário | Indicador de ano de Copa do Mundo (`1 = sim`, `0 = não`). |
| `match_result` | inteiro | Target do modelo: `0 = vitória visitante`, `1 = empate`, `2 = vitória mandante`. |

## Observações metodológicas

- Os rankings FIFA são associados à partida pela data mais próxima anterior ao jogo.
- Os valores de mercado usam o histórico do jogador e procuram o valor mais próximo e anterior à data da partida, com fallback para o registro mais próximo disponível em caso necessário.
- As features de forma recente usam os últimos jogos cronológicos de cada seleção.
- O histórico direto (`h2h_*`) considera apenas jogos anteriores à partida em análise.
- O dataset foi construído em ordem cronológica, evitando vazamento temporal na geração das features.

## Pipeline de ML

A modelagem em `ML_XGBoost_01/train_xgboost.py` usa:

1. `bucketização` da coluna `competition` em categorias simples.
2. `split temporal` para treino/validação/teste.
3. `frequency encoding` para times, calculado apenas no conjunto de treino.
4. `sample weights` balanceados para reduzir o viés de classe.
5. `XGBoost` com `early stopping` para evitar overfitting.

### Arquivos gerados no treino

- `xgb_match_result.json`: modelo treinado em formato XGBoost.
- `preprocessing_artifacts.pkl`: artefatos usados para reaplicar o mesmo pré-processamento em novos dados.
- `confusion_matrix.png`: matriz de confusão do conjunto de teste.
- `feature_importance.png`: importância das features.

## Saídas de predição

O script `ML_XGBoost_01/predict_2026.py` produz:

- `comparativo_real_vs_predito_2026.csv`: tabela com resultado real e predito para as partidas de 2026.
- `football_matches_ml_worldcup2026.csv`: subconjunto de partidas envolvendo seleções classificadas para a Copa de 2026.

## Como usar

1. Execute `script.py` para gerar o dataset principal.
2. Execute `ML_XGBoost_01/train_xgboost.py` para treinar o modelo.
3. Execute `ML_XGBoost_01/predict_2026.py` para gerar previsões para 2026.

## Target do modelo

| Valor | Classe |
| --- | --- |
| `0` | Vitória visitante |
| `1` | Empate |
| `2` | Vitória mandante |
