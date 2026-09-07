# Projeto de ML para previsão de resultados de partidas

Este diretório reúne o pipeline de preparação de dados e treino de um classificador para prever o resultado de partidas de futebol com base em características históricas e de desempenho recente.

## Estrutura

- `script.py`: gera o dataset principal `football_matches_ml.csv` a partir de:
  - `matches.csv`
  - `rankings_fifa.csv`
  - `valor_mercado_jogadores.csv`
  - `paises_siglas_relacao.csv`
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
- `evaluation_metrics.csv`: tabela com métricas de validação e teste.
- `evaluation_metrics.json`: relatório estruturado de métricas com classificação detalhada.

## Pipeline unificado

- `run_pipeline.py`: executa automaticamente a geração do dataset, o treino do modelo e a predição em 2026 em sequência.
- Use `python run_pipeline.py` para rodar todo o fluxo em um único passo.

## Saídas de predição

O script `ML_XGBoost_01/predict_2026.py` produz:

- `comparativo_real_vs_predito_2026.csv`: tabela com resultado real e predito para as partidas de 2026.
- `prediction_metrics.json`: métricas da predição em 2026 (se o gabarito estiver presente).

## Processo reprodutível

### Passo a passo

1. Garanta que os arquivos brutos estejam no diretório:
   - `matches.csv`
   - `rankings_fifa.csv`
   - `valor_mercado_jogadores.csv`
   - `paises_siglas_relacao.csv`
2. Execute `python run_pipeline.py` no diretório raiz do projeto.
3. O script fará em sequência:
   - geração do dataset principal (`script.py`)
   - treino do modelo XGBoost (`ML_XGBoost_01/train_xgboost.py`)
   - predição em 2026 (`ML_XGBoost_01/predict_2026.py`)
4. Ao final, revise os arquivos gerados:
   - `football_matches_ml.csv`
   - `ML_XGBoost_01/xgb_match_result.json`
   - `ML_XGBoost_01/preprocessing_artifacts.pkl`
   - `ML_XGBoost_01/confusion_matrix.png`
   - `ML_XGBoost_01/feature_importance.png`
   - `ML_XGBoost_01/evaluation_metrics.csv`
   - `ML_XGBoost_01/evaluation_metrics.json`
   - `ML_XGBoost_01/comparativo_real_vs_predito_2026.csv`
   - `ML_XGBoost_01/prediction_metrics.json`

## Como usar

1. Execute `python run_pipeline.py` para rodar todo o pipeline.
2. Se preferir executar etapas separadas:
   - `python script.py`
   - `python ML_XGBoost_01/train_xgboost.py`
   - `python ML_XGBoost_01/predict_2026.py`

## Target do modelo

| Valor | Classe |
| --- | --- |
| `0` | Vitória visitante |
| `1` | Empate |
| `2` | Vitória mandante |
