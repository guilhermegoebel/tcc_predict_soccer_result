# Projeto de Machine Learning para previsão de resultados de partidas

Este repositório reúne o pipeline de preparação de dados, treinamento e avaliação de modelos de classificação para prever o resultado de partidas de futebol usando informações históricas, recentes e contextuais. O projeto compara diferentes algoritmos e consolida a pipeline principal de geração do dataset e uso em produção.

## Objetivo geral

O objetivo do projeto é prever, antes da partida, o resultado de uma partida em três classes:

- 0 = vitória visitante
- 1 = empate
- 2 = vitória mandante

A classificação usa variáveis pré-jogo, como:

- ranking e pontos FIFA
- valor de mercado dos jogadores
- histórico recente de desempenho
- histórico direto entre as seleções
- contexto da competição
- dados temporais como mês e ano

---

## Estrutura do projeto

A estrutura atual do diretório é a seguinte:

- `script.py` — gera o dataset principal `football_matches_ml.csv` a partir dos dados brutos.
- `ML_XGBoost_01/` — módulo de treinamento, tuning e inferência do modelo XGBoost.
- `ML_RandomForest_01/` — implementação do modelo Random Forest para comparação de desempenho.
- `ML_LogisticRegression_01/` — implementação do modelo de regressão logística.
---

## Dados brutos e preparação

Os dados brutos principais estão no diretório raiz e incluem:

- `matches.csv`
- `rankings_fifa.csv`
- `valor_mercado_jogadores.csv`
- `paises_siglas_relacao.csv`
- `Copa2026.csv`

O script `script.py` consolida estas fontes e gera o dataset de modelagem:

- `football_matches_ml.csv`

Esse dataset já contém features derivadas para uso em aprendizado supervisionado.

---

## Dataset principal

O arquivo principal é `football_matches_ml.csv`.

### Principais colunas

- `match_id` — identificador da partida
- `date` — data da partida
- `competition` — competição
- `home_team`, `away_team` — seleções mandante e visitante
- `home_goals`, `away_goals` — gols da partida
- `home_rank`, `away_rank` — ranking FIFA antes do jogo
- `home_points`, `away_points` — pontos FIFA antes do jogo
- `rank_diff` — diferença de ranking
- `points_diff` — diferença de pontos
- `home_market_value_avg`, `away_market_value_avg` — média do valor de mercado
- `market_value_avg_diff` — diferença de valor de mercado
- `home_found_count`, `away_found_count` — quantos jogadores tiveram valor encontrado
- `home_recent_win_rate`, `away_recent_win_rate` — taxa de vitórias recentes
- `home_recent_goals_scored`, `away_recent_goals_scored` — gols marcados recentes
- `home_recent_goals_conceded`, `away_recent_goals_conceded` — gols sofridos recentes
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws` — histórico direto
- `h2h_goal_diff` — saldo de gols no histórico direto
- `year`, `month` — contexto temporal
- `world_cup` — indicador de ano de Copa do Mundo
- `match_result` — alvo do modelo

### Mapeamento do target

- 0 = vitória visitante
- 1 = empate
- 2 = vitória mandante

---

## Convenções metodológicas

O projeto foi construído com atenção a alguns pilares fundamentais para manter a validade do modelo:

### 1. Separação temporal

Os splits são feitos em ordem cronológica e não aleatória. Isso evita que o modelo aprenda com informações futuras para prever resultados do passado.

### 2. Evitação de vazamento

As variáveis derivadas, como o histórico recente e o confronto direto, são construídas apenas com base em dados anteriores à partida em análise.

### 3. Tratamento de classes desbalanceadas

Os modelos aplicam estratégias de ponderação alternativa ou ajustes para mitigar o viés de classes majoritárias, especialmente a tendência de vitórias do mandante.

### 4. Explicabilidade

Além do desempenho preditivo, o projeto também inclui análise de importância de variáveis e explicação por feature impact, com foco em SHAP e permutation importance no módulo XGBoost.

---

## Pipeline principal

O fluxo recomendando para uso do projeto é:

1. Preparar os dados brutos.
2. Gerar o dataset principal com `script.py`.
3. Treinar um modelo.
4. Validar o desempenho em conjunto de validação e teste.
5. Interpretar a importância das features.
6. Aplicar o modelo em dados futuros, como 2026.

---

## Modelos disponíveis

- XGBoost
- Random Forest
- Regressão logística

---

## Observações finais

- O projeto foi organizado para permitir comparação de algoritmos em um mesmo problema.
- A manutenção dos critérios de temporização, preprocessing e reprodutibilidade é essencial para garantir validez experimental.
- Cada modelo deve ser interpretado dentro do seu contexto e comparado com a linha de base adequada.

Este README serve como visão geral do projeto. Para detalhes específicos de um modelo, consulte os READMEs individuais em cada diretório correspondente.
