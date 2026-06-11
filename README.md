# Dicionário de Dados — `football_matches_ml.csv`

| Variável                     | Tipo     | Descrição                                                                                                  |
| ---------------------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| `match_id`                   | Inteiro  | Identificador único da partida                                                                             |
| `date`                       | Data     | Data da partida                                                                                            |
| `competition`                | Texto    | Competição do jogo                                                                                         |
| `home_team`                  | Texto    | Seleção mandante                                                                                           |
| `away_team`                  | Texto    | Seleção visitante                                                                                          |
| `home_goals`                 | Inteiro  | Gols marcados pela seleção mandante                                                                        |
| `away_goals`                 | Inteiro  | Gols marcados pela seleção visitante                                                                       |
| `home_rank`                  | Numérico | Ranking FIFA da seleção mandante antes da partida                                                          |
| `away_rank`                  | Numérico | Ranking FIFA da seleção visitante antes da partida                                                         |
| `home_points`                | Numérico | Pontuação FIFA da seleção mandante                                                                         |
| `away_points`                | Numérico | Pontuação FIFA da seleção visitante                                                                        |
| `rank_diff`                  | Numérico | Diferença entre rankings FIFA (`away_rank - home_rank`)                                                    |
| `points_diff`                | Numérico | Diferença entre pontos FIFA (`home_points - away_points`)                                                  |
| `home_market_value`          | Numérico | Soma do valor de mercado dos últimos 5 jogadores da escalação mandante                                     |
| `away_market_value`          | Numérico | Soma do valor de mercado dos últimos 5 jogadores da escalação visitante                                    |
| `market_value_diff`          | Numérico | Diferença entre valores de mercado (`home_market_value - away_market_value`)                               |
| `home_recent_win_rate`       | Numérico | Taxa de vitórias da seleção mandante nos últimos N jogos                                                   |
| `away_recent_win_rate`       | Numérico | Taxa de vitórias da seleção visitante nos últimos N jogos                                                  |
| `home_recent_goals_scored`   | Numérico | Média de gols marcados pela seleção mandante nos últimos jogos                                             |
| `away_recent_goals_scored`   | Numérico | Média de gols marcados pela seleção visitante nos últimos jogos                                            |
| `home_recent_goals_conceded` | Numérico | Média de gols sofridos pela seleção mandante nos últimos jogos                                             |
| `away_recent_goals_conceded` | Numérico | Média de gols sofridos pela seleção visitante nos últimos jogos                                            |
| `h2h_home_wins`              | Inteiro  | Quantidade de vitórias históricas do mandante em confrontos diretos                                        |
| `h2h_away_wins`              | Inteiro  | Quantidade de vitórias históricas do visitante em confrontos diretos                                       |
| `h2h_draws`                  | Inteiro  | Quantidade de empates históricos entre as seleções                                                         |
| `h2h_goal_diff`              | Numérico | Saldo de gols histórico do mandante nos confrontos diretos                                                 |
| `home_elo`                   | Numérico | Rating Elo da seleção mandante antes da partida                                                            |
| `away_elo`                   | Numérico | Rating Elo da seleção visitante antes da partida                                                           |
| `elo_diff`                   | Numérico | Diferença entre ratings Elo (`home_elo - away_elo`)                                                        |
| `year`                       | Inteiro  | Ano da realização da partida                                                                               |
| `month`                      | Inteiro  | Mês da realização da partida                                                                               |
| `is_world_cup_year`          | Binário  | Indica se o jogo ocorreu em ano de Copa do Mundo (`1 = sim`, `0 = não`)                                    |
| `match_result`               | Inteiro  | Variável alvo da classificação multiclasse (`2 = vitória mandante`, `1 = empate`, `0 = vitória visitante`) |

---

# Variáveis derivadas

## Diferença de ranking FIFA

rank_diff = away_rank - home_rank

---

## Diferença de pontos FIFA

points_diff = home_points - away_points

---

## Diferença de valor de mercado

market_value_diff = home_market_value - away_market_value

---

## Diferença Elo

elo_diff = home_elo - away_elo

---

# Variável alvo

## Resultado da partida

| Valor | Classe                       |
| ----- | ---------------------------- |
| 2     | Vitória da seleção mandante  |
| 1     | Empate                       |
| 0     | Vitória da seleção visitante |

---

# Observações metodológicas

* Rankings FIFA são associados à partida utilizando a data mais próxima anterior ao jogo.
* Valores de mercado são calculados com base nos últimos registros anteriores à data da partida.
* O cálculo de forma recente utiliza os últimos `N` jogos anteriores da seleção.
* O Elo é atualizado dinamicamente em ordem cronológica ao longo das partidas.
* Os confrontos diretos (`H2H`) consideram apenas jogos anteriores à partida analisada.

--------------------------------------------------------------------------

# Cálculo do Rating Elo

O sistema Elo é um método de pontuação dinâmica utilizado para medir a força relativa de equipes ao longo do tempo. No conjunto de dados, o Elo foi utilizado para representar o desempenho acumulado das seleções nacionais antes de cada partida.

Cada seleção inicia com uma pontuação base:

Elo_{inicial} = 1500

Após cada jogo, o rating é atualizado considerando:

* força do adversário;
* resultado obtido;
* expectativa pré-jogo.

---

# Probabilidade esperada

Antes da partida, calcula-se a expectativa de vitória da seleção mandante utilizando:

E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}

Onde:

| Símbolo | Descrição                            |
| ------- | ------------------------------------ |
| (E_A)   | expectativa de resultado da equipe A |
| (R_A)   | Elo atual da equipe A                |
| (R_B)   | Elo atual da equipe B                |

O mesmo cálculo é realizado para a equipe visitante.

---

# Resultado real da partida

O resultado real é representado por:

| Resultado | Valor |
| --------- | ----- |
| Vitória   | 1     |
| Empate    | 0.5   |
| Derrota   | 0     |

---

# Atualização do Elo

Após o término da partida, o novo rating é calculado pela fórmula:

R'_A = R_A + K(S_A - E_A)

Onde:

| Símbolo | Descrição          |
| ------- | ------------------ |
| (R'_A)  | novo Elo da equipe |
| (R_A)   | Elo anterior       |
| (K)     | fator de ajuste    |
| (S_A)   | resultado real     |
| (E_A)   | resultado esperado |

No projeto foi utilizado:

K = 20

---

# Interpretação

O sistema funciona da seguinte forma:

* vencer adversários fortes aumenta mais o Elo;
* perder para equipes fracas reduz mais o Elo;
* empates ajustam o rating de forma intermediária;
* equipes com desempenho consistente acumulam ratings maiores ao longo do tempo.

---

# Exemplo simplificado

Considere:

| Seleção | Elo  |
| ------- | ---- |
| Brasil  | 1700 |
| Japão   | 1500 |

O Brasil possui maior expectativa de vitória. Caso:

* o Brasil vença, o ajuste será pequeno;
* o Japão vença, o ganho do Japão será elevado e a perda do Brasil significativa.

---

# Uso no dataset

Para cada partida foram armazenadas as variáveis:

* `home_elo`
* `away_elo`
* `elo_diff`

A feature `elo_diff` representa:

elo_diff = home_elo - away_elo

Valores positivos indicam vantagem histórica da equipe mandante segundo o sistema Elo.

---
script.py
            ↓
football_matches_ml.csv
            ↓
dataset_copa2026.py
            ↓
football_matches_ml_worldcup2026.csv
            ↓
xgboost_ml.py 
            ↓
worldcup_2026_group_stage.csv
            ↓
Predição
            ↓
worldcup_2026_predictions.csv