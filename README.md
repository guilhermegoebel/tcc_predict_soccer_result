# Projeto de Machine Learning para previsão de resultados de partidas

## Visão geral

Este repositório reúne as etapas de coleta, integração, preparação de dados, treinamento, avaliação, explicabilidade e inferência de modelos de classificação para prever resultados de partidas internacionais de futebol.

São comparados três algoritmos:

- Regressão Logística;
- Random Forest;
- XGBoost.

O problema é formulado como uma classificação em três classes:

- `0` = vitória visitante (`away_win`);
- `1` = empate (`draw`);
- `2` = vitória mandante (`home_win`).

---

## Estrutura do projeto

- `web_scraping/` — scripts de coleta de partidas, rankings FIFA e valores de mercado;
- `script.py` — integra as fontes e gera o dataset principal de modelagem;
- `ML_LogisticRegression_01/` — Modelo Regressão Logística;
- `ML_RandomForest_01/` — Modelo Random Forest;
- `ML_XGBoost_01/` — Modelo XGBoost;
- `requirements.txt` — versões das dependências utilizadas.

Cada diretório de modelo pode conter um README específico com os comandos e os artefatos daquela implementação.

---

## Coleta por web scraping

Os scripts de coleta estão em `web_scraping/`:

- `scrape_national_football_teams.py` — percorre páginas de partidas do National Football Teams por identificador, coleta placar, competição e escalações e ignora IDs já processados;
- `generate_player_list.py` — extrai a lista de jogadores a partir das escalações coletadas;
- `scrape_transfermarkt.py` — consulta jogadores no Transfermarkt, coleta o histórico de valores de mercado e retoma a execução pelos nomes já registrados;
- `scrape_fifa_rankings.py` — utiliza Selenium para navegar pelas publicações históricas do ranking masculino da FIFA e retoma pelas combinações de ano e data já salvas.

As fontes consultadas são:

- FIFA: <https://inside.fifa.com/fifa-world-ranking/men>;
- National Football Teams: <https://www.national-football-teams.com/>;
- Transfermarkt: <https://www.transfermarkt.com/>.

Os coletores utilizam intervalos entre requisições, tentativas limitadas após falhas e persistência progressiva dos resultados. Como as páginas permanecem em atualização, alterações no HTML, nos endereços ou nos mecanismos de navegação podem exigir adaptações futuras.

A coleta não faz parte de `run_pipeline.py` e deve ser executada separadamente. Os arquivos produzidos pelos coletores passam por normalização antes de serem utilizados pelo script de integração.

---

## Dados de entrada e integração

Os principais arquivos consumidos por `script.py` são:

- `matches.csv` — partidas internacionais e escalações;
- `rankings_fifa.csv` — publicações históricas do ranking masculino da FIFA;
- `valor_mercado_jogadores.csv` — histórico de valores de mercado dos jogadores;
- `paises_siglas_relacao.csv` — correspondência entre nomes de seleções e códigos;
- `Copa2026.csv` — partidas utilizadas na construção da base específica de 2026.

O processo de integração:

1. normaliza nomes e datas;
2. relaciona seleções aos códigos utilizados no ranking;
3. utiliza o ranking disponível até a data de cada partida;
4. utiliza somente valores de mercado registrados até a data da partida;
5. calcula atributos recentes e confrontos diretos em ordem cronológica;
6. gera os arquivos de modelagem.

As principais saídas são:

- `football_matches_ml.csv` — base histórica de treinamento, validação e teste;
- `football_matches_2026.csv` — base preparada para a aplicação dos modelos em 2026.

---

## Dataset principal

O arquivo `football_matches_ml.csv` contém variáveis pré-jogo e a classe observada.

### Identificação e resultado

- `match_id`, `date`, `competition`;
- `home_team`, `away_team`;
- `home_goals`, `away_goals`;
- `match_result`.

### Ranking FIFA

- `home_rank`, `away_rank`;
- `home_points`, `away_points`;
- `rank_diff`, `points_diff`.

### Valores de mercado

- `home_market_value_avg`, `away_market_value_avg`;
- `market_value_avg_diff`;
- `home_found_count`, `away_found_count`.

### Desempenho recente

- `home_recent_win_rate`, `away_recent_win_rate`;
- `home_recent_goals_scored`, `away_recent_goals_scored`;
- `home_recent_goals_conceded`, `away_recent_goals_conceded`.

### Confrontos diretos

- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`;
- `h2h_goal_diff`.

### Contexto temporal

- `year`, `month`;
- `world_cup`.

Os modelos atuais utilizam as mesmas 24 features numéricas. Nomes das equipes, competição, frequência das seleções e flags de seleção conhecida não são usados como preditores.

---

## Convenções metodológicas

### Separação temporal

As partidas são ordenadas cronologicamente. O protocolo comum utiliza:

- treino: partidas até 2021;
- validação: partidas de 2022 e 2023;
- teste: partidas a partir de 2024.

### Prevenção de vazamento temporal

Rankings, valores de mercado, desempenho recente e confrontos diretos são associados ou calculados somente com informações disponíveis até a data da partida correspondente. Imputadores, escaladores e demais transformações são ajustados apenas com os dados permitidos em cada etapa.

### Classes desbalanceadas

O conjunto possui frequências diferentes para vitória visitante, empate e vitória mandante. Os modelos utilizam ponderação de classes ou de amostras durante o treinamento, e o F1-macro é adotado como métrica principal de seleção.

### Baseline

As avaliações incluem um baseline que prevê vitória do mandante para todas as partidas. A comparação permite verificar se os modelos superam a regra baseada somente na classe mais frequente.

### Probabilidades

Os três pipelines utilizam o conjunto de validação para ajustar calibração isotônica por classe. As probabilidades brutas e calibradas são mantidas separadamente para avaliação.

### Explicabilidade

O projeto utiliza abordagens adequadas a cada modelo, incluindo coeficientes padronizados, permutation importance e valores SHAP. Essas medidas descrevem o comportamento preditivo dos modelos e não devem ser interpretadas como efeitos causais.

---

## Modelos disponíveis

### Regressão Logística

Diretório: `ML_LogisticRegression_01/`

### Random Forest

Diretório: `ML_RandomForest_01/`

### XGBoost

Diretório: `ML_XGBoost_01/`

---

## Instalação

Recomenda-se utilizar um ambiente virtual. A partir da raiz do projeto:

```bash
python -m venv .venv
```

No PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Execução do pipeline de dados

Para gerar novamente o dataset histórico a partir das fontes já preparadas:

```bash
python script.py
```

---

## Reprodutibilidade

As versões das bibliotecas estão fixadas em `requirements.txt`, e os scripts de modelagem utilizam `random_state=42` nas operações estocásticas identificadas. As divisões temporais não embaralham as partidas.

Para associar resultados a uma versão específica do projeto, registre o hash do commit utilizado juntamente com os arquivos de métricas. Arquivos de dados, imagens e modelos serializados podem estar excluídos do versionamento; portanto, devem ser regenerados pelos scripts ou preservados separadamente quando necessários.

---

## Observações

- A coleta automatizada depende da disponibilidade e da estrutura das páginas externas.
- Os arquivos de modelagem devem ser regenerados quando as fontes ou as regras de integração forem alteradas.
- Alterações nas features, nos cortes temporais ou nos hiperparâmetros devem ser registradas para manter a rastreabilidade dos experimentos.
- As previsões representam estimativas dos modelos e não garantem o resultado de uma partida.
