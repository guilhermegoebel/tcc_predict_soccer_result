# =========================================================
# DATASET OTIMIZADO PARA ML DE FUTEBOL
# =========================================================
# OTIMIZAÇÕES:
# - cache de jogadores
# - cache de partidas por seleção
# - cache H2H
# - processamento cronológico único
# - redução de filtros em dataframe
# - uso de dicionários O(1)
# =========================================================

import sys

import pandas as pd
import numpy as np
import csv
import time
import unicodedata
import statistics
from bisect import bisect_right
from collections import defaultdict, deque

# =========================================================
# CONFIGURAÇÕES
# =========================================================

TEST_MODE = False

# Quantidade de partidas processadas quando TEST_MODE = True.
# Se quiser processar só 1 partida específica, use TEST_MATCH_INDEX
# e defina TEST_N_MATCHES = 1.
TEST_N_MATCHES = 10

TEST_MATCH_INDEX = 0

# Quando True, imprime no console a comparação de nomes de jogadores:
# nome original (do CSV de partidas) -> nome normalizado -> se foi
# encontrado no cache de valores de mercado -> valor utilizado.
# É automaticamente ativado quando TEST_MODE = True (ver abaixo),
# mas pode ser forçado manualmente aqui também.
SHOW_PLAYER_MATCHING = False

MAX_PLAYERS_PER_TEAM = 5

LAST_MATCHES_FORM = 5

INITIAL_ELO = 1500

# K base, usado apenas quando ENABLE_COMPETITION_WEIGHT = False
ELO_K = 20

# Se True, o peso (K) de cada partida varia conforme a importância da
# competição (amistoso pesa menos, fase final de grande torneio pesa
# mais), em vez de usar sempre o mesmo ELO_K fixo.
ENABLE_COMPETITION_WEIGHT = True

# Se True, aplica o multiplicador clássico de margem de vitória
# (diferença de gols) na atualização do Elo: vencer por vários gols
# de diferença altera o rating mais do que vencer por 1 gol.
ENABLE_GOAL_DIFF_MULTIPLIER = True

# Se True, soma HOME_ADVANTAGE ao rating do mandante apenas para
# calcular o resultado esperado (não altera o rating "bruto" salvo
# como feature). Sem isso, o Elo tende a confundir vantagem de jogar
# em casa com força real do time.
ENABLE_HOME_ADVANTAGE = True
HOME_ADVANTAGE = 100

# Se True, no início de cada novo ano regride uma fração do rating de
# cada time de volta para o INITIAL_ELO, refletindo a maior incerteza
# sobre times que ficaram muito tempo sem jogar.
ENABLE_ANNUAL_REGRESSION = True
ANNUAL_REGRESSION_FACTOR = 0.30

DEBUG = True

debug_stats = {
    'invalid_match_dates': 0,
    'invalid_ranking_dates': 0,
    'invalid_market_dates': 0,
    'invalid_scores': 0,
    'ranking_not_found': 0,
    'ranking_before_date_not_found': 0,
    'market_player_not_found': 0,
    'market_value_not_found': 0,
    'market_value_fallback_nearest': 0,
    'market_incomplete_5_players': 0,
    'missing_player_list': 0,
    'unmapped_fifa_codes': 0
}

def debug(level, message):
    if DEBUG:
        print(f'[{level}] {message}')

def parse_market_value(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    text = text.replace('€', '')
    text = text.replace('$', '')
    text = text.replace(' ', '')

    text = text.lower()

    if text in {'', '-', 'nan', 'none', 'na', 'semdados', 'sem dados'}:
        return np.nan

    multiplier = 1

    if text.endswith('m'):
        multiplier = 1_000_000
        text = text[:-1]

    elif text.endswith('k'):
        multiplier = 1_000
        text = text[:-1]

    if ',' in text and '.' in text:
        text = text.replace('.', '').replace(',', '.')
    elif ',' in text:
        text = text.replace(',', '.')

    try:
        return float(text) * multiplier
    except Exception:
        return np.nan


def normalize_player_name(player):

    if pd.isna(player):
        return ''

    text = str(player).strip().lower()

    # Remove acentuação/diacríticos (é -> e, ñ -> n, ã -> a, ô -> o, etc.)
    # tanto para o nome que está sendo buscado quanto para os nomes
    # armazenados no cache de valores de mercado.
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.replace('.', '')
    text = text.replace('-', '')
    text = text.replace("'", '')
    text = text.replace(' ', '')

    return text

# =========================================================
# TIMER GLOBAL
# =========================================================

global_start = time.time()

# =========================================================
# 1. LEITURA DOS CSVs
# =========================================================

print('Carregando datasets...')

matches = pd.read_csv(
    'matches.csv',
    sep=',',
    engine='python',
    encoding='utf-8',
    quoting=csv.QUOTE_MINIMAL,
    #on_bad_lines='skip', #rever se isso faz sentido, quais dados estamos perdendo aqui?
    usecols=[
        'match_id',
        'date',
        'home_team',
        'away_team',
        'score',
        'competition',
        'home_players',
        'away_players'
    ]
)

# Normaliza a coluna para evitar problemas com espaços
matches['score'] = (
    matches['score']
    .astype(str)
    .str.strip()
)

filtro_remocao = (
    (matches['score'] == 'canc. Status: Cancelled')
    |
    (matches['score'].isna())
)

removidos_matches = filtro_remocao.sum()

# Mantém apenas os registros válidos
matches = matches[
    ~filtro_remocao
].copy()

print(
    f'  Registros de partidas removidos na limpeza: {removidos_matches:,}'
)

print(
    f'  Registros restantes de partidas: {len(matches):,}'
)

rankings = pd.read_csv(
    'rankings_fifa.csv',
    sep=',',
    engine='python',
    encoding='utf-8',
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines='skip',
    usecols=[
        'year',
        'date_label',
        'rank',
        'team',
        'points'
    ]
)

market_value = pd.read_csv(
    'valor_mercado_jogadores.csv',
    sep=',',
    engine='python',
    encoding='utf-8',
    quoting=csv.QUOTE_MINIMAL,
    on_bad_lines='skip',
    usecols=[
        'tm_id',
        'player_search_name',
        'tm_name',
        'date',
        'market_value_str'
    ]
)

# Normaliza a coluna para evitar problemas com espaços
market_value['tm_id'] = (
    market_value['tm_id']
    .astype(str)
    .str.strip()
)

market_value['market_value_num'] = (
    market_value['market_value_str']
    .apply(parse_market_value)
)

filtro_remocao = (
    (market_value['tm_id'] == 'NAO_ENCONTRADO')
)

removidos = filtro_remocao.sum()

# Mantém apenas os registros válidos
market_value = market_value[
    ~filtro_remocao
].copy()

print(
    f'  Registros de mercado removidos na limpeza: {removidos:,}'
)

print(
    f'  Registros restantes do mercado: {len(market_value):,}'
)

print('Datasets carregados. ============================')

# =========================================================
# CONVERTER TIPOS NUMÉRICOS
# =========================================================

rankings['rank'] = pd.to_numeric(
    rankings['rank'],
    errors='coerce'
)

rankings['points'] = pd.to_numeric(
    rankings['points'],
    errors='coerce'
)

# =========================================================
# 2. NORMALIZAÇÃO DATAS
# =========================================================

print('Normalizando datas...')

matches['date'] = pd.to_datetime(
    matches['date'],
    errors='coerce'
)

rankings['ranking_date'] = pd.to_datetime(
    rankings['date_label'].astype(str)
    + ' '
    + rankings['year'].astype(str),
    format='%d %B %Y',
    errors='coerce'
)

market_value['market_date'] = pd.to_datetime(
    market_value['date'],
    errors='coerce'
)

invalid_matches_dates = matches['date'].isna().sum()
invalid_dates = matches[
    matches['date'].isna()
]

""" print("\n=== PARTIDAS COM DATA INVÁLIDA ===")
print(invalid_dates[['match_id', 'date']]) """

if invalid_matches_dates:

    debug_stats['invalid_match_dates'] = invalid_matches_dates

    debug(
        'ERRO',
        f'{invalid_matches_dates} partidas possuem datas inválidas'
    )

invalid_ranking_dates = rankings[
    'ranking_date'
].isna().sum()

if invalid_ranking_dates:

    debug_stats['invalid_ranking_dates'] = invalid_ranking_dates

    debug(
        'ERRO',
        f'{invalid_ranking_dates} rankings possuem datas inválidas'
    )

""" 
invalid_rankings = rankings[
    rankings['ranking_date'].isna()
]

print("\n=== RANKINGS COM DATA INVÁLIDA ===")
print(
    invalid_rankings[
        ['year', 'date_label', 'team']
    ]
) """

missing_market_dates = market_value[
    'market_date'
].isna().sum()

if missing_market_dates:

    debug(
        'INFO',
        f'{missing_market_dates} jogadores sem histórico de valor de mercado'
    )

invalid_market = market_value[
    market_value['market_date'].isna()
]

""" print("\n=== EXEMPLOS MARKET VALUE ===")

print(
    invalid_market[
        ['player_search_name', 'date']
    ].head(20)
) """

# =========================================================
# 3. EXTRAÇÃO GOLS
# =========================================================

score_extract = matches['score'].astype(str).str.extract(
    r'(\d+)\s*:\s*(\d+)'
)

matches['home_goals'] = pd.to_numeric(
    score_extract[0],
    errors='coerce'
)

matches['away_goals'] = pd.to_numeric(
    score_extract[1],
    errors='coerce'
)

""" 
invalid_scores = (
    matches['home_goals'].isna() | matches['away_goals'].isna()
)

print("\n=== SCORES INVÁLIDOS ===")

print(
    matches.loc[
        invalid_scores,
        ['match_id', 'date', 'home_team', 'away_team', 'score']
    ]
) """

# =========================================================
# 4. MODO TESTE
# =========================================================

if TEST_MODE:

    matches = (
        matches
        .sort_values('date')
        .iloc[TEST_MATCH_INDEX:TEST_MATCH_INDEX + TEST_N_MATCHES]
        .copy()
    )

    # Em modo teste, liga automaticamente a exibição da comparação
    # de nomes de jogadores (pode ser desligada manualmente na
    # seção de configurações se não for necessária).
    SHOW_PLAYER_MATCHING = True

    debug(
        'TESTE',
        f'Modo teste ativo: processando {len(matches)} partida(s) '
        f'a partir do índice {TEST_MATCH_INDEX}'
    )
# =========================================================
# 5. NORMALIZAÇÃO FIFA
# =========================================================

from collections import defaultdict

print('Normalizando nomes FIFA...')

# Ex:
# FRA -> {"France", "France (B)"}
# POR -> {"Portugal"}
fifa_aliases = defaultdict(set)

with open(
    "paises_siglas_relacao.csv",
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        sigla = row["sigla"].strip()
        pais = row["pais"].strip()

        if sigla and pais:

            fifa_aliases[sigla].add(
                pais
            )

# DEBUG
print(
    f'FRA -> {fifa_aliases.get("FRA", set())}'
)

# Mantém um nome principal para compatibilidade
rankings['team_name'] = rankings[
    'team'
].apply(
    lambda x: (
        sorted(fifa_aliases[x])[0]
        if x in fifa_aliases
        else x
    )
)

unmapped = rankings[
    ~rankings['team'].isin(
        fifa_aliases.keys()
    )
]['team'].unique()

if len(unmapped):

    debug_stats[
        'unmapped_fifa_codes'
    ] = len(unmapped)

    debug(
        'ERRO',
        f'{len(unmapped)} siglas FIFA não encontradas'
    )

    debug(
        'INFO',
        f'Primeiras siglas sem mapeamento: {list(unmapped[:20])}'
    )
# =========================================================
# 6. OTIMIZAÇÃO TIPOS
# =========================================================

categorical_columns = [
    'home_team',
    'away_team',
    'competition'
]

for col in categorical_columns:
    matches[col] = matches[col].astype(
        'category'
    )

# =========================================================
# 7. CACHE RANKINGS
# =========================================================

print('Criando cache rankings...')

rankings = rankings.sort_values(
    'ranking_date'
)

ranking_cache = defaultdict(list)

for _, row in rankings.iterrows():

    sigla = row['team']

    aliases = fifa_aliases.get(
        sigla,
        {sigla}
    )

    for alias in aliases:

        ranking_cache[
            alias
        ].append(
            (
                row['ranking_date'],
                row['rank'],
                row['points']
            )
        )


# =========================================================
# 8. CACHE MARKET VALUE
# =========================================================

print('Criando cache valores mercado...')

market_value['player_norm'] = market_value[
    'player_search_name'
].apply(normalize_player_name)

market_value['tm_name_norm'] = market_value[
    'tm_name'
].apply(normalize_player_name)

market_value = market_value.sort_values(
    'market_date'
)

market_cache = defaultdict(list)

for _, row in market_value.iterrows():

    player_names = [
        name for name in [
            row['player_norm'],
            row['tm_name_norm']
        ] if name
    ]

    if not player_names:
        continue

    value = row['market_value_num']

    if pd.isna(value):
        continue

    for player_name in set(player_names):
        market_cache[player_name].append(
            (
                row['market_date'],
                value
            )
        )

for player_name in market_cache:
    market_cache[player_name].sort(
        key=lambda item: item[0]
    )

# =========================================================
# 9. ESTADOS DINÂMICOS
# =========================================================

team_recent_matches = defaultdict(
    lambda: deque(maxlen=LAST_MATCHES_FORM)
)

elo_ratings = defaultdict(
    lambda: INITIAL_ELO
)

h2h_cache = defaultdict(list)

# =========================================================
# 10. FUNÇÕES AUXILIARES
# =========================================================

def get_latest_ranking(team, match_date):

    rankings_list = ranking_cache.get(
        team,
        []
    )

    if not rankings_list:

        debug_stats[
            'ranking_not_found'
        ] += 1

        """ debug(
            'RANKING',
            f'Nenhum ranking encontrado para "{team}"'
        ) """

        # DEBUG EXTRA
        similares = [
            t for t in ranking_cache.keys()
            if team.lower() in str(t).lower()
            or str(t).lower() in team.lower()
        ][:20]

        """        debug(
                    'RANKING',
                    f'Times parecidos encontrados: {similares}'
                )
        """
        return np.nan, np.nan

    dates = [
        r_date
        for r_date, _, _ in rankings_list
        if pd.notna(r_date)
    ]

    if not dates:

        debug(
            'RANKING',
            f'{team} possui registros mas todas as datas são inválidas'
        )

        debug(
            'RANKING',
            f'Primeiros registros: {rankings_list[:5]}'
        )

        return np.nan, np.nan

    match_date = pd.Timestamp(match_date)

    pos = bisect_right(
        dates,
        match_date
    ) - 1

    if pos < 0:

        debug_stats[
            'ranking_before_date_not_found'
        ] += 1

        """ debug(
            'RANKING',
            f'{team} não possui ranking antes de {match_date}'
        ) """

        """ debug(
            'RANKING',
            f'Primeira data disponível: {dates[0]}'
        ) """

        """ debug(
            'RANKING',
            f'Última data disponível: {dates[-1]}'
        ) """

        return np.nan, np.nan

    selected_date, latest_rank, latest_points = (
        rankings_list[pos]
    )

    return latest_rank, latest_points

def pick_market_value(player_history, match_date):
    """
    player_history: lista de tuplas (data, valor) ordenada por data
    crescente para um jogador.

    Retorna (valor, data_usada, é_anterior_ou_igual):
    - Preferência: o valor mais recente com data <= match_date.
    - Caso não exista nenhum valor anterior/igual à data da partida,
      usa o valor mais próximo disponível (o primeiro registro
      seguinte cronologicamente, já que a lista está ordenada).
    """

    best_before = None
    best_after = None

    for m_date, value in player_history:

        if pd.isna(m_date):
            continue

        if m_date <= match_date:
            best_before = (m_date, value)

        else:
            best_after = (m_date, value)
            break

    if best_before is not None:
        return best_before[1], best_before[0], True

    if best_after is not None:
        return best_after[1], best_after[0], False

    return None, None, False


def get_market_value(players, match_date, team_label=None):

    if not isinstance(players, str):

        debug_stats[
            'missing_player_list'
        ] += 1

        return 0.0, 0

    player_list = [
        player for player in players.split('|')
        if str(player).strip()
    ]

    if not player_list:
        return 0.0, 0

    match_date = pd.Timestamp(match_date)

    # Ordem de tentativa: primeiro os últimos MAX_PLAYERS_PER_TEAM
    # jogadores da lista. Se não completarem 5 valores encontrados,
    # busca no restante do elenco daquela partida (do mais próximo do
    # final para o início), até encontrar 5 jogadores com valor ou
    # esgotar a lista de jogadores da partida.
    last_n = player_list[-MAX_PLAYERS_PER_TEAM:]
    remaining = list(
        reversed(
            player_list[:-MAX_PLAYERS_PER_TEAM]
        )
    )

    ordered_candidates = last_n + remaining

    total_value = 0.0
    found_count = 0

    if SHOW_PLAYER_MATCHING:

        print(
            f'\n--- Comparação de nomes de jogadores '
            f'({team_label or "?"} | partida em {match_date.date()}) ---'
        )

    for player in ordered_candidates:

        if found_count >= MAX_PLAYERS_PER_TEAM:
            break

        player_key = normalize_player_name(player)

        if not player_key:
            continue

        player_history = market_cache.get(
            player_key,
            []
        )

        if not player_history:

            debug_stats[
                'market_player_not_found'
            ] += 1

            if SHOW_PLAYER_MATCHING:

                print(
                    f'  "{player}" -> "{player_key}" '
                    f'=> NÃO ENCONTRADO no cache '
                    f'(buscando próximo jogador do elenco)'
                )

            continue

        value, used_date, is_before_or_equal = pick_market_value(
            player_history,
            match_date
        )

        if value is None:

            # Não deveria ocorrer já que player_history não está vazio,
            # mas mantemos por segurança.
            debug_stats[
                'market_value_not_found'
            ] += 1

            continue

        if not is_before_or_equal:

            debug_stats[
                'market_value_fallback_nearest'
            ] += 1

        total_value += float(value)
        found_count += 1

        if SHOW_PLAYER_MATCHING:

            tag = (
                'valor na data/anterior'
                if is_before_or_equal
                else 'SEM valor anterior -> usando o mais próximo (fallback)'
            )

            print(
                f'  "{player}" -> "{player_key}" '
                f'=> encontrado ({tag}, referência: {used_date.date()})! '
                f'valor = €{value:,.0f}  '
                f'[{found_count}/{MAX_PLAYERS_PER_TEAM}]'
            )

    if found_count < MAX_PLAYERS_PER_TEAM:

        debug_stats[
            'market_incomplete_5_players'
        ] += 1

        if SHOW_PLAYER_MATCHING:

            print(
                f'  Aviso: apenas {found_count}/{MAX_PLAYERS_PER_TEAM} '
                f'jogadores com valor de mercado encontrados nesta '
                f'partida (elenco possuía {len(player_list)} jogador(es))'
            )

    return total_value, found_count


def calculate_recent_form(team):

    recent = list(team_recent_matches[team])[-LAST_MATCHES_FORM:]

    if not recent:

        return (
            0.0,
            0.0,
            0.0
        )

    wins = sum(item['win'] for item in recent)
    goals_scored = sum(item['goals_scored'] for item in recent)
    goals_conceded = sum(item['goals_conceded'] for item in recent)

    total = len(recent)

    return (
        wins / total,
        goals_scored / total,
        goals_conceded / total
    )

# =========================================================

def expected_result(rating_a, rating_b):

    return 1 / (
        1 + 10 ** (
            (rating_b - rating_a) / 400
        )
    )


def get_competition_weight(competition):
    """
    Deriva o peso (K) da partida a partir do texto da competição,
    seguindo a mesma lógica usada por sistemas de referência como o
    World Football Elo Ratings (eloratings.net):
      - Fase final de Copa do Mundo: peso mais alto (60)
      - Finais de torneios continentais/grandes torneios: peso alto (50)
      - Eliminatórias (Copa do Mundo, continentais): peso médio (40)
      - Amistosos: peso mais baixo (20)
      - Demais torneios/copas regionais: peso médio-baixo (30)
    """

    comp = str(competition).lower()

    if 'friendl' in comp:
        return 20

    if 'qualification' in comp:
        return 40

    if 'world cup' in comp and 'final tournament' in comp:
        return 60

    if (
        ('championship' in comp or 'cup' in comp)
        and 'final tournament' in comp
    ):
        return 50

    return 30


def goal_diff_multiplier(goal_diff):
    """
    Multiplicador clássico de margem de vitória (goal difference),
    usado no World Football Elo Ratings:
      - Diferença de 0 ou 1 gol: multiplicador 1.0
      - Diferença de 2 gols: multiplicador 1.5
      - Diferença de 3+ gols: multiplicador (11 + diferença) / 8
    """

    diff = abs(goal_diff)

    if diff <= 1:
        return 1.0

    if diff == 2:
        return 1.5

    return (11 + diff) / 8

# =========================================================
# 11. PROCESSAMENTO CRONOLÓGICO
# =========================================================

print('Processando partidas...')

matches = matches.sort_values('date')

processed_rows = []

# Quantidade de jogadores efetivamente contabilizados no valor de
# mercado: um valor por time (home e away separados) e um valor
# por jogo (soma de home + away). Usado para calcular média/mediana
# ao final do processamento.
players_found_per_team = []
players_found_per_match = []

# Controla em qual ano estamos, para aplicar a regressão anual do
# Elo à média (ver ENABLE_ANNUAL_REGRESSION) sempre que o processamento
# cronológico avança para um novo ano.
current_elo_year = None

for idx, row in matches.iterrows():

    match_date = row['date']

    home_team = row['home_team']

    away_team = row['away_team']

    home_goals = row['home_goals']

    away_goals = row['away_goals']

    # =====================================================
    # REGRESSÃO ANUAL DO ELO À MÉDIA
    # =====================================================

    if ENABLE_ANNUAL_REGRESSION:

        match_year = match_date.year

        if current_elo_year is None:
            current_elo_year = match_year

        while match_year > current_elo_year:

            for team in list(elo_ratings.keys()):

                elo_ratings[team] = (
                    INITIAL_ELO
                    + (elo_ratings[team] - INITIAL_ELO)
                    * (1 - ANNUAL_REGRESSION_FACTOR)
                )

            current_elo_year += 1

    # =====================================================
    # RANKINGS
    # =====================================================

    """     print(f'Processando partida: {home_team} vs {away_team}'
          f' em {match_date.date()}',
          f'\n Jogadores: {row["home_players"]}'
    ) """

    home_rank, home_points = (
        get_latest_ranking(
            home_team,
            match_date
        )
    )

    away_rank, away_points = (
        get_latest_ranking(
            away_team,
            match_date
        )
    )

    # =====================================================
    # MARKET VALUE
    # =====================================================
    
    home_market, home_found_count = get_market_value(
        row['home_players'],
        match_date,
        team_label=f'HOME: {home_team}'
    )

    away_market, away_found_count = get_market_value(
        row['away_players'],
        match_date,
        team_label=f'AWAY: {away_team}'
    )

    players_found_per_team.append(home_found_count)
    players_found_per_team.append(away_found_count)
    players_found_per_match.append(home_found_count + away_found_count)

    # =====================================================
    # FORMA RECENTE
    # =====================================================

    (
        home_win_rate,
        home_gs,
        home_gc
    ) = calculate_recent_form(
        home_team
    )

    (
        away_win_rate,
        away_gs,
        away_gc
    ) = calculate_recent_form(
        away_team
    )

    # =====================================================
    # H2H
    # =====================================================

    h2h_key = tuple(sorted([
        home_team,
        away_team
    ]))

    previous_h2h = h2h_cache[h2h_key]

    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0
    h2h_goal_diff = 0

    for item in previous_h2h:

        if item['winner'] == home_team:
            h2h_home_wins += 1

        elif item['winner'] == away_team:
            h2h_away_wins += 1

        else:
            h2h_draws += 1

        if item['perspective'] == home_team:
            h2h_goal_diff += item[
                'goal_diff'
            ]

    # =====================================================
    # ELO
    # =====================================================

    home_elo = elo_ratings[
        home_team
    ]

    away_elo = elo_ratings[
        away_team
    ]

    # A vantagem de mandante e somada apenas para calcular o
    # resultado esperado (nao altera o rating 'bruto' que e salvo
    # como feature no dataset final).
    home_elo_for_expectation = (
        home_elo + HOME_ADVANTAGE
        if ENABLE_HOME_ADVANTAGE
        else home_elo
    )

    expected_home = expected_result(
        home_elo_for_expectation,
        away_elo
    )

    # Simetrico por construcao: expected_away = 1 - expected_home,
    # mesmo com a vantagem de mandante aplicada apenas do lado home.
    expected_away = 1 - expected_home

    if home_goals > away_goals:

        home_score = 1
        away_score = 0

        winner = home_team

    elif home_goals < away_goals:

        home_score = 0
        away_score = 1

        winner = away_team

    else:

        home_score = 0.5
        away_score = 0.5

        winner = 'draw'

    match_weight = (
        get_competition_weight(row['competition'])
        if ENABLE_COMPETITION_WEIGHT
        else ELO_K
    )

    g_multiplier = (
        goal_diff_multiplier(home_goals - away_goals)
        if ENABLE_GOAL_DIFF_MULTIPLIER
        else 1.0
    )

    new_home_elo = (
        home_elo +
        match_weight * g_multiplier * (
            home_score - expected_home
        )
    )

    new_away_elo = (
        away_elo +
        match_weight * g_multiplier * (
            away_score - expected_away
        )
    )

    elo_ratings[
        home_team
    ] = new_home_elo

    elo_ratings[
        away_team
    ] = new_away_elo

    # =====================================================
    # TARGET
    # =====================================================

    if home_goals > away_goals:
        match_result = 2

    elif home_goals < away_goals:
        match_result = 0

    else:
        match_result = 1

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    processed_rows.append({

        'match_id':
        row['match_id'],

        'date':
        match_date,

        'competition':
        row['competition'],

        'home_team':
        home_team,

        'away_team':
        away_team,

        'home_goals':
        home_goals,

        'away_goals':
        away_goals,

        'home_rank':
        home_rank,

        'away_rank':
        away_rank,

        'home_points':
        home_points,

        'away_points':
        away_points,

        'rank_diff':
        (
            away_rank - home_rank
            if pd.notna(home_rank)
            and pd.notna(away_rank)
            else np.nan
        ),

        'points_diff':
        (
            home_points - away_points
            if pd.notna(home_points)
            and pd.notna(away_points)
            else np.nan
        ),

        'home_market_value':
        home_market,

        'away_market_value':
        away_market,

        'market_value_diff':
        home_market - away_market,

        'home_recent_win_rate':
        home_win_rate,

        'away_recent_win_rate':
        away_win_rate,

        'home_recent_goals_scored':
        home_gs,

        'away_recent_goals_scored':
        away_gs,

        'home_recent_goals_conceded':
        home_gc,

        'away_recent_goals_conceded':
        away_gc,

        'h2h_home_wins':
        h2h_home_wins,

        'h2h_away_wins':
        h2h_away_wins,

        'h2h_draws':
        h2h_draws,

        'h2h_goal_diff':
        h2h_goal_diff,

        'home_elo':
        home_elo,

        'away_elo':
        away_elo,

        'elo_diff':
        home_elo - away_elo,

        'year':
        match_date.year,

        'month':
        match_date.month,

        'is_world_cup_year':
        1 if match_date.year in [
            2010,
            2014,
            2018,
            2022,
            2026
        ] else 0,

        'match_result':
        match_result
    })

    # =====================================================
    # ATUALIZAR ESTADO RECENTE
    # =====================================================

    home_recent = {

        'win':
        1 if home_goals > away_goals else 0,

        'goals_scored':
        home_goals,

        'goals_conceded':
        away_goals
    }

    away_recent = {

        'win':
        1 if away_goals > home_goals else 0,

        'goals_scored':
        away_goals,

        'goals_conceded':
        home_goals
    }

    team_recent_matches[
        home_team
    ].append(home_recent)

    team_recent_matches[
        away_team
    ].append(away_recent)

    # =====================================================
    # ATUALIZAR H2H
    # =====================================================

    h2h_cache[h2h_key].append({

        'winner':
        winner,

        'goal_diff':
        home_goals - away_goals,

        'perspective':
        home_team
    })

# =========================================================
# 12. DATAFRAME FINAL
# =========================================================

print('Criando dataframe final...')

df = pd.DataFrame(
    processed_rows
)


# =========================================================
# 14. SALVAR
# =========================================================

print('\n===================================')
print('DIAGNÓSTICO DE QUALIDADE')
print('===================================')

for key, value in debug_stats.items():

    print(
        f'{key}: {value:,}'
    )

print('\nValores ausentes no dataset final')

print(
    f'home_rank: {df["home_rank"].isna().sum():,}'
)

print(
    f'away_rank: {df["away_rank"].isna().sum():,}'
)

print(
    f'home_points: {df["home_points"].isna().sum():,}'
)

print(
    f'away_points: {df["away_points"].isna().sum():,}'
)

print(
    f'home_market_value = 0: '
    f'{(df["home_market_value"] == 0).sum():,}'
)

print(
    f'away_market_value = 0: '
    f'{(df["away_market_value"] == 0).sum():,}'
)

print('\n===================================')
print('JOGADORES CONTABILIZADOS NO VALOR DE MERCADO')
print('===================================')

if players_found_per_team:

    print(
        f'Por time (home e away separados, meta = {MAX_PLAYERS_PER_TEAM}):'
    )

    print(
        f'  Média:   {statistics.mean(players_found_per_team):.2f}'
    )

    print(
        f'  Mediana: {statistics.median(players_found_per_team):.2f}'
    )

if players_found_per_match:

    print(
        f'\nPor jogo (home + away somados, '
        f'meta = {MAX_PLAYERS_PER_TEAM * 2}):'
    )

    print(
        f'  Média:   {statistics.mean(players_found_per_match):.2f}'
    )

    print(
        f'  Mediana: {statistics.median(players_found_per_match):.2f}'
    )

output_file = 'football_matches_ml.csv'

df.to_csv(
    output_file,
    index=False
)

# =========================================================
# 15. RESULTADO
# =========================================================

global_end = time.time()

print('\n===================================')
print('DATASET CRIADO')
print('===================================')

print(df.head())

print('\nShape:')
print(df.shape)

print(
    f'\nTempo total: '
    f'{global_end - global_start:.2f}s'
)

print(f'\nArquivo salvo: {output_file}')