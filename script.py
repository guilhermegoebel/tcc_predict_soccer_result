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
from bisect import bisect_right
from collections import defaultdict, deque

# =========================================================
# CONFIGURAÇÕES
# =========================================================

TEST_MODE = False

TEST_MATCH_INDEX = 0

MAX_PLAYERS_PER_TEAM = 5

LAST_MATCHES_FORM = 5

INITIAL_ELO = 1500

ELO_K = 20

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
    'missing_player_list': 0,
    'unmapped_fifa_codes': 0
}

def debug(level, message):
    if DEBUG:
        print(f'[{level}] {message}')

def parse_market_value(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    value = value.replace('€', '')

    multiplier = 1

    if value.endswith('m'):
        multiplier = 1_000_000
        value = value[:-1]

    elif value.endswith('k'):
        multiplier = 1_000
        value = value[:-1]

    try:
        return float(value) * multiplier
    except:
        return np.nan

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

market_value['market_value_str'] = (
    market_value['market_value_str']
    .apply(parse_market_value)
)

filtro_remocao = (
    (market_value['tm_id'] == 'NAO_ENCONTRADO')
    |
    (market_value['market_value_str'] == '-')
    |
    (market_value['market_value_str'] == 'sem dados')
    |
    (market_value['market_value_str'].isna())
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

market_value['market_value_str'] = (
    market_value['market_value_str']
    .apply(parse_market_value)
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

    matches = matches.iloc[
        [TEST_MATCH_INDEX]
    ].copy()
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

market_value[
    'player_norm'
] = (
    market_value['player_search_name']
    .astype(str)
    .str.lower()
    .str.strip()
)

market_value = market_value.sort_values(
    'market_date'
)

market_cache = {}

for player, group in market_value.groupby(
    'player_norm'
):

    market_cache[player] = list(

        zip(
            group['market_date'],
            group['market_value_str']
        )
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

        debug(
            'RANKING',
            f'Nenhum ranking encontrado para "{team}"'
        )

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

        debug(
            'RANKING',
            f'{team} não possui ranking antes de {match_date}'
        )

        debug(
            'RANKING',
            f'Primeira data disponível: {dates[0]}'
        )

        debug(
            'RANKING',
            f'Última data disponível: {dates[-1]}'
        )

        return np.nan, np.nan

    selected_date, latest_rank, latest_points = (
        rankings_list[pos]
    )

    return latest_rank, latest_points

def get_market_value(players, match_date):

    total_value = 0

    if not isinstance(players, str):

        debug_stats[
            'missing_player_list'
        ] += 1

        return 0

    players = players.split('|')

    jogador_nao_encontrado = 0
    jogador_sem_valor = 0
    jogadores_utilizados = 0

    match_date = pd.Timestamp(match_date)

    for player in reversed(players):

        if jogadores_utilizados >= MAX_PLAYERS_PER_TEAM:
            break

        player = player.lower().strip()

        player_history = market_cache.get(
            player,
            []
        )

        if not player_history:

            debug_stats[
                'market_player_not_found'
            ] += 1

            jogador_nao_encontrado += 1

            continue

        latest_value = None
        latest_date = None

        for m_date, value in player_history:

            if pd.isna(m_date):
                continue

            if m_date > match_date:
                break

            latest_date = m_date
            latest_value = value

        if latest_value is None:

            jogador_sem_valor += 1

            debug_stats[
                'market_value_not_found'
            ] += 1

            continue

        total_value += latest_value
        jogadores_utilizados += 1

    """
    print(
        f'Jogadores válidos={jogadores_utilizados} | '
        f'Sem histórico={jogador_nao_encontrado} | '
        f'Sem valor na data={jogador_sem_valor} | '
        f'Valor total={total_value:,.0f}'
    )
    """

    return total_value

def calculate_recent_form(team):

    recent = team_recent_matches[team]

    if len(recent) == 0:

        return (
            0,
            0,
            0
        )

    wins = 0

    goals_scored = 0

    goals_conceded = 0

    for item in recent:

        wins += item['win']

        goals_scored += item[
            'goals_scored'
        ]

        goals_conceded += item[
            'goals_conceded'
        ]

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

# =========================================================
# 11. PROCESSAMENTO CRONOLÓGICO
# =========================================================

print('Processando partidas...')

matches = matches.sort_values('date')

processed_rows = []

for idx, row in matches.iterrows():

    match_date = row['date']

    home_team = row['home_team']

    away_team = row['away_team']

    home_goals = row['home_goals']

    away_goals = row['away_goals']

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
    
    home_market = get_market_value(
        row['home_players'],
        match_date
    )

    away_market = get_market_value(
        row['away_players'],
        match_date
    )

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

    expected_home = expected_result(
        home_elo,
        away_elo
    )

    expected_away = expected_result(
        away_elo,
        home_elo
    )

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

    new_home_elo = (
        home_elo +
        ELO_K * (
            home_score - expected_home
        )
    )

    new_away_elo = (
        away_elo +
        ELO_K * (
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
