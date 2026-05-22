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

import pandas as pd
import numpy as np
import csv
import time
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
    on_bad_lines='skip', #rever se isso faz sentido, quais dados estamos perdendo aqui?
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
        'player_search_name',
        'date',
        'market_value_eur'
    ]
)

print('Datasets carregados.')

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

market_value['market_value_eur'] = pd.to_numeric(
    market_value['market_value_eur'],
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

print('Normalizando nomes FIFA...')

fifa_country_map = {

    'BRA': 'Brazil',
    'ARG': 'Argentina',
    'FRA': 'France',
    'GER': 'Germany',
    'ESP': 'Spain',
    'POR': 'Portugal',
    'ENG': 'England',
    'ITA': 'Italy',
    'NED': 'Netherlands',
    'BEL': 'Belgium',
    'CRO': 'Croatia',
    'URU': 'Uruguay',
    'USA': 'United States',
    'MEX': 'Mexico',
    'JPN': 'Japan',
    'KOR': 'South Korea',
    'MAR': 'Morocco',
    'SUI': 'Switzerland',
    'DEN': 'Denmark',
    'SEN': 'Senegal',
    'POL': 'Poland',
    'SRB': 'Serbia',
    'CMR': 'Cameroon',
    'CAN': 'Canada',
    'AUS': 'Australia',
    'CRC': 'Costa Rica',
    'QAT': 'Qatar',
    'GHA': 'Ghana',
    'TUN': 'Tunisia',
    'IRN': 'Iran',
    'KSA': 'Saudi Arabia',
    'ECU': 'Ecuador',
    'WAL': 'Wales'
}

rankings['team_name'] = rankings[
    'team'
].map(fifa_country_map)

rankings['team_name'] = rankings[
    'team_name'
].fillna(rankings['team'])

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

    ranking_cache[
        row['team_name']
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
            group['market_value_eur']
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

    latest_rank = np.nan
    latest_points = np.nan

    for r_date, rank, points in rankings_list:

        if r_date <= match_date:

            latest_rank = rank
            latest_points = points

        else:
            break

    return latest_rank, latest_points

# =========================================================

def get_market_value(players, match_date):

    total_value = 0

    if not isinstance(players, str):
        return 0

    players = players.split('|')[
        -MAX_PLAYERS_PER_TEAM:
    ]

    for player in players:

        player = player.lower().strip()

        player_history = market_cache.get(
            player,
            []
        )

        latest_value = 0

        for m_date, value in player_history:

            if pd.isna(m_date):
                continue

            if m_date <= match_date:

                try:
                    latest_value = float(value)
                except:
                    latest_value = 0

            else:
                break

        total_value += latest_value

    return total_value

# =========================================================

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
