import pandas as pd

# Carrega o dataset
df = pd.read_csv(
    'football_matches_ml.csv'
)

# Seleções classificadas para a Copa de 2026
world_cup_2026_teams = {
    'Canada',
    'Mexico',
    'USA',

    'Australia',
    'Iraq',
    'IR Iran',
    'Japan',
    'Jordan',
    'Korea Republic',
    'Qatar',
    'Saudi Arabia',
    'Uzbekistan',

    'Algeria',
    'Cabo Verde',
    'Congo DR',
    "Côte d'Ivoire",
    'Egypt',
    'Ghana',
    'Morocco',
    'Senegal',
    'South Africa',
    'Tunisia',

    'Curacao',
    'Haiti',
    'Panama',

    'Argentina',
    'Brazil',
    'Colombia',
    'Ecuador',
    'Paraguay',
    'Uruguay',

    'New Zealand',

    'Austria',
    'Belgium',
    'Bosnia and Herzegovina',
    'Croatia',
    'Czechia',
    'England',
    'France',
    'Germany',
    'Netherlands',
    'Norway',
    'Portugal',
    'Scotland',
    'Spain',
    'Sweden',
    'Switzerland',
    'Türkiye'
}

# Mantém apenas partidas em que um dos times são classificados
df_filtered = df[
    df['home_team'].isin(world_cup_2026_teams)
    |
    df['away_team'].isin(world_cup_2026_teams)
].copy()

print(f'Registros originais: {len(df):,}')
print(f'Registros filtrados: {len(df_filtered):,}')

# Salva resultado
df_filtered.to_csv(
    'football_matches_ml_worldcup2026.csv',
    index=False
)

print(
    'Arquivo salvo: football_matches_ml_worldcup2026.csv'
)