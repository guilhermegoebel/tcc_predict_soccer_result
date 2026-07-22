"""
Análise de cobertura de dados por ano/década.

Objetivo: visualizar a % de partidas com found_count == 0 (nenhum
jogador com valor de mercado encontrado) e a % de rankings ausentes
(home_rank/away_rank nulos), agrupado por ano e por década, para
decidir se faz sentido cortar a base a partir de um ano em que a
cobertura estabiliza.

Uso:
    python analise_cobertura.py

Espera encontrar 'football_matches_ml.csv' (gerado pelo script.py)
no mesmo diretório. Ajuste INPUT_FILE abaixo se necessário.
"""

import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = 'football_matches_ml.csv'

df = pd.read_csv(INPUT_FILE)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['decade'] = (df['year'] // 10) * 10

# =========================================================
# MÉTRICAS DE COBERTURA POR LINHA
# =========================================================

# found_count == 0 em home OU away conta como "partida com problema
# de cobertura de mercado" nesse lado específico. Analisamos home e
# away separadamente, pois a cobertura pode variar por força do time
# mandante/visitante (ex.: seleções menores jogando fora).
df['home_market_missing'] = df['home_found_count'] == 0
df['away_market_missing'] = df['away_found_count'] == 0

df['home_ranking_missing'] = df['home_rank'].isna()
df['away_ranking_missing'] = df['away_rank'].isna()

# =========================================================
# AGREGAÇÃO POR ANO
# =========================================================

by_year = df.groupby('year').agg(
    pct_home_market_missing=('home_market_missing', 'mean'),
    pct_away_market_missing=('away_market_missing', 'mean'),
    pct_home_ranking_missing=('home_ranking_missing', 'mean'),
    pct_away_ranking_missing=('away_ranking_missing', 'mean'),
    n_matches=('match_id', 'count')
).reset_index()

for col in [
    'pct_home_market_missing',
    'pct_away_market_missing',
    'pct_home_ranking_missing',
    'pct_away_ranking_missing'
]:
    by_year[col] = by_year[col] * 100

# =========================================================
# AGREGAÇÃO POR DÉCADA
# =========================================================

by_decade = df.groupby('decade').agg(
    pct_home_market_missing=('home_market_missing', 'mean'),
    pct_away_market_missing=('away_market_missing', 'mean'),
    pct_home_ranking_missing=('home_ranking_missing', 'mean'),
    pct_away_ranking_missing=('away_ranking_missing', 'mean'),
    n_matches=('match_id', 'count')
).reset_index()

for col in [
    'pct_home_market_missing',
    'pct_away_market_missing',
    'pct_home_ranking_missing',
    'pct_away_ranking_missing'
]:
    by_decade[col] = by_decade[col] * 100

# =========================================================
# GRÁFICO
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# --- Ano: valor de mercado ausente ---
ax = axes[0, 0]
ax.plot(by_year['year'], by_year['pct_home_market_missing'], label='Home', marker='o', markersize=3)
ax.plot(by_year['year'], by_year['pct_away_market_missing'], label='Away', marker='o', markersize=3)
ax.set_title('% partidas com found_count == 0 (valor de mercado) por ano')
ax.set_xlabel('Ano')
ax.set_ylabel('% de partidas')
ax.legend()
ax.grid(alpha=0.3)

# --- Ano: ranking ausente ---
ax = axes[0, 1]
ax.plot(by_year['year'], by_year['pct_home_ranking_missing'], label='Home', marker='o', markersize=3)
ax.plot(by_year['year'], by_year['pct_away_ranking_missing'], label='Away', marker='o', markersize=3)
ax.set_title('% partidas com ranking ausente por ano')
ax.set_xlabel('Ano')
ax.set_ylabel('% de partidas')
ax.legend()
ax.grid(alpha=0.3)

# --- Década: valor de mercado ausente ---
ax = axes[1, 0]
width = 3
ax.bar(by_decade['decade'] - width/2, by_decade['pct_home_market_missing'], width=width, label='Home')
ax.bar(by_decade['decade'] + width/2, by_decade['pct_away_market_missing'], width=width, label='Away')
ax.set_title('% partidas com found_count == 0 (valor de mercado) por década')
ax.set_xlabel('Década')
ax.set_ylabel('% de partidas')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# --- Década: ranking ausente ---
ax = axes[1, 1]
ax.bar(by_decade['decade'] - width/2, by_decade['pct_home_ranking_missing'], width=width, label='Home')
ax.bar(by_decade['decade'] + width/2, by_decade['pct_away_ranking_missing'], width=width, label='Away')
ax.set_title('% partidas com ranking ausente por década')
ax.set_xlabel('Década')
ax.set_ylabel('% de partidas')
ax.legend()
ax.grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig('cobertura_por_ano_decada.png', dpi=150)

print('Gráfico salvo em cobertura_por_ano_decada.png')
print('\nResumo por década:')
print(
    by_decade[[
        'decade',
        'n_matches',
        'pct_home_market_missing',
        'pct_away_market_missing',
        'pct_home_ranking_missing',
        'pct_away_ranking_missing'
    ]].round(1).to_string(index=False)
)