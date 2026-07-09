import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor

# =====================================================
# CARREGAR TREINAMENTO
# =====================================================

df = pd.read_csv(
    "football_matches_ml_worldcup2026.csv"
)

df['date'] = pd.to_datetime(
    df['date']
)

df = df.sort_values(
    'date'
)

# =====================================================
# FEATURES DE DATA
# =====================================================

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# =====================================================
# MAPEAMENTO DE TIMES
# =====================================================

all_teams = sorted(
    set(df['home_team'].unique())
    |
    set(df['away_team'].unique())
)

team_map = {
    team: idx
    for idx, team in enumerate(all_teams)
}

df['home_team_id'] = (
    df['home_team']
    .map(team_map)
)

df['away_team_id'] = (
    df['away_team']
    .map(team_map)
)

# =====================================================
# FEATURES
# =====================================================

X = df[
    [
        'year',
        'month',
        'day',
        'home_team_id',
        'away_team_id'
    ]
]

# =====================================================
# TARGETS
# =====================================================

y = df[
    [
        'home_goals',
        'away_goals'
    ]
]

# =====================================================
# SPLIT TEMPORAL
# =====================================================

split_index = int(
    len(df) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =====================================================
# MODELO
# =====================================================

base_model = XGBRegressor(
    objective='reg:squarederror',

    n_estimators=500,
    max_depth=8,

    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1
)

model = MultiOutputRegressor(
    base_model
)

# =====================================================
# TREINAMENTO
# =====================================================

print("\nTreinando modelo...")

model.fit(
    X_train,
    y_train
)

# =====================================================
# AVALIAÇÃO
# =====================================================

pred = model.predict(
    X_test
)

pred_home = pred[:, 0]
pred_away = pred[:, 1]

mae_home = mean_absolute_error(
    y_test['home_goals'],
    pred_home
)

mae_away = mean_absolute_error(
    y_test['away_goals'],
    pred_away
)

rmse_home = np.sqrt(
    mean_squared_error(
        y_test['home_goals'],
        pred_home
    )
)

rmse_away = np.sqrt(
    mean_squared_error(
        y_test['away_goals'],
        pred_away
    )
)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(
    f"MAE Home Goals: {mae_home:.3f}"
)

print(
    f"MAE Away Goals: {mae_away:.3f}"
)

print(
    f"RMSE Home Goals: {rmse_home:.3f}"
)

print(
    f"RMSE Away Goals: {rmse_away:.3f}"
)

# =====================================================
# PREVISÃO COPA 2026
# =====================================================

print(
    "\nCarregando jogos da Copa..."
)

future = pd.read_csv(
    "worldcup_2026_group_stage.csv"
)

future['date'] = pd.to_datetime(
    future['date']
)

future['year'] = future['date'].dt.year
future['month'] = future['date'].dt.month
future['day'] = future['date'].dt.day

# =====================================================
# TRATAR TIMES NOVOS
# =====================================================

next_id = max(team_map.values()) + 1

for team in pd.concat([
    future['home_team'],
    future['away_team']
]).unique():

    if team not in team_map:

        team_map[team] = next_id
        next_id += 1

future['home_team_id'] = (
    future['home_team']
    .map(team_map)
)

future['away_team_id'] = (
    future['away_team']
    .map(team_map)
)

# =====================================================
# FEATURES FUTURAS
# =====================================================

X_future = future[
    [
        'year',
        'month',
        'day',
        'home_team_id',
        'away_team_id'
    ]
]

# =====================================================
# PREVER
# =====================================================

predictions = model.predict(
    X_future
)

future['pred_home_goals'] = (
    predictions[:, 0]
    .round()
    .astype(int)
)

future['pred_away_goals'] = (
    predictions[:, 1]
    .round()
    .astype(int)
)

# =====================================================
# RESULTADO
# =====================================================

def get_result(row):

    if row['pred_home_goals'] > row['pred_away_goals']:
        return 'HOME_WIN'

    if row['pred_home_goals'] < row['pred_away_goals']:
        return 'AWAY_WIN'

    return 'DRAW'

future['predicted_result'] = (
    future.apply(
        get_result,
        axis=1
    )
)

# =====================================================
# SALVAR
# =====================================================

output_cols = [
    'date',
    'competition',
    'home_team',
    'away_team',
    'pred_home_goals',
    'pred_away_goals',
    'predicted_result'
]

future[
    output_cols
].to_csv(
    'worldcup_2026_predictions.csv',
    index=False
)

print(
    "\nArquivo salvo:"
)

print(
    "worldcup_2026_predictions.csv"
)

print(
    future[
        output_cols
    ].head(20)
)