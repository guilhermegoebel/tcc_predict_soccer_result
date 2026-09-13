import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================
# CONFIGURAÇÃO
# =============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR.parent / "football_matches_ml.csv"

TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023

TARGET_COLUMN = "match_result"
CLASS_LABELS = [0, 1, 2]

# Mantém as features derivadas, assim como o treinamento de RL e o XGBoost.
DROP_COLUMNS = [
    "match_id",
    "date",
    "home_goals",
    "away_goals",
    TARGET_COLUMN,
]

EXCLUDED_CATEGORICAL_COLUMNS = [
    "home_team",
    "away_team",
    "competition",
    "competition_bucket",
]

EXCLUDED_LEGACY_COLUMNS = [
    "home_team_freq",
    "away_team_freq",
    "home_team_seen",
    "away_team_seen",
]

N_SPLITS = 5
N_ITER = 40
SCORING = "f1_macro"
RANDOM_STATE = 42
N_JOBS = -1
MAX_ITER = 5000

OUTPUT_JSON = SCRIPT_DIR / "rl_best_hyperparams.json"
OUTPUT_CV_RESULTS = SCRIPT_DIR / "rl_cv_results_hyperparam_search.csv"


def validate_input_data(df: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "year",
        TARGET_COLUMN,
        "home_team",
        "away_team",
        "competition",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    if df["date"].isna().any():
        raise ValueError("Existem datas inválidas ou ausentes na base.")
    if df["year"].isna().any():
        raise ValueError("Existem anos ausentes na base.")
    unexpected_targets = sorted(
        set(df[TARGET_COLUMN].dropna().unique()).difference(CLASS_LABELS)
    )
    if unexpected_targets:
        raise ValueError(
            f"Valores inesperados em {TARGET_COLUMN}: {unexpected_targets}"
        )


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Deve permanecer idêntica à função do script de treinamento."""
    excluded = set(
        DROP_COLUMNS
        + EXCLUDED_CATEGORICAL_COLUMNS
        + EXCLUDED_LEGACY_COLUMNS
    )
    feature_columns = [column for column in df.columns if column not in excluded]
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(
            "Features não numéricas encontradas. Inclua-as explicitamente no "
            f"pré-processamento ou na lista de exclusão: {non_numeric}"
        )
    if not feature_columns:
        raise ValueError("Nenhuma feature foi selecionada.")
    return feature_columns


def temporal_splits_by_date(dates: pd.Series, n_splits: int):
    """
    Cria folds temporais sobre datas únicas, evitando dividir partidas do
    mesmo dia entre treino e validação.
    """
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) <= n_splits:
        raise ValueError(
            f"Há somente {len(unique_dates)} datas únicas para {n_splits} folds."
        )

    date_splitter = TimeSeriesSplit(n_splits=n_splits)
    date_positions = np.arange(len(unique_dates))

    for train_date_positions, validation_date_positions in date_splitter.split(
        date_positions
    ):
        train_dates = unique_dates[train_date_positions]
        validation_dates = unique_dates[validation_date_positions]
        train_indices = np.flatnonzero(dates.isin(train_dates).to_numpy())
        validation_indices = np.flatnonzero(
            dates.isin(validation_dates).to_numpy()
        )
        yield train_indices, validation_indices


def make_json_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    validate_input_data(df)
    df = df.sort_values("date").reset_index(drop=True)

    # Mesmo pool utilizado pelo tuner do XGBoost: treino + validação.
    # O conjunto de teste permanece completamente isolado.
    search_df = (
        df.loc[df["year"] <= VAL_END_YEAR]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if search_df.empty:
        raise ValueError("O pool de busca ficou vazio.")

    feature_columns = select_feature_columns(search_df)
    X_search = search_df[feature_columns]
    y_search = search_df[TARGET_COLUMN]

    print(
        f"Pool de busca (até {VAL_END_YEAR}): "
        f"{len(search_df):,} partidas"
    )
    print(f"Features usadas ({len(feature_columns)}):")
    print(feature_columns)

    cv_folds = list(
        temporal_splits_by_date(search_df["date"], n_splits=N_SPLITS)
    )
    print("\nVerificação dos folds temporais:")
    for fold_number, (train_indices, validation_indices) in enumerate(cv_folds, 1):
        train_dates = search_df.loc[train_indices, "date"]
        validation_dates = search_df.loc[validation_indices, "date"]
        if train_dates.max() >= validation_dates.min():
            raise AssertionError(
                f"Vazamento temporal detectado no fold {fold_number}."
            )
        print(
            f"  Fold {fold_number}: treino até {train_dates.max().date()} "
            f"({len(train_indices):,}) -> valida "
            f"{validation_dates.min().date()} a "
            f"{validation_dates.max().date()} "
            f"({len(validation_indices):,})"
        )

    # Imputer e scaler ficam dentro do Pipeline para serem ajustados novamente
    # em cada fold, usando somente o trecho de treino daquele fold.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # A lista separa combinações compatíveis de solver e penalização.
    param_distributions = [
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": loguniform(1e-3, 1e2),
        },
        {
            "model__solver": ["saga"],
            "model__penalty": ["l1", "l2"],
            "model__C": loguniform(1e-3, 1e2),
        },
        {
            "model__solver": ["saga"],
            "model__penalty": ["elasticnet"],
            "model__C": loguniform(1e-3, 1e2),
            "model__l1_ratio": uniform(0.0, 1.0),
        },
    ]

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv_folds,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=2,
        refit=False,
        return_train_score=True,
        error_score="raise",
    )

    print(
        f"\nIniciando busca: {N_ITER} combinações x {N_SPLITS} folds "
        f"= {N_ITER * N_SPLITS} treinamentos."
    )
    search.fit(X_search, y_search)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(OUTPUT_CV_RESULTS, index=False)

    best_hyperparams = {
        name.removeprefix("model__"): make_json_serializable(value)
        for name, value in search.best_params_.items()
        if name.startswith("model__")
    }

    # Evita manter l1_ratio quando a penalização vencedora não é elasticnet.
    if best_hyperparams.get("penalty") != "elasticnet":
        best_hyperparams.pop("l1_ratio", None)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(best_hyperparams, file, indent=2, ensure_ascii=False)

    print(f"\nMelhor F1-macro médio: {search.best_score_:.4f}")
    print("Melhores hiperparâmetros:")
    for name, value in best_hyperparams.items():
        print(f"  {name}: {value}")
    print(f"\nResultados completos: {OUTPUT_CV_RESULTS}")
    print(f"Parâmetros para o treinamento: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()