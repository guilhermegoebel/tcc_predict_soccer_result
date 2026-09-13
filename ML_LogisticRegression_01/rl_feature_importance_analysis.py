import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.pipeline import Pipeline


# =============================================================
# CONFIGURAÇÃO
# =============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR.parent / "football_matches_ml.csv"
MODEL_FILE = SCRIPT_DIR / "rl_match_result_model.pkl"
ARTIFACTS_FILE = SCRIPT_DIR / "rl_preprocessing_artifacts.pkl"

PERMUTATION_CSV_FILE = SCRIPT_DIR / "rl_permutation_importance.csv"
PERMUTATION_PNG_FILE = SCRIPT_DIR / "rl_permutation_importance.png"
SHAP_CSV_FILE = SCRIPT_DIR / "rl_shap_importance.csv"
SHAP_PNG_FILE = SCRIPT_DIR / "rl_shap_importance.png"
SHAP_SUMMARY_HOME_FILE = SCRIPT_DIR / "rl_shap_summary_home_win.png"

TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023
TARGET_COLUMN = "match_result"

CLASS_ORDER = [0, 1, 2]
CLASS_NAMES = ["away_win", "draw", "home_win"]

N_REPEATS = 30
BACKGROUND_SIZE = 1000
TOP_N = 20
RANDOM_STATE = 42
N_JOBS = -1

LEGACY_FEATURES = {
    "home_team_freq",
    "away_team_freq",
    "home_team_seen",
    "away_team_seen",
}


# =============================================================
# FUNÇÕES AUXILIARES
# =============================================================

def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def validate_artifacts(model, artifacts: dict):
    required_keys = {
        "feature_columns",
        "imputer",
        "scaler",
        "model_classes",
    }
    missing_keys = sorted(required_keys.difference(artifacts))
    if missing_keys:
        raise ValueError(
            "O arquivo de artefatos não corresponde ao treinamento atual. "
            f"Chaves ausentes: {missing_keys}"
        )

    feature_columns = list(artifacts["feature_columns"])
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    artifact_classes = np.asarray(artifacts["model_classes"], dtype=int)
    model_classes = np.asarray(model.classes_, dtype=int)

    if not feature_columns:
        raise ValueError("A lista feature_columns está vazia.")

    legacy_found = sorted(LEGACY_FEATURES.intersection(feature_columns))
    dummy_features = sorted(
        feature for feature in feature_columns if feature.startswith("comp_")
    )
    if legacy_found or dummy_features:
        raise ValueError(
            "Foram encontradas features incompatíveis com o pipeline atual. "
            f"Features antigas: {legacy_found}; dummies: {dummy_features}"
        )

    if not np.array_equal(artifact_classes, model_classes):
        raise ValueError(
            "As classes do modelo diferem das classes dos artefatos: "
            f"modelo={model_classes.tolist()}, "
            f"artefatos={artifact_classes.tolist()}"
        )

    if set(model_classes.tolist()) != set(CLASS_ORDER):
        raise ValueError(
            "O modelo deve conter exatamente as classes 0, 1 e 2. "
            f"Classes encontradas: {model_classes.tolist()}"
        )

    objects_to_check = {
        "modelo": model,
        "imputer": imputer,
        "scaler": scaler,
    }
    for object_name, fitted_object in objects_to_check.items():
        number_of_features = getattr(
            fitted_object,
            "n_features_in_",
            len(feature_columns),
        )
        if number_of_features != len(feature_columns):
            raise ValueError(
                f"Quantidade de features incompatível no {object_name}: "
                f"{number_of_features}; esperado={len(feature_columns)}"
            )

    return feature_columns, imputer, scaler, model_classes


def validate_numeric_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    missing_features = [
        feature for feature in feature_columns if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "A base histórica não contém todas as features do modelo: "
            f"{missing_features}"
        )

    non_numeric = [
        feature
        for feature in feature_columns
        if not pd.api.types.is_numeric_dtype(df[feature])
    ]
    if non_numeric:
        raise ValueError(
            "As seguintes features deveriam ser numéricas: "
            f"{non_numeric}"
        )


def normalize_shap_values(
    shap_result,
    number_of_samples: int,
    number_of_features: int,
    number_of_classes: int,
) -> list[np.ndarray]:
    """
    Normaliza os formatos retornados por diferentes versões do SHAP
    para uma lista contendo um array (amostras x features) por classe.
    """
    if hasattr(shap_result, "values"):
        shap_result = shap_result.values

    if isinstance(shap_result, list):
        values_per_class = [np.asarray(values) for values in shap_result]
    else:
        values = np.asarray(shap_result)

        if values.ndim == 2:
            if number_of_classes != 1:
                raise ValueError(
                    "O SHAP retornou um array bidimensional para um modelo "
                    f"com {number_of_classes} classes."
                )
            values_per_class = [values]

        elif values.ndim == 3:
            if values.shape == (
                number_of_samples,
                number_of_features,
                number_of_classes,
            ):
                values_per_class = [
                    values[:, :, class_index]
                    for class_index in range(number_of_classes)
                ]
            elif values.shape == (
                number_of_classes,
                number_of_samples,
                number_of_features,
            ):
                values_per_class = [
                    values[class_index]
                    for class_index in range(number_of_classes)
                ]
            elif values.shape == (
                number_of_samples,
                number_of_classes,
                number_of_features,
            ):
                values_per_class = [
                    values[:, class_index, :]
                    for class_index in range(number_of_classes)
                ]
            else:
                raise ValueError(
                    "Formato tridimensional de SHAP não reconhecido: "
                    f"{values.shape}"
                )
        else:
            raise ValueError(
                f"Formato de SHAP não reconhecido: {values.shape}"
            )

    if len(values_per_class) != number_of_classes:
        raise ValueError(
            "Quantidade de classes retornada pelo SHAP incompatível: "
            f"{len(values_per_class)}; esperado={number_of_classes}"
        )

    for class_index, class_values in enumerate(values_per_class):
        expected_shape = (number_of_samples, number_of_features)
        if class_values.shape != expected_shape:
            raise ValueError(
                f"Formato SHAP incompatível na classe {class_index}: "
                f"{class_values.shape}; esperado={expected_shape}"
            )

    return values_per_class


# =============================================================
# EXECUÇÃO
# =============================================================

def main() -> None:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "A biblioteca shap não está instalada. Execute: "
            "python -m pip install shap"
        ) from exc

    print("Carregando modelo e artefatos da Regressão Logística...")

    model = load_pickle(MODEL_FILE)
    artifacts = load_pickle(ARTIFACTS_FILE)
    if not isinstance(artifacts, dict):
        raise ValueError("O arquivo de artefatos não contém um dicionário.")

    (
        feature_columns,
        imputer,
        scaler,
        model_classes,
    ) = validate_artifacts(model, artifacts)

    print(
        f"Modelo e artefatos carregados. "
        f"{len(feature_columns)} features."
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Base histórica não encontrada: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    required_columns = {"date", "year", TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Existem datas inválidas ou ausentes na base.")

    df = df.sort_values("date").reset_index(drop=True)
    validate_numeric_features(df, feature_columns)

    train_mask = df["year"] <= TRAIN_END_YEAR
    test_mask = df["year"] > VAL_END_YEAR

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise ValueError(
            "O conjunto de treinamento ou de teste ficou vazio."
        )

    X_train = df_train[feature_columns].copy()
    X_test = df_test[feature_columns].copy()
    y_test = df_test[TARGET_COLUMN].astype(int)

    unexpected_targets = sorted(
        set(y_test.unique()).difference(CLASS_ORDER)
    )
    if unexpected_targets:
        raise ValueError(
            f"Classes inesperadas no teste: {unexpected_targets}"
        )

    # Os transformadores são apenas aplicados. Não há fit nesta análise.
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train_processed_array = scaler.transform(X_train_imputed)
    X_test_processed_array = scaler.transform(X_test_imputed)

    X_train_processed = pd.DataFrame(
        X_train_processed_array,
        columns=feature_columns,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=feature_columns,
        index=X_test.index,
    )

    print(
        f"Conjunto de teste reconstruído: {len(X_test):,} partidas "
        f"(a partir de {VAL_END_YEAR + 1})."
    )

    # Verificação de que o conjunto reconstruído reproduz a avaliação.
    y_test_pred = model.predict(X_test_processed_array)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(
        y_test,
        y_test_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )
    print(
        "Verificação do modelo no teste — "
        f"Accuracy: {test_accuracy:.4f}; "
        f"F1-macro: {test_f1_macro:.4f}"
    )

    # =========================================================
    # PERMUTATION IMPORTANCE
    # =========================================================

    # O Pipeline usa os objetos já ajustados e permite embaralhar as
    # features na escala original antes da imputação e padronização.
    fitted_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
            ("model", model),
        ]
    )

    f1_macro_scorer = make_scorer(
        f1_score,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    print(
        f"\nCalculando permutation importance "
        f"({N_REPEATS} repetições por feature)..."
    )

    permutation_result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        scoring=f1_macro_scorer,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": permutation_result.importances_mean,
            "importance_std": permutation_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    permutation_df.to_csv(PERMUTATION_CSV_FILE, index=False)

    top_permutation = (
        permutation_df.head(TOP_N)
        .sort_values("importance_mean")
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(
        top_permutation["feature"],
        top_permutation["importance_mean"],
        xerr=top_permutation["importance_std"],
        color="#2e7d32",
        ecolor="#a5a5a5",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Top {TOP_N} features — Permutation Importance — RL"
    )
    ax.set_xlabel(
        "Variação média do F1-macro após o embaralhamento"
    )
    fig.tight_layout()
    fig.savefig(
        PERMUTATION_PNG_FILE,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Resultados salvos em: {PERMUTATION_CSV_FILE}")
    print(f"Gráfico salvo em: {PERMUTATION_PNG_FILE}")
    print("\nTop 15 — permutation importance:")
    print(
        permutation_df[
            ["feature", "importance_mean", "importance_std"]
        ]
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    # =========================================================
    # SHAP — LINEAR EXPLAINER
    # =========================================================

    background_size = min(BACKGROUND_SIZE, len(X_train_processed))
    random_generator = np.random.default_rng(RANDOM_STATE)
    background_positions = random_generator.choice(
        len(X_train_processed),
        size=background_size,
        replace=False,
    )
    X_background = X_train_processed.iloc[background_positions].copy()

    print(
        f"\nCalculando valores SHAP com LinearExplainer "
        f"(background={background_size} partidas)..."
    )

    masker = shap.maskers.Independent(X_background)
    explainer = shap.LinearExplainer(model, masker)

    try:
        shap_result = explainer(X_test_processed)
    except TypeError:
        # Compatibilidade com versões antigas da biblioteca.
        shap_result = explainer.shap_values(X_test_processed)

    shap_values_per_class = normalize_shap_values(
        shap_result=shap_result,
        number_of_samples=len(X_test_processed),
        number_of_features=len(feature_columns),
        number_of_classes=len(model_classes),
    )

    mean_abs_shap_per_class = np.stack(
        [
            np.abs(class_values).mean(axis=0)
            for class_values in shap_values_per_class
        ]
    )
    global_shap_importance = mean_abs_shap_per_class.mean(axis=0)

    shap_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap_global": global_shap_importance,
        }
    )

    class_to_shap_index = {
        int(class_label): class_index
        for class_index, class_label in enumerate(model_classes)
    }
    for class_label, class_name in zip(CLASS_ORDER, CLASS_NAMES):
        shap_class_index = class_to_shap_index[class_label]
        shap_importance_df[
            f"mean_abs_shap_{class_name}"
        ] = mean_abs_shap_per_class[shap_class_index]

    shap_importance_df = shap_importance_df.sort_values(
        "mean_abs_shap_global",
        ascending=False,
    )
    shap_importance_df.to_csv(SHAP_CSV_FILE, index=False)

    top_shap = (
        shap_importance_df.head(TOP_N)
        .sort_values("mean_abs_shap_global")
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(
        top_shap["feature"],
        top_shap["mean_abs_shap_global"],
        color="#1565c0",
    )
    ax.set_title(
        f"Top {TOP_N} features — SHAP — Regressão Logística"
    )
    ax.set_xlabel("Média do valor absoluto de SHAP entre as classes")
    fig.tight_layout()
    fig.savefig(
        SHAP_PNG_FILE,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Resultados SHAP salvos em: {SHAP_CSV_FILE}")
    print(f"Gráfico SHAP salvo em: {SHAP_PNG_FILE}")
    print("\nTop 15 — SHAP global:")
    print(
        shap_importance_df[
            ["feature", "mean_abs_shap_global"]
        ]
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    # Mesmo gráfico direcional gerado para o XGBoost: classe home_win.
    home_win_model_index = class_to_shap_index[2]
    home_win_shap_values = shap_values_per_class[
        home_win_model_index
    ]

    # Os valores SHAP correspondem às entradas padronizadas do modelo.
    # X_test é usado apenas para colorir o gráfico na escala original.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        home_win_shap_values,
        X_test,
        feature_names=feature_columns,
        show=False,
        max_display=TOP_N,
    )
    plt.title(
        'SHAP summary — Regressão Logística — classe "home_win"'
    )
    plt.tight_layout()
    plt.savefig(
        SHAP_SUMMARY_HOME_FILE,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(
        "Resumo direcional da classe home_win salvo em: "
        f"{SHAP_SUMMARY_HOME_FILE}"
    )
    print(
        "\nAnálise concluída. Os valores SHAP explicam os escores "
        "lineares do modelo antes da calibração isotônica."
    )


if __name__ == "__main__":
    main()