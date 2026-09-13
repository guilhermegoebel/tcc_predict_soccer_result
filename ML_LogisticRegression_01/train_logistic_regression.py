import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
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
CLASS_NAMES = ["away_win", "draw", "home_win"]

# Mantém rank_diff, points_diff e market_value_avg_diff, em conformidade
# com o conjunto de informações usado pelo XGBoost.
DROP_COLUMNS = [
    "match_id",
    "date",
    "home_goals",
    "away_goals",
    TARGET_COLUMN,
]

# As variáveis categóricas abaixo não são usadas pelo XGBoost de referência.
# Por isso, também não são codificadas nem utilizadas nesta RL.
EXCLUDED_CATEGORICAL_COLUMNS = [
    "home_team",
    "away_team",
    "competition",
    "competition_bucket",
]

# Compatibilidade defensiva caso a base já contenha colunas geradas por uma
# versão anterior do pipeline. Elas não fazem parte deste experimento.
EXCLUDED_LEGACY_COLUMNS = [
    "home_team_freq",
    "away_team_freq",
    "home_team_seen",
    "away_team_seen",
]

DEFAULT_HYPERPARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "penalty": "l2",
}

HYPERPARAMS_FILE = SCRIPT_DIR / "rl_best_hyperparams.json"
RANDOM_STATE = 42
MAX_ITER = 5000


def output_path(filename: str) -> Path:
    """Retorna um caminho de saída ao lado deste script."""
    return SCRIPT_DIR / filename


def load_hyperparameters() -> dict:
    """Carrega os parâmetros do tuner ou usa os valores padrão."""
    if not HYPERPARAMS_FILE.exists():
        warnings.warn(
            f"{HYPERPARAMS_FILE.name} não encontrado; usando parâmetros "
            "padrão. Execute tune_logistic_regression_hyperparams.py "
            "antes deste script para usar parâmetros otimizados."
        )
        return DEFAULT_HYPERPARAMS.copy()

    with HYPERPARAMS_FILE.open("r", encoding="utf-8") as file:
        tuned_hyperparams = json.load(file)

    model_hyperparams = {**DEFAULT_HYPERPARAMS, **tuned_hyperparams}
    print(f"Hiperparâmetros carregados de {HYPERPARAMS_FILE.name}:")
    for name, value in model_hyperparams.items():
        print(f"  {name}: {value}")
    return model_hyperparams


def validate_input_data(df: pd.DataFrame) -> None:
    """Valida as colunas mínimas e os campos usados no corte temporal."""
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
    """Seleciona apenas as features numéricas utilizadas pelos modelos."""
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


def calculate_classification_metrics(y_true, y_pred) -> dict:
    """Calcula as métricas de classificação comuns aos modelos."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
    }


def fit_isotonic_calibrators(y_true, raw_probabilities, model_classes) -> dict:
    """Ajusta um calibrador isotônico para cada classe do modelo."""
    calibrators = {}
    for probability_index, class_label in enumerate(model_classes):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            raw_probabilities[:, probability_index],
            (np.asarray(y_true) == class_label).astype(int),
        )
        calibrators[int(class_label)] = calibrator
    return calibrators


def calibrate_probabilities(
    raw_probabilities: np.ndarray,
    calibrators: dict,
    model_classes,
) -> np.ndarray:
    """Aplica a calibração por classe e renormaliza as probabilidades."""
    calibrated = np.zeros_like(raw_probabilities, dtype=float)
    for probability_index, class_label in enumerate(model_classes):
        calibrated[:, probability_index] = calibrators[int(class_label)].predict(
            raw_probabilities[:, probability_index]
        )

    row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    calibrated[zero_rows] = raw_probabilities[zero_rows]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / row_sums


def build_evaluation_entry(
    split_name: str,
    y_true,
    y_pred,
    raw_probabilities,
    calibrated_probabilities,
) -> dict:
    metrics = calculate_classification_metrics(y_true, y_pred)
    return {
        "split": split_name,
        "num_samples": int(len(y_true)),
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "log_loss_raw": float(
            log_loss(y_true, raw_probabilities, labels=CLASS_LABELS)
        ),
        "log_loss_calibrated": float(
            log_loss(y_true, calibrated_probabilities, labels=CLASS_LABELS)
        ),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def export_summary_table(metrics_df: pd.DataFrame) -> Path:
    path = output_path("rl_resumo_executivo.png")
    table_df = metrics_df[
        [
            "split",
            "accuracy",
            "f1_macro",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
        ]
    ].copy()
    table_df.columns = [
        "Conjunto",
        "Accuracy",
        "F1-macro",
        "Balanced accuracy",
        "Precision macro",
        "Recall macro",
    ]

    fig, ax = plt.subplots(figsize=(11, max(2.7, 1.2 + len(table_df) * 0.55)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.round(4).astype(str).to_numpy(),
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#dfeaf7"] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor("#b9c7d6")
        if row == 0:
            cell.set_facecolor("#bfcfe6")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def export_class_performance_table(class_report: dict) -> Path:
    path = output_path("rl_desempenho_por_classe.png")
    table_df = pd.DataFrame(class_report).T.loc[
        CLASS_NAMES, ["precision", "recall", "f1-score"]
    ].copy().round(2)
    table_df.index = [
        "Vitória Visitante (0)",
        "Empate (1)",
        "Vitória Mandante (2)",
    ]
    table_df.columns = ["Precisão", "Recall", "F1-score"]

    fig, ax = plt.subplots(figsize=(8, max(2.8, 1.2 + len(table_df) * 0.7)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.to_numpy(),
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        loc="center",
        cellLoc="center",
        colColours=["#dfeaf7"] * len(table_df.columns),
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor("#b9c7d6")
        if row == 0:
            cell.set_facecolor("#bfcfe6")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    model_hyperparams = load_hyperparameters()

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    validate_input_data(df)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Total de partidas: {len(df):,}")
    print("Distribuição do target (%):")
    print(df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100)

    train_mask = df["year"] <= TRAIN_END_YEAR
    validation_mask = (df["year"] > TRAIN_END_YEAR) & (
        df["year"] <= VAL_END_YEAR
    )
    test_mask = df["year"] > VAL_END_YEAR

    df_train = df.loc[train_mask].copy()
    df_validation = df.loc[validation_mask].copy()
    df_test = df.loc[test_mask].copy()

    if df_train.empty or df_validation.empty or df_test.empty:
        raise ValueError(
            "Treino, validação ou teste ficaram vazios. Ajuste "
            "TRAIN_END_YEAR e VAL_END_YEAR ao intervalo da base."
        )

    print(f"\nTreino: {len(df_train):,} partidas (até {TRAIN_END_YEAR})")
    print(
        f"Validação: {len(df_validation):,} partidas "
        f"({TRAIN_END_YEAR + 1}-{VAL_END_YEAR})"
    )
    print(
        f"Teste: {len(df_test):,} partidas "
        f"(a partir de {VAL_END_YEAR + 1})"
    )

    feature_columns = select_feature_columns(df_train)
    print(f"\nFeatures usadas ({len(feature_columns)}):")
    print(feature_columns)

    X_train = df_train[feature_columns]
    y_train = df_train[TARGET_COLUMN]
    X_validation = df_validation[feature_columns]
    y_validation = df_validation[TARGET_COLUMN]
    X_test = df_test[feature_columns]
    y_test = df_test[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train_imputed)
    X_validation_processed = scaler.transform(X_validation_imputed)
    X_test_processed = scaler.transform(X_test_imputed)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        **model_hyperparams,
    )
    model.fit(X_train_processed, y_train)

    y_validation_pred = model.predict(X_validation_processed)
    y_validation_proba_raw = model.predict_proba(X_validation_processed)
    y_test_pred = model.predict(X_test_processed)
    y_test_proba_raw = model.predict_proba(X_test_processed)

    calibrators = fit_isotonic_calibrators(
        y_validation, y_validation_proba_raw, model.classes_
    )
    y_validation_proba = calibrate_probabilities(
        y_validation_proba_raw, calibrators, model.classes_
    )
    y_test_proba = calibrate_probabilities(
        y_test_proba_raw, calibrators, model.classes_
    )

    validation_results = build_evaluation_entry(
        "validation",
        y_validation,
        y_validation_pred,
        y_validation_proba_raw,
        y_validation_proba,
    )
    test_results = build_evaluation_entry(
        "test", y_test, y_test_pred, y_test_proba_raw, y_test_proba
    )

    baseline_pred = np.full(len(y_test), 2, dtype=int)
    baseline_metrics = calculate_classification_metrics(y_test, baseline_pred)
    baseline_results = {
        "split": "baseline_test",
        "num_samples": int(len(y_test)),
        "accuracy": float(baseline_metrics["accuracy"]),
        "f1_macro": float(baseline_metrics["f1_macro"]),
        "log_loss_raw": None,
        "log_loss_calibrated": None,
        "balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
        "precision_macro": float(baseline_metrics["precision_macro"]),
        "recall_macro": float(baseline_metrics["recall_macro"]),
        "classification_report": None,
    }

    evaluation_metrics = {
        "validation": validation_results,
        "test": test_results,
        "baseline_test": baseline_results,
    }

    for split_name in ("validation", "test"):
        result = evaluation_metrics[split_name]
        print(f"\n{'=' * 35}")
        print(f"RESULTADOS: {split_name.upper()}")
        print("=" * 35)
        print(f"F1-macro: {result['f1_macro']:.4f}")
        print(
            "Log-loss (bruto / calibrado): "
            f"{result['log_loss_raw']:.4f} / "
            f"{result['log_loss_calibrated']:.4f}"
        )
        print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
        print(f"Precision macro: {result['precision_macro']:.4f}")
        print(f"Recall macro: {result['recall_macro']:.4f}")
        print(f"Accuracy: {result['accuracy']:.4f}")

    print("\nClassification report do teste:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(
        '\nBaseline "sempre mandante vence" — '
        f"F1-macro: {baseline_results['f1_macro']:.4f}"
    )

    metrics_json_path = output_path("rl_evaluation_metrics.json")
    with metrics_json_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation_metrics, file, indent=2, ensure_ascii=False)

    metrics_df = pd.DataFrame(
        [
            {
                "split": row["split"],
                "num_samples": row["num_samples"],
                "accuracy": row["accuracy"],
                "f1_macro": row["f1_macro"],
                "log_loss_raw": row["log_loss_raw"],
                "log_loss_calibrated": row["log_loss_calibrated"],
                "balanced_accuracy": row["balanced_accuracy"],
                "precision_macro": row["precision_macro"],
                "recall_macro": row["recall_macro"],
            }
            for row in evaluation_metrics.values()
        ]
    )
    metrics_csv_path = output_path("rl_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    summary_path = export_summary_table(metrics_df)
    class_table_path = export_class_performance_table(
        test_results["classification_report"]
    )

    confusion = confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion, display_labels=CLASS_NAMES
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão — teste — Regressão Logística")
    fig.tight_layout()
    confusion_path = output_path("rl_confusion_matrix.png")
    fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    coefficients_df = pd.DataFrame(
        model.coef_,
        index=[CLASS_NAMES[CLASS_LABELS.index(int(label))] for label in model.classes_],
        columns=feature_columns,
    )
    coefficients_matrix_path = output_path("rl_coefficients_by_class.csv")
    coefficients_df.to_csv(coefficients_matrix_path, encoding="utf-8")

    coefficients_long_df = (
        coefficients_df.reset_index()
        .rename(columns={"index": "class"})
        .melt(id_vars="class", var_name="feature", value_name="coefficient")
    )
    coefficients_detailed_path = output_path("rl_coefficients_detailed.csv")
    coefficients_long_df.to_csv(
        coefficients_detailed_path, index=False, encoding="utf-8"
    )

    importance_values = np.mean(np.abs(model.coef_), axis=0)
    importances = pd.Series(
        importance_values, index=feature_columns, name="mean_absolute_coefficient"
    ).sort_values(ascending=False)
    importance_csv_path = output_path("rl_feature_importance.csv")
    importances.to_csv(importance_csv_path, header=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    importances.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 features — Regressão Logística")
    ax.set_xlabel("Média do valor absoluto do coeficiente padronizado")
    fig.tight_layout()
    importance_path = output_path("rl_feature_importance.png")
    fig.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for class_name in coefficients_df.index:
        class_coefficients = coefficients_df.loc[class_name]
        top_features = class_coefficients.loc[
            class_coefficients.abs().nlargest(20).index
        ].sort_values()
        colors = ["#c44e52" if value < 0 else "#4c72b0" for value in top_features]
        fig, ax = plt.subplots(figsize=(9, 8))
        top_features.plot(kind="barh", ax=ax, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Coeficientes — classe {class_name}")
        ax.set_xlabel("Coeficiente padronizado")
        fig.tight_layout()
        class_plot_path = output_path(f"rl_coefficients_{class_name}.png")
        fig.savefig(class_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    model_path = output_path("rl_match_result_model.pkl")
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    imputer_path = output_path("rl_imputer.pkl")
    with imputer_path.open("wb") as file:
        pickle.dump(imputer, file)

    scaler_path = output_path("rl_scaler.pkl")
    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    # Este arquivo contém somente objetos realmente usados na inferência.
    preprocessing_artifacts = {
        "feature_columns": feature_columns,
        "imputer": imputer,
        "scaler": scaler,
        "calibrators": calibrators,
        "model_classes": model.classes_.tolist(),
        "target_column": TARGET_COLUMN,
        "class_labels": CLASS_LABELS,
        "class_names": CLASS_NAMES,
        "excluded_columns": sorted(
            set(
                DROP_COLUMNS
                + EXCLUDED_CATEGORICAL_COLUMNS
                + EXCLUDED_LEGACY_COLUMNS
            )
        ),
        "train_end_year": TRAIN_END_YEAR,
        "validation_end_year": VAL_END_YEAR,
        "model_hyperparameters": model_hyperparams,
    }
    artifacts_path = output_path("rl_preprocessing_artifacts.pkl")
    with artifacts_path.open("wb") as file:
        pickle.dump(preprocessing_artifacts, file)

    print("\nArquivos principais gerados:")
    for path in (
        metrics_csv_path,
        metrics_json_path,
        summary_path,
        class_table_path,
        confusion_path,
        importance_path,
        importance_csv_path,
        coefficients_matrix_path,
        coefficients_detailed_path,
        model_path,
        imputer_path,
        scaler_path,
        artifacts_path,
    ):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()