import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# =============================================================
# CONFIGURAÇÃO
# =============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR.parent / "football_matches_2026.csv"
MODEL_FILE = SCRIPT_DIR / "rl_match_result_model.pkl"
ARTIFACTS_FILE = SCRIPT_DIR / "rl_preprocessing_artifacts.pkl"

OUTPUT_FILE = SCRIPT_DIR / "rl_comparativo_real_vs_predito_2026.csv"
METRICS_FILE = SCRIPT_DIR / "rl_prediction_metrics_2026.json"
METRICS_CSV_FILE = SCRIPT_DIR / "rl_prediction_metrics_2026.csv"
CLASS_METRICS_FILE = SCRIPT_DIR / "rl_class_metrics_2026.csv"
CONFUSION_MATRIX_FILE = SCRIPT_DIR / "rl_confusion_matrix_2026.png"
CLASS_PERFORMANCE_FILE = SCRIPT_DIR / "rl_desempenho_por_classe_2026.png"

TARGET_COLUMN = "match_result"
CLASS_ORDER = [0, 1, 2]

RESULT_LABELS = {
    0: "away_win",
    1: "draw",
    2: "home_win",
}

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
    """Valida o contrato salvo pelo novo treinamento da RL."""
    required_keys = {
        "feature_columns",
        "imputer",
        "scaler",
        "calibrators",
        "model_classes",
    }
    missing_keys = sorted(required_keys.difference(artifacts))
    if missing_keys:
        raise ValueError(
            "O arquivo de artefatos não corresponde ao novo treinamento da RL. "
            f"Chaves ausentes: {missing_keys}"
        )

    feature_columns = list(artifacts["feature_columns"])
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    calibrators = artifacts["calibrators"]
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
            "Foram encontradas features que não pertencem ao novo pipeline. "
            f"Features antigas: {legacy_found}; dummies: {dummy_features}"
        )

    if not np.array_equal(artifact_classes, model_classes):
        raise ValueError(
            "As classes do modelo diferem das registradas nos artefatos: "
            f"modelo={model_classes.tolist()}, "
            f"artefatos={artifact_classes.tolist()}"
        )

    if set(model_classes.tolist()) != set(CLASS_ORDER):
        raise ValueError(
            "O modelo não contém exatamente as classes 0, 1 e 2: "
            f"{model_classes.tolist()}"
        )

    objects_to_check = {
        "modelo": model,
        "imputer": imputer,
        "scaler": scaler,
    }
    for object_name, fitted_object in objects_to_check.items():
        number_of_features = getattr(
            fitted_object, "n_features_in_", len(feature_columns)
        )
        if number_of_features != len(feature_columns):
            raise ValueError(
                f"Quantidade de features incompatível no {object_name}: "
                f"{number_of_features}; esperado={len(feature_columns)}"
            )

    return feature_columns, imputer, scaler, calibrators, model_classes


def validate_prediction_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    non_numeric = [
        feature
        for feature in feature_columns
        if feature in df.columns
        and not pd.api.types.is_numeric_dtype(df[feature])
    ]
    if non_numeric:
        raise ValueError(
            "As seguintes features deveriam ser numéricas: "
            f"{non_numeric}"
        )


def calibrate_probabilities(
    raw_probabilities: np.ndarray,
    calibrators: dict,
    model_classes: np.ndarray,
) -> np.ndarray:
    """Reproduz a calibração aplicada no treinamento."""
    calibrated = np.zeros_like(raw_probabilities, dtype=float)

    for probability_index, class_label in enumerate(model_classes):
        class_key = int(class_label)
        if class_key not in calibrators:
            raise ValueError(f"Calibrador ausente para a classe {class_key}.")
        calibrated[:, probability_index] = calibrators[class_key].predict(
            raw_probabilities[:, probability_index]
        )

    row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0

    # Evita uma linha com três probabilidades iguais a zero.
    calibrated[zero_rows] = raw_probabilities[zero_rows]
    row_sums = calibrated.sum(axis=1, keepdims=True)

    return calibrated / row_sums


def reorder_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    """Ordena as probabilidades como 0=away, 1=draw e 2=home."""
    class_to_index = {
        int(class_label): probability_index
        for probability_index, class_label in enumerate(model_classes)
    }
    return np.column_stack(
        [
            probabilities[:, class_to_index[class_label]]
            for class_label in CLASS_ORDER
        ]
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
    }


def export_class_performance_chart(class_metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(class_metrics_df))
    width = 0.25

    ax.bar(
        positions - width,
        class_metrics_df["precision"],
        width,
        label="Precisão",
    )
    ax.bar(
        positions,
        class_metrics_df["recall"],
        width,
        label="Recall",
    )
    ax.bar(
        positions + width,
        class_metrics_df["f1_score"],
        width,
        label="F1-score",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Vitória\nVisitante", "Empate", "Vitória\nMandante"]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Desempenho por Classe — Regressão Logística — 2026")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CLASS_PERFORMANCE_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_empty_metrics(num_matches: int, num_evaluated: int) -> dict:
    """Cria o contrato do JSON mesmo quando não há resultado real."""
    return {
        "model": "Logistic Regression",
        "year": 2026,
        "num_matches": num_matches,
        "num_matches_with_real_result": num_evaluated,
        "has_real_result": num_evaluated > 0,
        "accuracy": None,
        "f1_macro": None,
        "log_loss_raw": None,
        "log_loss_calibrated": None,
        "balanced_accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_per_class": None,
        "precision_per_class": None,
        "recall_per_class": None,
        "support_per_class": None,
        "confusion_matrix": None,
    }


# =============================================================
# EXECUÇÃO
# =============================================================

def main() -> None:
    print("Carregando modelo e artefatos da Regressão Logística...")

    model = load_pickle(MODEL_FILE)
    artifacts = load_pickle(ARTIFACTS_FILE)
    if not isinstance(artifacts, dict):
        raise ValueError("O arquivo de artefatos não contém um dicionário.")

    (
        feature_columns,
        imputer,
        scaler,
        calibrators,
        model_classes,
    ) = validate_artifacts(model, artifacts)

    print("Modelo e artefatos carregados com sucesso.")
    print(f"Número de features utilizadas: {len(feature_columns)}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Base de 2026 não encontrada: {INPUT_FILE}")

    print("\nCarregando dados de 2026...")
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        raise ValueError("A base de 2026 está vazia.")

    identity_columns = ["home_team", "away_team"]
    missing_identity_columns = [
        column for column in identity_columns if column not in df.columns
    ]
    if missing_identity_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{missing_identity_columns}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = int(df["date"].isna().sum())
        if invalid_dates:
            print(
                f"[AVISO] {invalid_dates} partidas possuem data inválida."
            )

    print(f"Total de partidas na base de 2026: {len(df):,}")

    validate_prediction_features(df, feature_columns)

    missing_features = [
        feature for feature in feature_columns if feature not in df.columns
    ]
    if missing_features:
        print(
            "\n[AVISO] Features ausentes. Elas serão preenchidas com NaN "
            "e tratadas pelo imputer treinado:"
        )
        for feature in missing_features:
            df[feature] = np.nan
            print(f"  - {feature}: 100.0% ausente")

    partially_missing = [
        feature
        for feature in feature_columns
        if feature not in missing_features and df[feature].isna().any()
    ]
    if partially_missing:
        print("\n[AVISO] Features parcialmente ausentes:")
        for feature in partially_missing:
            missing_percentage = df[feature].isna().mean() * 100
            print(f"  - {feature}: {missing_percentage:.1f}% ausente")

    # A ordem vem exclusivamente do artefato salvo no treinamento.
    X = df[feature_columns].copy()
    X_imputed = imputer.transform(X)
    X_processed = scaler.transform(X_imputed)

    print(
        f"\nMatriz de predição: {X_processed.shape[0]} partidas x "
        f"{X_processed.shape[1]} features"
    )
    print("Realizando predições...")

    y_pred = model.predict(X_processed).astype(int)
    raw_probabilities_model_order = model.predict_proba(X_processed)

    if calibrators:
        calibrated_probabilities_model_order = calibrate_probabilities(
            raw_probabilities_model_order,
            calibrators,
            model_classes,
        )
        print("Probabilidades calibradas com regressão isotônica.")
    else:
        calibrated_probabilities_model_order = (
            raw_probabilities_model_order.copy()
        )
        print(
            "[AVISO] Calibradores ausentes. As probabilidades calibradas "
            "serão iguais às probabilidades brutas."
        )

    raw_probabilities = reorder_probabilities(
        raw_probabilities_model_order,
        model_classes,
    )
    calibrated_probabilities = reorder_probabilities(
        calibrated_probabilities_model_order,
        model_classes,
    )

    df["resultado_predito"] = pd.Series(
        y_pred,
        index=df.index,
    ).map(RESULT_LABELS)

    # prob_* mantém o contrato do XGBoost e contém probabilidades calibradas.
    # prob_raw_* permite auditar o efeito da calibração.
    df["prob_away_win"] = calibrated_probabilities[:, 0]
    df["prob_draw"] = calibrated_probabilities[:, 1]
    df["prob_home_win"] = calibrated_probabilities[:, 2]
    df["prob_raw_away_win"] = raw_probabilities[:, 0]
    df["prob_raw_draw"] = raw_probabilities[:, 1]
    df["prob_raw_home_win"] = raw_probabilities[:, 2]

    if TARGET_COLUMN in df.columns:
        target_numeric = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce",
        )
        invalid_non_null = df[TARGET_COLUMN].notna() & target_numeric.isna()
        if invalid_non_null.any():
            invalid_values = sorted(
                df.loc[invalid_non_null, TARGET_COLUMN]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Valores não numéricos em {TARGET_COLUMN}: "
                f"{invalid_values}"
            )

        invalid_classes = (
            target_numeric.notna()
            & ~target_numeric.isin(CLASS_ORDER)
        )
        if invalid_classes.any():
            invalid_values = sorted(
                target_numeric.loc[invalid_classes].unique().tolist()
            )
            raise ValueError(
                f"Classes inesperadas em {TARGET_COLUMN}: "
                f"{invalid_values}"
            )
    else:
        target_numeric = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    evaluation_mask = target_numeric.notna()
    num_evaluated = int(evaluation_mask.sum())
    has_real_result = num_evaluated > 0

    df["resultado_real"] = (
        target_numeric.map(RESULT_LABELS).fillna("desconhecido")
    )
    df["acertou"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="boolean",
    )
    if has_real_result:
        evaluated_predictions = y_pred[evaluation_mask.to_numpy()]
        evaluated_targets = (
            target_numeric.loc[evaluation_mask].astype(int).to_numpy()
        )
        df.loc[evaluation_mask, "acertou"] = (
            evaluated_targets == evaluated_predictions
        )

    output_columns = []
    for optional_column in ("match_id", "date"):
        if optional_column in df.columns:
            output_columns.append(optional_column)

    output_columns.extend(["home_team", "away_team"])
    if "competition" in df.columns:
        output_columns.append("competition")

    output_columns.extend(
        [
            "resultado_real",
            "resultado_predito",
            "acertou",
            "prob_away_win",
            "prob_draw",
            "prob_home_win",
            "prob_raw_away_win",
            "prob_raw_draw",
            "prob_raw_home_win",
        ]
    )

    df_out = df[output_columns].copy()
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCSV comparativo salvo em: {OUTPUT_FILE}")
    print("\nPrimeiras 10 previsões:")
    print(df_out.head(10).to_string(index=False))

    prediction_metrics = build_empty_metrics(
        num_matches=int(len(df)),
        num_evaluated=num_evaluated,
    )

    if has_real_result:
        evaluation_indices = evaluation_mask.to_numpy()
        y_true = target_numeric.loc[evaluation_mask].astype(int).to_numpy()
        y_pred_evaluation = y_pred[evaluation_indices]
        raw_evaluation = raw_probabilities[evaluation_indices]
        calibrated_evaluation = calibrated_probabilities[evaluation_indices]

        metrics = calculate_metrics(y_true, y_pred_evaluation)
        raw_log_loss = log_loss(
            y_true,
            raw_evaluation,
            labels=CLASS_ORDER,
        )
        calibrated_log_loss = log_loss(
            y_true,
            calibrated_evaluation,
            labels=CLASS_ORDER,
        )

        report = classification_report(
            y_true,
            y_pred_evaluation,
            labels=CLASS_ORDER,
            target_names=list(RESULT_LABELS.values()),
            output_dict=True,
            zero_division=0,
        )

        class_metrics = []
        for class_id, class_name in RESULT_LABELS.items():
            class_result = report[class_name]
            class_metrics.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "precision": float(class_result["precision"]),
                    "recall": float(class_result["recall"]),
                    "f1_score": float(class_result["f1-score"]),
                    "support": int(class_result["support"]),
                }
            )

        class_metrics_df = pd.DataFrame(class_metrics)
        class_metrics_df.to_csv(CLASS_METRICS_FILE, index=False)

        confusion = confusion_matrix(
            y_true,
            y_pred_evaluation,
            labels=CLASS_ORDER,
        )

        print("\n" + "=" * 70)
        print("RESULTADOS — REGRESSÃO LOGÍSTICA — PARTIDAS DE 2026")
        print("=" * 70)
        print(f"Partidas avaliadas: {num_evaluated:,}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-macro: {metrics['f1_macro']:.4f}")
        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )
        print(f"Precision macro: {metrics['precision_macro']:.4f}")
        print(f"Recall macro: {metrics['recall_macro']:.4f}")
        print(
            "Log-loss (bruto / calibrado): "
            f"{raw_log_loss:.4f} / {calibrated_log_loss:.4f}"
        )

        print("\nMétricas por classe:")
        print(class_metrics_df.to_string(index=False))

        confusion_table = pd.DataFrame(
            confusion,
            index=[
                "Real: Visitante (0)",
                "Real: Empate (1)",
                "Real: Mandante (2)",
            ],
            columns=[
                "Pred: Visitante (0)",
                "Pred: Empate (1)",
                "Pred: Mandante (2)",
            ],
        )
        print("\nMatriz de confusão:")
        print(confusion_table.to_string())

        metrics_df = pd.DataFrame(
            [
                {
                    "modelo": "Logistic Regression",
                    "year": 2026,
                    "num_matches": num_evaluated,
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "log_loss_raw": raw_log_loss,
                    "log_loss_calibrated": calibrated_log_loss,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                }
            ]
        )
        metrics_df.to_csv(METRICS_CSV_FILE, index=False)

        fig, ax = plt.subplots(figsize=(8, 7))
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=[
                "Vitória\nVisitante",
                "Empate",
                "Vitória\nMandante",
            ],
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False,
        )
        ax.set_title(
            "Matriz de Confusão — Regressão Logística — 2026"
        )
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")
        fig.tight_layout()
        fig.savefig(
            CONFUSION_MATRIX_FILE,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        export_class_performance_chart(class_metrics_df)

        prediction_metrics.update(
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "log_loss_raw": float(raw_log_loss),
                "log_loss_calibrated": float(calibrated_log_loss),
                "balanced_accuracy": float(
                    metrics["balanced_accuracy"]
                ),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_per_class": {
                    f"{RESULT_LABELS[class_id]}_{class_id}": float(
                        report[RESULT_LABELS[class_id]]["f1-score"]
                    )
                    for class_id in CLASS_ORDER
                },
                "precision_per_class": {
                    f"{RESULT_LABELS[class_id]}_{class_id}": float(
                        report[RESULT_LABELS[class_id]]["precision"]
                    )
                    for class_id in CLASS_ORDER
                },
                "recall_per_class": {
                    f"{RESULT_LABELS[class_id]}_{class_id}": float(
                        report[RESULT_LABELS[class_id]]["recall"]
                    )
                    for class_id in CLASS_ORDER
                },
                "support_per_class": {
                    f"{RESULT_LABELS[class_id]}_{class_id}": int(
                        report[RESULT_LABELS[class_id]]["support"]
                    )
                    for class_id in CLASS_ORDER
                },
                "confusion_matrix": confusion.tolist(),
            }
        )

        print(f"\nMétricas globais salvas em: {METRICS_CSV_FILE}")
        print(f"Métricas por classe salvas em: {CLASS_METRICS_FILE}")
        print(f"Matriz de confusão salva em: {CONFUSION_MATRIX_FILE}")
        print(
            f"Desempenho por classe salvo em: "
            f"{CLASS_PERFORMANCE_FILE}"
        )
    else:
        print(
            "\n[AVISO] Nenhum resultado real está disponível. "
            "As previsões foram geradas, mas as métricas não foram "
            "calculadas."
        )

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            prediction_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Métricas estruturadas salvas em: {METRICS_FILE}")
    print("\nPredição da Regressão Logística concluída.")


if __name__ == "__main__":
    main()