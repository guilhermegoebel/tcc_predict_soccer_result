import pandas as pd
from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parent / 'ML_XGBoost_01' / 'train_xgboost.py'

spec = importlib.util.spec_from_file_location('train_xgboost_module', MODULE_PATH)
train_xgboost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_xgboost)


def test_export_executive_summary_table(tmp_path):
    metrics_df = pd.DataFrame([
        {'split': 'Validação', 'accuracy': 0.5440, 'f1_macro': 0.5058, 'balanced_accuracy': 0.5095, 'precision_macro': 0.5066, 'recall_macro': 0.5095},
        {'split': 'Teste', 'accuracy': 0.5737, 'f1_macro': 0.5277, 'balanced_accuracy': 0.5346, 'precision_macro': 0.5257, 'recall_macro': 0.5346},
    ])

    output_path = tmp_path / 'resumo_executivo.png'
    train_xgboost.export_summary_table(metrics_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_class_performance_table(tmp_path):
    report = pd.DataFrame({
        'precision': [0.54, 0.33, 0.71],
        'recall': [0.65, 0.27, 0.68],
        'f1-score': [0.59, 0.30, 0.69],
    }, index=['away_win', 'draw', 'home_win'])

    output_path = tmp_path / 'desempenho_por_classe.png'
    train_xgboost.export_class_performance_table(report, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
