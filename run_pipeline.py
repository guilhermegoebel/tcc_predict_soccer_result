"""Pipeline unificado do projeto de XGBoost.

Este script executa em sequência:
1) geração do dataset principal (`football_matches_ml.csv`)
2) treino do modelo XGBoost
3) aplicação do modelo em `football_matches_2026.csv`

Ele foi criado para tornar o fluxo reproduzível em um único comando.

Uso:
    python run_pipeline.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(ROOT, 'script.py')
TRAIN_PATH = os.path.join(ROOT, 'ML_XGBoost_01', 'train_xgboost.py')
PREDICT_PATH = os.path.join(ROOT, 'ML_XGBoost_01', 'predict_2026.py')

REQUIRED_PACKAGES = [
    'xgboost',
    'scikit-learn',
    'pandas',
    'numpy',
    'matplotlib'
]

STEPS = [
    ('Gerar dataset principal', [sys.executable, SCRIPT_PATH]),
    ('Treinar modelo XGBoost', [sys.executable, TRAIN_PATH]),
    ('Rodar predição em 2026', [sys.executable, PREDICT_PATH]),
]

OUTPUTS = [
    'football_matches_ml.csv',
    os.path.join('ML_XGBoost_01', 'xgb_match_result.json'),
    os.path.join('ML_XGBoost_01', 'preprocessing_artifacts.pkl'),
    os.path.join('ML_XGBoost_01', 'confusion_matrix.png'),
    os.path.join('ML_XGBoost_01', 'feature_importance.png'),
    os.path.join('ML_XGBoost_01', 'evaluation_metrics.csv'),
    os.path.join('ML_XGBoost_01', 'evaluation_metrics.json'),
    os.path.join('ML_XGBoost_01', 'comparativo_real_vs_predito_2026.csv'),
    os.path.join('ML_XGBoost_01', 'prediction_metrics.json'),
]


def check_dependencies():
    missing_packages = []
    for pkg in REQUIRED_PACKAGES:
        import_name = 'sklearn' if pkg == 'scikit-learn' else pkg
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print('Dependências Python ausentes detectadas:')
        for pkg in missing_packages:
            print(f' - {pkg}')
        print('\nInstale-as com:')
        print(f'    python -m pip install {' '.join(REQUIRED_PACKAGES)}')
        sys.exit(1)


def run_step(name, command):
    print('\n' + '=' * 80)
    print(f'START: {name}')
    print('=' * 80)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f'Etapa falhou: {name} (exit {result.returncode})')
    print(f'OK: {name}')


def main():
    check_dependencies()
    print('Pipeline unificado de XGBoost para previsão de resultados de partidas')
    print('Executando as etapas em sequência...\n')

    for name, command in STEPS:
        run_step(name, command)

    print('\nPipeline concluído com sucesso!')
    print('Arquivos gerados:')
    for output in OUTPUTS:
        print(f' - {output}')

    print('\nO fluxo final inclui:')
    print('  1. `football_matches_ml.csv`: dataset de treinamento e teste gerado a partir dos dados brutos.')
    print('  2. `ML_XGBoost_01/xgb_match_result.json`: modelo XGBoost treinado.')
    print('  3. `ML_XGBoost_01/preprocessing_artifacts.pkl`: artefatos de pré-processamento usados pelo modelo.')
    print('  4. `ML_XGBoost_01/confusion_matrix.png` e `ML_XGBoost_01/feature_importance.png`: gráficos de avaliação.')
    print('  5. `ML_XGBoost_01/evaluation_metrics.csv` e `evaluation_metrics.json`: tabela de métricas de validação e teste.')
    print('  6. `ML_XGBoost_01/comparativo_real_vs_predito_2026.csv`: predições para as partidas de 2026.')
    print('  7. `ML_XGBoost_01/prediction_metrics.json`: métricas da predição em 2026.')

    print('\nPróximo passo: revise os arquivos de métricas e o CSV de predição para validar a performance do modelo.')


if __name__ == '__main__':
    main()
