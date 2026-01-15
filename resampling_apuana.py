#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de Resampling para Regressão com Dados Desbalanceados
Adaptado para execução no Cluster Apuana
Versão com rel_thres configurável
"""

import argparse
import pandas as pd
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from glob import glob
import numpy as np
import os
import smogn
import resreg
from xgboost import XGBRegressor
import itertools as it
from datetime import datetime

from imbalance_metrics import regression_metrics as rm
import ImbalancedLearningRegression as iblr

import warnings
warnings.filterwarnings('ignore')


def train(regressor, strategy, X, y, dataset_name, rel_thres):
    """
    Treina o modelo com a estratégia de balanceamento especificada
    """
    train = np.column_stack((y, X))
    
    # Criar diretório temporário se não existir
    temp_dir = 'temp_train_files'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Remover qualquer caminho do dataset_name, manter apenas o nome do arquivo
    clean_dataset_name = os.path.basename(dataset_name).replace('.csv', '')
    
    train_output_file = os.path.join(temp_dir, f"train_SG_{clean_dataset_name}.csv")
    pd.DataFrame(train).to_csv(train_output_file, index=False)
    train = pd.read_csv(train_output_file)

    try:
        train = balance(train, strategy, rel_thres)
    except ValueError:
        pass

    X = train.drop([train.columns[0]], axis=1)
    y = train[train.columns[0]]

    model = regressor.fit(X.values, y.values)

    return model


def balance(train, strategy, rel_thres):
    """
    Aplica a estratégia de balanceamento aos dados de treino com parâmetros fixos
    """
    if strategy == "GN":
        train = iblr.gn(data=train, y="0", samp_method="balance", pert=0.1, rel_thres=rel_thres)
    elif strategy == "RO":
        train = iblr.ro(data=train, y="0", samp_method="balance", rel_thres=rel_thres)
    elif strategy == "RU":
        train = iblr.random_under(data=train, y="0", samp_method="balance", rel_thres=rel_thres)
    elif strategy == "SG":
        train = train.dropna()
        train = smogn.smoter(data=train, y=train.columns[0], samp_method="balance", k=5, 
                            pert=0.1, rel_xtrm_type='high', rel_thres=rel_thres)
        train = train.dropna()
    elif strategy == "SMT":
        train = iblr.smote(data=train, y="0", samp_method="balance", rel_thres=rel_thres)
    elif strategy == "WC":
        X_train = train.drop([train.columns[0]], axis=1)
        y_train = train[train.columns[0]]
        relevance = resreg.pdf_relevance(y_train)
        X_wercs, y_wercs = resreg.wercs(X_train, y_train, relevance, over=0.5, under=0.5)
        train = pd.DataFrame(np.column_stack((y_wercs, X_wercs)))
    
    return train


def repeatedKfold(X, y, dataset_name, rel_thres, output_base_dir, summary_data):
    """
    Executa validação cruzada com diferentes estratégias e modelos (SEM busca de hiperparâmetros)
    """
    outer = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)

    print(f"  Configuração: {outer}")

    # Estratégias com parâmetros fixos (SEM grid search)
    strategys = ["SMT", "RO", "RU", "GN", "SG", "WC"]

    regressors = {
        'BG': BaggingRegressor(),
        'DT': DecisionTreeRegressor(),
        'MLP': MLPRegressor(),
        'RF': RandomForestRegressor(),
        'SVM': SVR(),
        'XG': XGBRegressor()
    }

    for strategy in strategys:
        print(f"\n  {'='*50}")
        print(f"  Estratégia: {strategy}")
        print(f"  {'='*50}")

        for regressor_name, regressor in regressors.items():
            print(f"\n    Modelo: {regressor_name}")
            
            for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
                print(f"      Fold: {fold}", end=" ")
                X_train_outer, X_test_outer = X[train_index], X[test_index]
                y_train_outer, y_test_outer = y[train_index], y[test_index]

                # Treinar diretamente sem busca de hiperparâmetros
                model_outer = train(regressor, strategy, X_train_outer, y_train_outer, dataset_name, rel_thres)
                y_pred_outer = model_outer.predict(X_test_outer)
                sera_outer = rm.sera(y_test_outer, y_pred_outer)
                print(f"| SERA: {sera_outer:.4f}")

                model_name = type(model_outer).__name__

                # Criar estrutura de pastas e salvar predições
                output_dir = f'{output_base_dir}/appendices/{strategy}/{dataset_name}/{model_name}'
                os.makedirs(output_dir, exist_ok=True)

                # Salvar predições
                pred = np.column_stack((test_index, y_pred_outer))
                pd.DataFrame(pred, columns=['test_index', 'y_pred']).to_csv(
                    f'{output_dir}/Pred{fold}_{strategy}_{model_name}.csv', index=False
                )

                # Salvar valores reais para comparação
                test = np.column_stack((test_index, y_test_outer))
                pd.DataFrame(test, columns=['test_index', 'y_true']).to_csv(
                    f'{output_dir}/Test{fold}_{strategy}_{model_name}.csv', index=False
                )

                # Adicionar ao resumo
                summary_data.append({
                    'rel_thres': rel_thres,
                    'strategy': strategy,
                    'dataset': dataset_name,
                    'model': model_name,
                    'fold': fold,
                    'sera_score': sera_outer,
                    'config': 'fixed_params'  # Indica que usou parâmetros fixos
                })


def main():
    """
    Função principal que processa todos os datasets com um rel_thres específico
    """
    parser = argparse.ArgumentParser(description='Resampling com rel_thres configurável')
    parser.add_argument('--rel_thres', type=float, required=True, 
                        help='Valor do relevance threshold (ex: 0.5, 0.8, 1.0)')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Diretório base para salvar resultados')
    
    args = parser.parse_args()
    rel_thres = args.rel_thres
    
    # Criar diretório de saída para este rel_thres
    output_base_dir = f"{args.output_dir}/rel_thres_{rel_thres:.2f}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Configurar log
    log_file = f"{output_base_dir}/log_rel_thres_{rel_thres:.2f}.txt"
    
    print("="*80)
    print(f"INICIANDO EXPERIMENTO COM rel_thres = {rel_thres}")
    print(f"Diretório de saída: {output_base_dir}")
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    with open(log_file, 'w') as f:
        f.write(f"Experimento iniciado: {datetime.now()}\n")
        f.write(f"rel_thres = {rel_thres}\n\n")
    
    # Verificar se os datasets existem
    data_sets = sorted(glob(r'data/*.csv'))
    
    if len(data_sets) == 0:
        error_msg = "❌ ERRO: Nenhum dataset encontrado na pasta 'data/'"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(error_msg + "\n")
        return
    
    print(f"✅ Datasets encontrados: {len(data_sets)}")
    for ds in data_sets:
        print(f"  - {ds}")
    
    # Lista para armazenar resultados do resumo
    summary_data = []
    
    # Processar cada dataset
    for i, dataset in enumerate(data_sets):
        print(f"\n{'='*80}")
        print(f"PROCESSANDO DATASET {i+1}/{len(data_sets)}: {dataset}")
        print('='*80)

        ds = pd.read_csv(dataset)
        dataset_name = dataset.split('/')[-1].replace('.csv', '')

        X = ds.drop([ds.columns[0]], axis=1)
        y = ds[ds.columns[0]]
        X = X.to_numpy()
        y = y.to_numpy()

        repeatedKfold(X, y, dataset_name, rel_thres, output_base_dir, summary_data)

        print(f"\n✅ Dataset '{dataset_name}' concluído!")
        
        with open(log_file, 'a') as f:
            f.write(f"Dataset '{dataset_name}' processado com sucesso\n")
    
    # Salvar resumo consolidado
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_base_dir}/summary_rel_thres_{rel_thres:.2f}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "="*80)
    print("PROCESSAMENTO COMPLETO!")
    print(f"Resumo salvo em: {summary_file}")
    print("="*80)
    
    with open(log_file, 'a') as f:
        f.write(f"\nExperimento concluído: {datetime.now()}\n")
        f.write(f"Resumo salvo em: {summary_file}\n")


if __name__ == "__main__":
    main()
