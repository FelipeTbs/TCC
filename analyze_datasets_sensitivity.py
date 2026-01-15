#!/usr/bin/env python3
"""
Analisador por Dataset - Calcula variação de SERA por dataset
==============================================================

Este programa complementa o analyze_rel_thres_AJUSTADO.py gerando
a análise individual POR DATASET através dos diferentes valores de rel_thres.

Gera a Tabela 5 do artigo com:
- SERA_min, SERA_max, SERA_mediano por dataset
- Razão de Variabilidade (RV)
- Classificação de sensibilidade

Autor: Felipe Basto Tabosa
Data: Janeiro 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


class DatasetSensitivityAnalyzer:
    """Analisa sensibilidade de cada dataset ao rel_thres"""
    
    def __init__(self, base_dir, verbose=True):
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        
        self.rel_thres_values = [round(0.50 + i*0.02, 2) for i in range(26)]
        self.strategies = ['GN', 'RO', 'SMT', 'RU', 'SG', 'WC']
        self.models = [
            'XGBRegressor', 'RandomForestRegressor', 'BaggingRegressor',
            'DecisionTreeRegressor', 'MLPRegressor', 'SVR'
        ]
        
        self.log("="*80)
        self.log("ANÁLISE DE SENSIBILIDADE POR DATASET")
        self.log("="*80)
        
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def find_rel_thres_dir(self, rel_thres):
        """Localiza diretório de um rel_thres"""
        rel_str_dot = f"{rel_thres:.2f}"
        rel_str_underscore = f"{rel_thres:.2f}".replace('.', '_')
        
        patterns = [
            f"rel_thres_{rel_str_dot}",
            f"rel_thres_{rel_str_underscore}",
            f"rel_thresh_{rel_str_dot}",
            f"rel_thresh_{rel_str_underscore}",
        ]
        
        for pattern in patterns:
            dir_path = self.base_dir / pattern
            if dir_path.exists() and dir_path.is_dir():
                return dir_path
        
        return None
    
    def calculate_sera(self, test_file, pred_file):
        """Calcula SERA de um par Test/Pred"""
        try:
            test = pd.read_csv(test_file)
            pred = pd.read_csv(pred_file)
            
            y_true = test['y_true'].values
            y_pred = pred['y_pred'].values
            
            sera = np.sum((y_true - y_pred) ** 2)
            return sera
            
        except Exception:
            return np.nan
    
    def get_sera_for_dataset(self, rel_thres_dir, dataset_name):
        """
        Calcula SERA médio de um dataset em um rel_thres
        
        Agrega todos os modelos e estratégias
        """
        appendices_dir = rel_thres_dir / 'appendices'
        
        if not appendices_dir.exists():
            return np.nan
        
        all_seras = []
        
        # Percorrer todas as estratégias
        for strategy in self.strategies:
            strategy_dir = appendices_dir / strategy
            
            if not strategy_dir.exists():
                continue
            
            data_dir = strategy_dir / 'data'
            if not data_dir.exists():
                if (strategy_dir / dataset_name).exists():
                    data_dir = strategy_dir
                else:
                    continue
            
            dataset_dir = data_dir / dataset_name
            
            if not dataset_dir.exists():
                continue
            
            # Percorrer todos os modelos
            for model_name in self.models:
                model_dir = dataset_dir / model_name
                
                if not model_dir.exists():
                    continue
                
                # Calcular SERA de cada fold
                for fold_idx in range(1, 21):
                    test_file = model_dir / f"Test{fold_idx}_{strategy}_{model_name}.csv"
                    pred_file = model_dir / f"Pred{fold_idx}_{strategy}_{model_name}.csv"
                    
                    if test_file.exists() and pred_file.exists():
                        sera = self.calculate_sera(test_file, pred_file)
                        if not np.isnan(sera):
                            all_seras.append(sera)
        
        if len(all_seras) == 0:
            return np.nan
        
        # Retornar SERA médio deste dataset neste rel_thres
        return np.mean(all_seras)
    
    def discover_datasets(self):
        """
        Descobre automaticamente quais datasets existem
        """
        self.log("\n🔍 Descobrindo datasets disponíveis...")
        
        datasets = set()
        
        # Pegar primeiro rel_thres disponível
        first_rel_thres = None
        for rel_thres in self.rel_thres_values:
            rel_thres_dir = self.find_rel_thres_dir(rel_thres)
            if rel_thres_dir:
                first_rel_thres = rel_thres_dir
                break
        
        if not first_rel_thres:
            self.log("  ❌ Nenhum rel_thres encontrado!")
            return []
        
        # Explorar estrutura
        appendices_dir = first_rel_thres / 'appendices'
        
        if appendices_dir.exists():
            for strategy_dir in appendices_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue
                
                data_dir = strategy_dir / 'data'
                if not data_dir.exists():
                    data_dir = strategy_dir
                
                for dataset_dir in data_dir.iterdir():
                    if dataset_dir.is_dir():
                        datasets.add(dataset_dir.name)
        
        datasets = sorted(list(datasets))
        
        self.log(f"  ✓ {len(datasets)} datasets encontrados")
        if self.verbose:
            self.log(f"  📋 Datasets: {', '.join(datasets[:10])}" + 
                    (f" ... (+{len(datasets)-10})" if len(datasets) > 10 else ""))
        
        return datasets
    
    def analyze_dataset(self, dataset_name):
        """
        Analisa um dataset através de todos os rel_thres
        """
        sera_by_rel_thres = {}
        
        for rel_thres in self.rel_thres_values:
            rel_thres_dir = self.find_rel_thres_dir(rel_thres)
            
            if not rel_thres_dir:
                continue
            
            sera = self.get_sera_for_dataset(rel_thres_dir, dataset_name)
            
            if not np.isnan(sera):
                sera_by_rel_thres[rel_thres] = sera
        
        if len(sera_by_rel_thres) == 0:
            return None
        
        # Calcular estatísticas
        seras = list(sera_by_rel_thres.values())
        
        sera_min = np.min(seras)
        sera_max = np.max(seras)
        sera_median = np.median(seras)
        sera_mean = np.mean(seras)
        variacao = sera_max - sera_min
        
        # Razão de Variabilidade (RV)
        if sera_median > 0:
            rv = variacao / sera_median
        else:
            rv = 0
        
        # Classificação de sensibilidade
        if rv > 2.5:
            classe = 'ALTA'
        elif rv > 1.5:
            classe = 'MOD-ALTA'
        elif rv > 1.0:
            classe = 'MOD'
        elif rv > 0.7:
            classe = 'MOD-BAIXA'
        else:
            classe = 'BAIXA'
        
        return {
            'dataset': dataset_name,
            'sera_min': sera_min,
            'sera_max': sera_max,
            'sera_median': sera_median,
            'sera_mean': sera_mean,
            'variacao': variacao,
            'rv': rv,
            'classe': classe,
            'n_rel_thres': len(sera_by_rel_thres)
        }
    
    def analyze_all_datasets(self):
        """Analisa todos os datasets"""
        self.log("\n" + "="*80)
        self.log("ANALISANDO TODOS OS DATASETS")
        self.log("="*80)
        
        # Descobrir datasets
        datasets = self.discover_datasets()
        
        if len(datasets) == 0:
            self.log("\n❌ Nenhum dataset encontrado!")
            return []
        
        # Analisar cada dataset
        results = []
        
        for dataset in tqdm(datasets, desc="Processando datasets", 
                           disable=not self.verbose):
            if self.verbose:
                self.log(f"\n📊 Analisando: {dataset}")
            
            result = self.analyze_dataset(dataset)
            
            if result:
                results.append(result)
                if self.verbose:
                    self.log(f"  ✓ RV={result['rv']:.2f} ({result['classe']})")
        
        self.log(f"\n✅ {len(results)}/{len(datasets)} datasets analisados")
        
        return results
    
    def create_tables(self, results, output_dir='.'):
        """Cria tabelas de saída"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.log("\n" + "="*80)
        self.log("CRIANDO TABELAS")
        self.log("="*80)
        
        # DataFrame principal
        df = pd.DataFrame(results)
        
        # Ordenar por RV (sensibilidade)
        df = df.sort_values('rv', ascending=False)
        
        # Tabela completa
        complete_file = output_dir / 'tabela_datasets_sensibilidade.csv'
        df.to_csv(complete_file, index=False, float_format='%.6e')
        self.log(f"\n✓ Tabela completa: {complete_file}")
        
        # Tabela para TCC (simplificada)
        df_tcc = df[[
            'dataset', 'sera_median', 'variacao', 'rv', 'classe'
        ]].copy()
        
        df_tcc.columns = [
            'Dataset', 'SERA Mediano', 'Variacao', 'RV', 'Classe'
        ]
        
        tcc_file = output_dir / 'tabela_datasets_tcc.csv'
        df_tcc.to_csv(tcc_file, index=False, float_format='%.2e')
        self.log(f"✓ Tabela TCC: {tcc_file}")
        
        # Tabela estilo artigo (com min e max)
        df_artigo = df[[
            'dataset', 'sera_min', 'sera_max', 'sera_median', 
            'variacao', 'rv', 'classe'
        ]].copy()
        
        df_artigo.columns = [
            'Dataset', 'SERA Min', 'SERA Max', 'SERA Mediano',
            'Variacao', 'RV', 'Classe'
        ]
        
        artigo_file = output_dir / 'tabela_datasets_completa_artigo.csv'
        df_artigo.to_csv(artigo_file, index=False, float_format='%.2e')
        self.log(f"✓ Tabela artigo: {artigo_file}")
        
        # Estatísticas por classe
        stats_by_class = df.groupby('classe').agg({
            'dataset': 'count',
            'rv': 'mean',
            'sera_median': 'mean'
        }).reset_index()
        
        stats_by_class.columns = ['Classe', 'N_Datasets', 'RV_Medio', 'SERA_Medio']
        
        stats_file = output_dir / 'estatisticas_por_classe.csv'
        stats_by_class.to_csv(stats_file, index=False, float_format='%.2e')
        self.log(f"✓ Estatísticas por classe: {stats_file}")
        
        # Resumo
        self.log("\n" + "="*80)
        self.log("RESUMO")
        self.log("="*80)
        
        self.log(f"\n📊 Distribuição por sensibilidade:")
        for classe in ['ALTA', 'MOD-ALTA', 'MOD', 'MOD-BAIXA', 'BAIXA']:
            count = (df['classe'] == classe).sum()
            pct = (count / len(df)) * 100
            self.log(f"  • {classe:10s}: {count:2d} datasets ({pct:5.1f}%)")
        
        self.log(f"\n📈 Estatísticas gerais:")
        self.log(f"  • RV médio: {df['rv'].mean():.2f}")
        self.log(f"  • RV máximo: {df['rv'].max():.2f} ({df.loc[df['rv'].idxmax(), 'dataset']})")
        self.log(f"  • RV mínimo: {df['rv'].min():.2f} ({df.loc[df['rv'].idxmin(), 'dataset']})")
        
        alta_sens = df[df['classe'].isin(['ALTA', 'MOD-ALTA'])]
        self.log(f"\n⚠️  Datasets sensíveis (ALTA + MOD-ALTA):")
        self.log(f"  • Total: {len(alta_sens)} ({len(alta_sens)/len(df)*100:.1f}%)")
        self.log(f"  • RV médio: {alta_sens['rv'].mean():.2f}")
        
        return df, df_tcc, df_artigo, stats_by_class


def main():
    parser = argparse.ArgumentParser(
        description='Análise de Sensibilidade por Dataset',
        epilog="""
Exemplo:
  python %(prog)s D:\\TCC\\experiments
  python %(prog)s D:\\TCC\\experiments -o D:\\TCC\\resultados_datasets
        """
    )
    
    parser.add_argument('data_dir', help='Diretório base (ex: D:\\TCC\\experiments)')
    parser.add_argument('-o', '--output', default='.', help='Diretório de saída')
    parser.add_argument('-q', '--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ ERRO: Diretório não encontrado: {args.data_dir}")
        return 1
    
    analyzer = DatasetSensitivityAnalyzer(args.data_dir, verbose=not args.quiet)
    
    results = analyzer.analyze_all_datasets()
    
    if len(results) == 0:
        print("\n❌ ERRO: Nenhum resultado obtido!")
        return 1
    
    df, df_tcc, df_artigo, stats = analyzer.create_tables(results, args.output)
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("="*80)
    print(f"\n📁 Arquivos gerados em: {Path(args.output).absolute()}")
    print("  1. tabela_datasets_completa_artigo.csv  ⭐ PARA O ARTIGO (Tabela 5)")
    print("  2. tabela_datasets_sensibilidade.csv    - Dados completos")
    print("  3. tabela_datasets_tcc.csv              - Versão simplificada")
    print("  4. estatisticas_por_classe.csv          - Resumo por classe")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
