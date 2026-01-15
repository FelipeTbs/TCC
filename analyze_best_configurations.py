#!/usr/bin/env python3
"""
Analisador de Melhor Configuração por Dataset
==============================================

Encontra a melhor combinação (rel_thres × modelo × estratégia) para cada dataset.

Para cada dataset, identifica:
- Melhor rel_thres
- Melhor estratégia
- Melhor modelo
- SERA mínimo alcançado
- Ganho vs pior configuração

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


class BestConfigurationAnalyzer:
    """Encontra melhor configuração por dataset"""
    
    def __init__(self, base_dir, verbose=True):
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        
        self.rel_thres_values = [round(0.50 + i*0.02, 2) for i in range(26)]
        self.strategies = ['GN', 'RO', 'SMT', 'RU', 'SG', 'WC']
        self.models = [
            'XGBRegressor', 'RandomForestRegressor', 'BaggingRegressor',
            'DecisionTreeRegressor', 'MLPRegressor', 'SVR'
        ]
        
        # Cache de resultados
        self.results_cache = {}
        
        self.log("="*80)
        self.log("ANÁLISE DE MELHOR CONFIGURAÇÃO POR DATASET")
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
    
    def get_sera_for_config(self, rel_thres_dir, dataset_name, strategy, model):
        """
        Calcula SERA para uma configuração específica
        
        Args:
            rel_thres_dir: Diretório do rel_thres
            dataset_name: Nome do dataset
            strategy: Nome da estratégia
            model: Nome do modelo
            
        Returns:
            float: SERA médio ou np.nan
        """
        appendices_dir = rel_thres_dir / 'appendices'
        
        if not appendices_dir.exists():
            return np.nan
        
        strategy_dir = appendices_dir / strategy
        if not strategy_dir.exists():
            return np.nan
        
        data_dir = strategy_dir / 'data'
        if not data_dir.exists():
            if (strategy_dir / dataset_name).exists():
                data_dir = strategy_dir
            else:
                return np.nan
        
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            return np.nan
        
        model_dir = dataset_dir / model
        if not model_dir.exists():
            return np.nan
        
        # Calcular SERA de todos os folds
        seras = []
        for fold_idx in range(1, 21):
            test_file = model_dir / f"Test{fold_idx}_{strategy}_{model}.csv"
            pred_file = model_dir / f"Pred{fold_idx}_{strategy}_{model}.csv"
            
            if test_file.exists() and pred_file.exists():
                sera = self.calculate_sera(test_file, pred_file)
                if not np.isnan(sera):
                    seras.append(sera)
        
        if len(seras) == 0:
            return np.nan
        
        return np.mean(seras)
    
    def discover_datasets(self):
        """Descobre datasets disponíveis"""
        self.log("\n🔍 Descobrindo datasets...")
        
        datasets = set()
        
        first_rel_thres = None
        for rel_thres in self.rel_thres_values:
            rel_thres_dir = self.find_rel_thres_dir(rel_thres)
            if rel_thres_dir:
                first_rel_thres = rel_thres_dir
                break
        
        if not first_rel_thres:
            return []
        
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
        
        return datasets
    
    def analyze_dataset(self, dataset_name):
        """
        Analisa um dataset e encontra melhor configuração
        
        Returns:
            dict: Informações da melhor configuração
        """
        self.log(f"\n📊 Analisando: {dataset_name}")
        
        # Testar todas as combinações
        configurations = []
        
        for rel_thres in self.rel_thres_values:
            rel_thres_dir = self.find_rel_thres_dir(rel_thres)
            if not rel_thres_dir:
                continue
            
            for strategy in self.strategies:
                for model in self.models:
                    sera = self.get_sera_for_config(
                        rel_thres_dir, dataset_name, strategy, model
                    )
                    
                    if not np.isnan(sera):
                        configurations.append({
                            'rel_thres': rel_thres,
                            'strategy': strategy,
                            'model': model,
                            'sera': sera
                        })
        
        if len(configurations) == 0:
            self.log(f"  ⚠️ Nenhuma configuração válida")
            return None
        
        # Encontrar melhor e pior
        df_configs = pd.DataFrame(configurations)
        
        best_idx = df_configs['sera'].idxmin()
        worst_idx = df_configs['sera'].idxmax()
        
        best = df_configs.loc[best_idx]
        worst = df_configs.loc[worst_idx]
        
        # Calcular ganho
        ganho_absoluto = worst['sera'] - best['sera']
        ganho_percentual = (ganho_absoluto / worst['sera']) * 100
        
        result = {
            'dataset': dataset_name,
            'best_rel_thres': best['rel_thres'],
            'best_strategy': best['strategy'],
            'best_model': best['model'],
            'best_sera': best['sera'],
            'worst_sera': worst['sera'],
            'ganho_absoluto': ganho_absoluto,
            'ganho_percentual': ganho_percentual,
            'n_configs': len(configurations)
        }
        
        self.log(f"  ✓ Melhor: {best['model']} + {best['strategy']} + rel_thres={best['rel_thres']:.2f}")
        self.log(f"    SERA: {best['sera']:.2e} (ganho: {ganho_percentual:.1f}%)")
        
        return result
    
    def analyze_all_datasets(self):
        """Analisa todos os datasets"""
        self.log("\n" + "="*80)
        self.log("ANALISANDO TODOS OS DATASETS")
        self.log("="*80)
        
        datasets = self.discover_datasets()
        
        if len(datasets) == 0:
            self.log("\n❌ Nenhum dataset encontrado!")
            return []
        
        results = []
        
        for dataset in tqdm(datasets, desc="Processando", disable=not self.verbose):
            result = self.analyze_dataset(dataset)
            if result:
                results.append(result)
        
        self.log(f"\n✅ {len(results)}/{len(datasets)} datasets analisados")
        
        return results
    
    def create_tables(self, results, output_dir='.'):
        """Cria tabelas de saída"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.log("\n" + "="*80)
        self.log("CRIANDO TABELAS")
        self.log("="*80)
        
        df = pd.DataFrame(results)
        
        # Ordenar por ganho percentual
        df = df.sort_values('ganho_percentual', ascending=False)
        
        # Tabela principal
        df_main = df[[
            'dataset', 'best_rel_thres', 'best_strategy', 'best_model',
            'best_sera', 'ganho_percentual'
        ]].copy()
        
        df_main.columns = [
            'Dataset', 'Melhor rel_thres', 'Melhor Estratégia', 'Melhor Modelo',
            'SERA Mínimo', 'Ganho (%)'
        ]
        
        main_file = output_dir / 'tabela_melhores_configuracoes.csv'
        df_main.to_csv(main_file, index=False, float_format='%.2e')
        self.log(f"\n✓ Tabela principal: {main_file}")
        
        # Tabela completa (com pior)
        df_complete = df[[
            'dataset', 'best_rel_thres', 'best_strategy', 'best_model',
            'best_sera', 'worst_sera', 'ganho_absoluto', 'ganho_percentual'
        ]].copy()
        
        df_complete.columns = [
            'Dataset', 'rel_thres', 'Estratégia', 'Modelo',
            'SERA Melhor', 'SERA Pior', 'Ganho Absoluto', 'Ganho %'
        ]
        
        complete_file = output_dir / 'tabela_configuracoes_completa.csv'
        df_complete.to_csv(complete_file, index=False, float_format='%.2e')
        self.log(f"✓ Tabela completa: {complete_file}")
        
        # Análise de padrões
        self.log("\n" + "="*80)
        self.log("ANÁLISE DE PADRÕES")
        self.log("="*80)
        
        # Frequência de modelos
        self.log("\n📊 Modelos mais frequentes na melhor configuração:")
        model_counts = df['best_model'].value_counts()
        for model, count in model_counts.items():
            pct = (count / len(df)) * 100
            self.log(f"  • {model:25s}: {count:2d} datasets ({pct:5.1f}%)")
        
        # Frequência de estratégias
        self.log("\n📊 Estratégias mais frequentes:")
        strategy_counts = df['best_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            pct = (count / len(df)) * 100
            self.log(f"  • {strategy:5s}: {count:2d} datasets ({pct:5.1f}%)")
        
        # Distribuição de rel_thres
        self.log("\n📊 Distribuição de rel_thres ótimo:")
        df['rel_thres_faixa'] = pd.cut(
            df['best_rel_thres'],
            bins=[0.50, 0.70, 0.80, 0.90, 1.00],
            labels=['0.50-0.70', '0.70-0.80', '0.80-0.90', '0.90-1.00']
        )
        faixa_counts = df['rel_thres_faixa'].value_counts().sort_index()
        for faixa, count in faixa_counts.items():
            pct = (count / len(df)) * 100
            self.log(f"  • {faixa}: {count:2d} datasets ({pct:5.1f}%)")
        
        # Top 10 maiores ganhos
        self.log("\n📊 Top 10 datasets com maiores ganhos:")
        top10 = df.nlargest(10, 'ganho_percentual')
        for idx, row in top10.iterrows():
            self.log(f"  • {row['dataset']:20s}: {row['ganho_percentual']:6.1f}% "
                    f"({row['best_model'][:5]} + {row['best_strategy']} + {row['best_rel_thres']:.2f})")
        
        # Estatísticas de padrões
        patterns_data = {
            'Modelo': model_counts.to_dict(),
            'Estratégia': strategy_counts.to_dict(),
            'Faixa_rel_thres': faixa_counts.to_dict()
        }
        
        patterns_file = output_dir / 'padroes_configuracoes.csv'
        
        # Criar DataFrame de padrões
        patterns_rows = []
        for categoria, valores in patterns_data.items():
            for item, count in valores.items():
                patterns_rows.append({
                    'Categoria': categoria,
                    'Item': str(item),
                    'Frequencia': count,
                    'Percentual': (count / len(df)) * 100
                })
        
        df_patterns = pd.DataFrame(patterns_rows)
        df_patterns.to_csv(patterns_file, index=False, float_format='%.1f')
        self.log(f"\n✓ Padrões salvos: {patterns_file}")
        
        return df, df_main, df_complete, df_patterns


def main():
    parser = argparse.ArgumentParser(
        description='Análise de Melhores Configurações por Dataset',
        epilog="""
Exemplo:
  python %(prog)s D:\\TCC\\experiments
  python %(prog)s D:\\TCC\\experiments -o D:\\TCC\\resultados_configs
        """
    )
    
    parser.add_argument('data_dir', help='Diretório base')
    parser.add_argument('-o', '--output', default='.', help='Diretório de saída')
    parser.add_argument('-q', '--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ ERRO: Diretório não encontrado: {args.data_dir}")
        return 1
    
    analyzer = BestConfigurationAnalyzer(args.data_dir, verbose=not args.quiet)
    
    results = analyzer.analyze_all_datasets()
    
    if len(results) == 0:
        print("\n❌ ERRO: Nenhum resultado obtido!")
        return 1
    
    df, df_main, df_complete, df_patterns = analyzer.create_tables(results, args.output)
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("="*80)
    print(f"\n📁 Arquivos gerados em: {Path(args.output).absolute()}")
    print("  1. tabela_melhores_configuracoes.csv  ⭐ PARA O TCC")
    print("  2. tabela_configuracoes_completa.csv  - Com detalhes")
    print("  3. padroes_configuracoes.csv          - Análise de padrões")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
