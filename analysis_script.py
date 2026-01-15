#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de Análise Completa dos Resultados
Análise estatística e geração de gráficos para o artigo científico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_all_results(base_dir='experiments'):
    """
    Carrega todos os resultados dos experimentos
    """
    print("="*80)
    print("1. CARREGANDO DADOS")
    print("="*80)
    
    summary_files = glob(f'{base_dir}/rel_thres_*/summary_rel_thres_*.csv')
    
    if len(summary_files) == 0:
        print(f" ERRO: Nenhum arquivo de resultado encontrado em '{base_dir}/'")
        return None
    
    print(f" Encontrados {len(summary_files)} arquivos de resultado")
    
    all_results = []
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            print(f"  Erro ao ler {file}: {e}")
    
    df_complete = pd.concat(all_results, ignore_index=True)
    
    # Salvar dataset consolidado
    df_complete.to_csv(f'{base_dir}/complete_analysis.csv', index=False)
    print(f" Dataset consolidado salvo em '{base_dir}/complete_analysis.csv'")
    
    # Estatísticas gerais
    print(f"\n Resumo dos dados:")
    print(f"  - Total de experimentos: {len(df_complete):,}")
    print(f"  - Datasets: {df_complete['dataset'].nunique()}")
    print(f"  - Estratégias: {list(df_complete['strategy'].unique())}")
    print(f"  - Modelos: {list(df_complete['model'].unique())}")
    print(f"  - Valores de rel_thres: {len(df_complete['rel_thres'].unique())}")
    print(f"  - Range rel_thres: {df_complete['rel_thres'].min():.2f} - {df_complete['rel_thres'].max():.2f}")
    
    return df_complete


def analyze_rel_thres_impact(df, output_dir='experiments/analysis'):
    """
    Análise 1: Impacto do rel_thres no desempenho
    """
    print("\n" + "="*80)
    print("2. ANÁLISE: IMPACTO DO rel_thres")
    print("="*80)
    
    # Estatísticas por rel_thres
    sera_by_threshold = df.groupby('rel_thres')['sera_score'].agg([
        ('média', 'mean'),
        ('desvio', 'std'),
        ('mediana', 'median'),
        ('mínimo', 'min'),
        ('máximo', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75)),
        ('n', 'count')
    ]).round(4)
    
    print("\n Estatísticas por rel_thres:")
    print(sera_by_threshold)
    
    # Salvar tabela
    sera_by_threshold.to_csv(f'{output_dir}/table_rel_thres_stats.csv')
    
    # Tentar salvar LaTeX (opcional)
    try:
        sera_by_threshold.to_latex(f'{output_dir}/table_rel_thres_stats.tex')
    except ImportError:
        print("    Jinja2 não instalado, pulando geração de LaTeX")
    
    print(f"\n Tabela salva em '{output_dir}/table_rel_thres_stats.csv'")
    
    # Identificar melhor e pior threshold
    best_threshold = sera_by_threshold['média'].idxmin()
    worst_threshold = sera_by_threshold['média'].idxmax()
    
    best_sera = sera_by_threshold.loc[best_threshold, 'média']
    worst_sera = sera_by_threshold.loc[worst_threshold, 'média']
    improvement = ((worst_sera - best_sera) / worst_sera) * 100
    
    print(f"\n Melhor rel_thres: {best_threshold:.2f}")
    print(f"   SERA médio: {best_sera:.4f} ± {sera_by_threshold.loc[best_threshold, 'desvio']:.4f}")
    
    print(f"\n Pior rel_thres: {worst_threshold:.2f}")
    print(f"   SERA médio: {worst_sera:.4f} ± {sera_by_threshold.loc[worst_threshold, 'desvio']:.4f}")
    
    print(f"\n Melhoria: {improvement:.2f}% (melhor vs pior)")
    
    # Gráfico 1: Linha com erro
    plt.figure(figsize=(12, 6))
    plt.errorbar(sera_by_threshold.index, 
                 sera_by_threshold['média'], 
                 yerr=sera_by_threshold['desvio'],
                 marker='o', markersize=8, capsize=5, capthick=2,
                 linewidth=2, elinewidth=2, alpha=0.8)
    
    plt.axvline(best_threshold, color='green', linestyle='--', alpha=0.5, label=f'Melhor ({best_threshold:.2f})')
    plt.axvline(worst_threshold, color='red', linestyle='--', alpha=0.5, label=f'Pior ({worst_threshold:.2f})')
    
    plt.xlabel('rel_thres', fontsize=14, fontweight='bold')
    plt.ylabel('SERA (Squared Error Relevance Area)', fontsize=14, fontweight='bold')
    plt.title('Impacto do rel_thres no Desempenho Médio', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_rel_thres_impact.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: '{output_dir}/fig1_rel_thres_impact.png'")
    plt.close()
    
    # Gráfico 2: Boxplot por faixas
    plt.figure(figsize=(12, 6))
    
    # Criar faixas
    df['rel_thres_range'] = pd.cut(df['rel_thres'], 
                                     bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                     labels=['0.50-0.60', '0.60-0.70', '0.70-0.80', '0.80-0.90', '0.90-1.00'])
    
    sns.boxplot(data=df, x='rel_thres_range', y='sera_score', palette='Set2')
    plt.xlabel('Faixa de rel_thres', fontsize=14, fontweight='bold')
    plt.ylabel('SERA', fontsize=14, fontweight='bold')
    plt.title('Distribuição de SERA por Faixa de rel_thres', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_rel_thres_boxplot.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: '{output_dir}/fig2_rel_thres_boxplot.png'")
    plt.close()
    
    return sera_by_threshold


def analyze_strategies(df, output_dir='experiments/analysis'):
    """
    Análise 2: Comparação de estratégias de sampling
    """
    print("\n" + "="*80)
    print("3. ANÁLISE: COMPARAÇÃO DE ESTRATÉGIAS")
    print("="*80)
    
    # Estatísticas por estratégia
    strategy_stats = df.groupby('strategy')['sera_score'].agg([
        ('média', 'mean'),
        ('desvio', 'std'),
        ('mediana', 'median'),
        ('n', 'count')
    ]).sort_values('média').round(4)
    
    print("\n Desempenho por Estratégia:")
    print(strategy_stats)
    
    # Salvar tabela
    strategy_stats.to_csv(f'{output_dir}/table_strategies_stats.csv')
    
    # Tentar salvar LaTeX (opcional)
    try:
        strategy_stats.to_latex(f'{output_dir}/table_strategies_stats.tex')
    except ImportError:
        pass
    
    print(f"\n Tabela salva em '{output_dir}/table_strategies_stats.csv'")
    
    # Melhor e pior estratégia
    best_strategy = strategy_stats.index[0]
    worst_strategy = strategy_stats.index[-1]
    
    print(f"\n Melhor estratégia: {best_strategy}")
    print(f"   SERA médio: {strategy_stats.loc[best_strategy, 'média']:.4f}")
    
    print(f"\n Pior estratégia: {worst_strategy}")
    print(f"   SERA médio: {strategy_stats.loc[worst_strategy, 'média']:.4f}")
    
    # Gráfico: Boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='strategy', y='sera_score', 
                order=strategy_stats.index, palette='Set3')
    sns.stripplot(data=df, x='strategy', y='sera_score', 
                  order=strategy_stats.index, color='black', 
                  alpha=0.3, size=2)
    
    plt.xlabel('Estratégia de Sampling', fontsize=14, fontweight='bold')
    plt.ylabel('SERA', fontsize=14, fontweight='bold')
    plt.title('Comparação de Estratégias de Balanceamento', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_strategies_comparison.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: '{output_dir}/fig3_strategies_comparison.png'")
    plt.close()
    
    return strategy_stats


def analyze_models(df, output_dir='experiments/analysis'):
    """
    Análise 3: Desempenho dos modelos
    """
    print("\n" + "="*80)
    print("4. ANÁLISE: COMPARAÇÃO DE MODELOS")
    print("="*80)
    
    # Estatísticas por modelo
    model_stats = df.groupby('model')['sera_score'].agg([
        ('média', 'mean'),
        ('desvio', 'std'),
        ('mediana', 'median'),
        ('n', 'count')
    ]).sort_values('média').round(4)
    
    print("\n Desempenho por Modelo:")
    print(model_stats)
    
    # Salvar tabela
    model_stats.to_csv(f'{output_dir}/table_models_stats.csv')
    
    # Tentar salvar LaTeX (opcional)
    try:
        model_stats.to_latex(f'{output_dir}/table_models_stats.tex')
    except ImportError:
        pass
    
    print(f"\n Tabela salva em '{output_dir}/table_models_stats.csv'")
    
    # Gráfico: Barplot com erro
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df, x='model', y='sera_score', 
                     order=model_stats.index, palette='viridis',
                     errorbar='sd', capsize=0.1)
    
    # Adicionar valores no topo das barras
    for i, (idx, row) in enumerate(model_stats.iterrows()):
        ax.text(i, row['média'], f"{row['média']:.2f}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Modelo de Regressão', fontsize=14, fontweight='bold')
    plt.ylabel('SERA médio', fontsize=14, fontweight='bold')
    plt.title('Comparação de Modelos de Regressão', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_models_comparison.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: '{output_dir}/fig4_models_comparison.png'")
    plt.close()
    
    return model_stats


def analyze_interactions(df, output_dir='experiments/analysis'):
    """
    Análise 4: Interações entre rel_thres e estratégias
    """
    print("\n" + "="*80)
    print("5. ANÁLISE: INTERAÇÕES (rel_thres × Estratégias)")
    print("="*80)
    
    # Pivot table
    pivot_table = df.pivot_table(values='sera_score', 
                                   index='strategy', 
                                   columns='rel_thres', 
                                   aggfunc='mean')
    
    print("\n Tabela de interações (primeiras colunas):")
    print(pivot_table.iloc[:, :10])
    
    # Salvar
    pivot_table.to_csv(f'{output_dir}/table_interaction_matrix.csv')
    print(f"\n Matriz salva em '{output_dir}/table_interaction_matrix.csv'")
    
    # Heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_table, annot=False, cmap='RdYlGn_r', 
                cbar_kws={'label': 'SERA médio'}, 
                linewidths=0.5, linecolor='gray')
    plt.xlabel('rel_thres', fontsize=14, fontweight='bold')
    plt.ylabel('Estratégia', fontsize=14, fontweight='bold')
    plt.title('Heatmap: Interação entre rel_thres e Estratégias', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_heatmap_interaction.png', dpi=300, bbox_inches='tight')
    print(f" Heatmap salvo: '{output_dir}/fig5_heatmap_interaction.png'")
    plt.close()
    
    # Linhas por estratégia
    plt.figure(figsize=(14, 7))
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].groupby('rel_thres')['sera_score'].mean()
        plt.plot(strategy_data.index, strategy_data.values, 
                marker='o', label=strategy, linewidth=2, alpha=0.8)
    
    plt.xlabel('rel_thres', fontsize=14, fontweight='bold')
    plt.ylabel('SERA médio', fontsize=14, fontweight='bold')
    plt.title('Evolução do SERA por Estratégia', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_strategies_evolution.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: '{output_dir}/fig6_strategies_evolution.png'")
    plt.close()


def statistical_tests(df, output_dir='experiments/analysis'):
    """
    Análise 5: Testes estatísticos
    """
    print("\n" + "="*80)
    print("6. TESTES ESTATÍSTICOS")
    print("="*80)
    
    results_file = open(f'{output_dir}/statistical_tests_results.txt', 'w', encoding='utf-8')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    # Teste de normalidade
    write_and_print("\n Teste de Normalidade (Shapiro-Wilk):")
    sample = df['sera_score'].sample(min(5000, len(df)))  # Máx 5000 para eficiência
    stat, p_value = stats.shapiro(sample)
    write_and_print(f"  Estatística W: {stat:.6f}")
    write_and_print(f"  p-value: {p_value:.6e}")
    write_and_print(f"  Distribuição: {'Normal' if p_value > 0.05 else 'Não-normal'} (alpha=0.05)")
    
    is_normal = p_value > 0.05
    
    # Teste para rel_thres
    write_and_print("\n Teste de Diferença entre grupos de rel_thres:")
    groups = [df[df['rel_thres'] == rt]['sera_score'].values 
              for rt in sorted(df['rel_thres'].unique())]
    
    if is_normal:
        # ANOVA
        f_stat, p_anova = stats.f_oneway(*groups)
        write_and_print(f"  Teste: ANOVA")
        write_and_print(f"  F-statistic: {f_stat:.4f}")
        write_and_print(f"  p-value: {p_anova:.6e}")
        test_p = p_anova
    else:
        # Kruskal-Wallis
        h_stat, p_kw = stats.kruskal(*groups)
        write_and_print(f"  Teste: Kruskal-Wallis")
        write_and_print(f"  H-statistic: {h_stat:.4f}")
        write_and_print(f"  p-value: {p_kw:.6e}")
        test_p = p_kw
    
    write_and_print(f"  Resultado: Diferença {'SIGNIFICATIVA ✓' if test_p < 0.05 else 'não significativa'} (alpha=0.05)")
    
    # Teste para estratégias
    write_and_print("\n Teste de Diferença entre Estratégias:")
    strategy_groups = [df[df['strategy'] == s]['sera_score'].values 
                       for s in df['strategy'].unique()]
    
    if is_normal:
        f_stat_s, p_s = stats.f_oneway(*strategy_groups)
        write_and_print(f"  Teste: ANOVA")
        write_and_print(f"  F-statistic: {f_stat_s:.4f}")
        write_and_print(f"  p-value: {p_s:.6e}")
    else:
        h_stat_s, p_s = stats.kruskal(*strategy_groups)
        write_and_print(f"  Teste: Kruskal-Wallis")
        write_and_print(f"  H-statistic: {h_stat_s:.4f}")
        write_and_print(f"  p-value: {p_s:.6e}")
    
    write_and_print(f"  Resultado: Diferença {'SIGNIFICATIVA ✓' if p_s < 0.05 else 'não significativa'} (alpha=0.05)")
    
    # Post-hoc se significativo
    if p_s < 0.05:
        write_and_print("\n Teste Post-hoc (Mann-Whitney U com correção de Bonferroni):")
        strategies = sorted(df['strategy'].unique())
        n_comparisons = len(list(combinations(strategies, 2)))
        alpha_corrected = 0.05 / n_comparisons
        
        write_and_print(f"  Número de comparações: {n_comparisons}")
        write_and_print(f"  alpha corrigido (Bonferroni): {alpha_corrected:.6f}")
        write_and_print(f"\n  Comparações significativas:")
        
        sig_count = 0
        for s1, s2 in combinations(strategies, 2):
            group1 = df[df['strategy'] == s1]['sera_score']
            group2 = df[df['strategy'] == s2]['sera_score']
            stat_mw, p_mw = stats.mannwhitneyu(group1, group2)
            
            if p_mw < alpha_corrected:
                sig_count += 1
                write_and_print(f"    {s1} vs {s2}: U={stat_mw:.2f}, p={p_mw:.6e} *")
        
        write_and_print(f"\n  Total de diferenças significativas: {sig_count}/{n_comparisons}")
    
    results_file.close()
    print(f"\n Resultados salvos em '{output_dir}/statistical_tests_results.txt'")


def generate_top_configurations(df, output_dir='experiments/analysis', top_n=20):
    """
    Análise 6: Ranking das melhores configurações
    """
    print("\n" + "="*80)
    print(f"7. TOP {top_n} MELHORES CONFIGURAÇÕES")
    print("="*80)
    
    # Agrupar por configuração
    config_performance = df.groupby(['rel_thres', 'strategy', 'model'])['sera_score'].agg([
        ('SERA_médio', 'mean'),
        ('SERA_std', 'std'),
        ('n_experimentos', 'count')
    ]).reset_index()
    
    # Ordenar
    best_configs = config_performance.sort_values('SERA_médio').head(top_n).round(4)
    best_configs.index = range(1, top_n + 1)
    best_configs.index.name = 'Rank'
    
    print(f"\n Top {top_n} configurações:")
    print(best_configs)
    
    # Salvar
    best_configs.to_csv(f'{output_dir}/table_top{top_n}_configs.csv')
    
    # Tentar salvar LaTeX (opcional)
    try:
        best_configs.to_latex(f'{output_dir}/table_top{top_n}_configs.tex')
    except ImportError:
        pass
    
    print(f"\n Ranking salvo em '{output_dir}/table_top{top_n}_configs.csv'")
    
    return best_configs


def generate_summary_report(df, output_dir='experiments/analysis'):
    """
    Gera relatório resumido para o artigo
    """
    print("\n" + "="*80)
    print("8. GERANDO RELATÓRIO RESUMIDO")
    print("="*80)
    
    report_file = open(f'{output_dir}/summary_report.txt', 'w', encoding='utf-8')
    
    def write_report(text):
        report_file.write(text + '\n')
    
    write_report("="*80)
    write_report("RELATÓRIO RESUMIDO - ANÁLISE DE RESULTADOS")
    write_report("="*80)
    write_report("")
    
    # Informações gerais
    write_report("1. INFORMAÇÕES GERAIS")
    write_report(f"  - Total de experimentos: {len(df):,}")
    write_report(f"  - Datasets analisados: {df['dataset'].nunique()}")
    write_report(f"  - Estratégias testadas: {len(df['strategy'].unique())}")
    write_report(f"  - Modelos avaliados: {len(df['model'].unique())}")
    write_report(f"  - Range de rel_thres: {df['rel_thres'].min():.2f} - {df['rel_thres'].max():.2f}")
    write_report("")
    
    # Melhor configuração
    best_idx = df['sera_score'].idxmin()
    best_row = df.loc[best_idx]
    
    write_report("2. MELHOR CONFIGURAÇÃO GLOBAL")
    write_report(f"  - rel_thres: {best_row['rel_thres']:.2f}")
    write_report(f"  - Estratégia: {best_row['strategy']}")
    write_report(f"  - Modelo: {best_row['model']}")
    write_report(f"  - Dataset: {best_row['dataset']}")
    write_report(f"  - SERA: {best_row['sera_score']:.4f}")
    write_report("")
    
    # Melhores por categoria
    write_report("3. MELHORES POR CATEGORIA")
    
    best_threshold = df.groupby('rel_thres')['sera_score'].mean().idxmin()
    write_report(f"  - Melhor rel_thres: {best_threshold:.2f}")
    write_report(f"    SERA médio: {df[df['rel_thres']==best_threshold]['sera_score'].mean():.4f}")
    
    best_strategy = df.groupby('strategy')['sera_score'].mean().idxmin()
    write_report(f"  - Melhor estratégia: {best_strategy}")
    write_report(f"    SERA médio: {df[df['strategy']==best_strategy]['sera_score'].mean():.4f}")
    
    best_model = df.groupby('model')['sera_score'].mean().idxmin()
    write_report(f"  - Melhor modelo: {best_model}")
    write_report(f"    SERA médio: {df[df['model']==best_model]['sera_score'].mean():.4f}")
    write_report("")
    
    # Estatísticas gerais
    write_report("4. ESTATÍSTICAS GERAIS DE SERA")
    write_report(f"  - Média: {df['sera_score'].mean():.4f}")
    write_report(f"  - Desvio padrão: {df['sera_score'].std():.4f}")
    write_report(f"  - Mediana: {df['sera_score'].median():.4f}")
    write_report(f"  - Mínimo: {df['sera_score'].min():.4f}")
    write_report(f"  - Máximo: {df['sera_score'].max():.4f}")
    write_report(f"  - Q1 (25%): {df['sera_score'].quantile(0.25):.4f}")
    write_report(f"  - Q3 (75%): {df['sera_score'].quantile(0.75):.4f}")
    
    report_file.close()
    print(f" Relatório salvo em '{output_dir}/summary_report.txt'")
    
    # Exibir na tela também
    with open(f'{output_dir}/summary_report.txt', 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    """
    Função principal - executa todas as análises
    """
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA DOS RESULTADOS EXPERIMENTAIS")
    print("Script para geração de figuras e tabelas do artigo")
    print("="*80)
    
    # Criar diretório de análise
    output_dir = 'experiments/analysis'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carregar dados
    df = load_all_results('experiments')
    
    if df is None:
        print("\n Não foi possível carregar os dados. Encerrando.")
        return
    
    # 2. Análises
    sera_by_threshold = analyze_rel_thres_impact(df, output_dir)
    strategy_stats = analyze_strategies(df, output_dir)
    model_stats = analyze_models(df, output_dir)
    analyze_interactions(df, output_dir)
    statistical_tests(df, output_dir)
    best_configs = generate_top_configurations(df, output_dir, top_n=20)
    generate_summary_report(df, output_dir)
    
    # Resumo final
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO! ")
    print("="*80)
    print(f"\n Todos os arquivos foram salvos em: '{output_dir}/'")
    print("\n Arquivos gerados:")
    print("  Gráficos:")
    print("    - fig1_rel_thres_impact.png")
    print("    - fig2_rel_thres_boxplot.png")
    print("    - fig3_strategies_comparison.png")
    print("    - fig4_models_comparison.png")
    print("    - fig5_heatmap_interaction.png")
    print("    - fig6_strategies_evolution.png")
    print("\n  Tabelas:")
    print("    - table_rel_thres_stats.csv / .tex")
    print("    - table_strategies_stats.csv / .tex")
    print("    - table_models_stats.csv / .tex")
    print("    - table_interaction_matrix.csv")
    print("    - table_top20_configs.csv / .tex")
    print("\n  Relatórios:")
    print("    - statistical_tests_results.txt")
    print("    - summary_report.txt")
    print("    - complete_analysis.csv (dataset consolidado)")
    print("\n Utilize esses arquivos no seu artigo científico!")
    print("="*80)


if __name__ == "__main__":
    main()