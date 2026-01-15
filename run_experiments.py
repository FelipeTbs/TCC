#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script customizável para executar experimentos com diferentes valores de rel_thres
Permite escolher o range de valores a executar
"""

import subprocess
import numpy as np
from datetime import datetime
import os
import sys


def main():
    """
    Executa experimentos customizáveis com rel_thres
    """
    # Caminhos locais (mesma pasta do script)
    output_dir = 'experiments'
    resampling_script = 'resampling_apuana.py'
    
    # Usar o mesmo interpretador Python que está executando este script
    python_executable = sys.executable
    
    print("="*80)
    print("CONFIGURAÇÃO DE EXPERIMENTOS")
    print("="*80)
    print("\nVocê pode:")
    print("  1. Executar um range específico (ex: 0.50 a 0.58)")
    print("  2. Executar valores específicos (ex: 0.50, 0.60, 0.70)")
    print("  3. Executar todos os 26 experimentos (0.50 a 1.00)")
    print("="*80)
    
    # Solicitar entrada do usuário
    print("\nEscolha o modo:")
    print("  [1] Range (ex: 0.50 até 0.58)")
    print("  [2] Valores específicos (ex: 0.50, 0.60, 0.70)")
    print("  [3] Todos (0.50 até 1.00)")
    
    choice = input("\nDigite sua escolha (1, 2 ou 3): ").strip()
    
    if choice == "1":
        # Range
        start = float(input("Valor inicial (ex: 0.50): ").strip())
        end = float(input("Valor final (ex: 0.58): ").strip())
        step = float(input("Passo/incremento (padrão 0.02): ").strip() or "0.02")
        
        # Adicionar pequeno epsilon para incluir o valor final
        rel_thres_values = np.arange(start, end + 0.001, step)
        
    elif choice == "2":
        # Valores específicos
        values_input = input("Digite os valores separados por vírgula (ex: 0.50, 0.60, 0.70): ").strip()
        rel_thres_values = np.array([float(v.strip()) for v in values_input.split(",")])
        
    elif choice == "3":
        # Todos
        rel_thres_values = np.arange(0.50, 1.02, 0.02)
        
    else:
        print("❌ Escolha inválida! Usando todos os valores (0.50 a 1.00).")
        rel_thres_values = np.arange(0.50, 1.02, 0.02)
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("INICIANDO BATERIA DE EXPERIMENTOS")
    print("="*80)
    print(f"Valores de rel_thres: {len(rel_thres_values)} experimentos")
    print(f"Valores: {[f'{v:.2f}' for v in rel_thres_values]}")
    print(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Criar diretório de experimentos
    os.makedirs(output_dir, exist_ok=True)
    
    # Contador de sucessos e falhas
    success_count = 0
    failed_experiments = []
    
    # Executar cada experimento
    for i, rel_thres in enumerate(rel_thres_values):
        print(f"\n[{i+1}/{len(rel_thres_values)}] Executando rel_thres = {rel_thres:.2f}...")
        print(f"Hora: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Executar resampling_apuana.py com o mesmo Python do ambiente virtual
            result = subprocess.run(
                [python_executable, resampling_script, '--rel_thres', str(rel_thres), '--output_dir', output_dir],
                check=True,
                capture_output=False  # Mostrar saída em tempo real
            )
            
            print(f"✅ rel_thres={rel_thres:.2f} - CONCLUÍDO")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ rel_thres={rel_thres:.2f} - ERRO (código: {e.returncode})")
            failed_experiments.append(rel_thres)
        
        except Exception as e:
            print(f"❌ rel_thres={rel_thres:.2f} - ERRO: {e}")
            failed_experiments.append(rel_thres)
    
    # Relatório final
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("RELATÓRIO FINAL")
    print("="*80)
    print(f"Total de experimentos: {len(rel_thres_values)}")
    print(f"Bem-sucedidos: {success_count}")
    print(f"Com erro: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nExperimentos que falharam:")
        for rt in failed_experiments:
            print(f"  - rel_thres = {rt:.2f}")
    
    print(f"\nTempo total: {duration}")
    print(f"Término: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Salvar resumo em arquivo
    summary_file = f"{output_dir}/summary_execution.txt"
    with open(summary_file, 'a') as f:  # 'a' para append (adicionar ao arquivo)
        f.write(f"\n{'='*50}\n")
        f.write(f"Execução concluída: {end_time}\n")
        f.write(f"Range executado: {[f'{v:.2f}' for v in rel_thres_values]}\n")
        f.write(f"Tempo total: {duration}\n")
        f.write(f"Sucessos: {success_count}/{len(rel_thres_values)}\n")
        f.write(f"Falhas: {len(failed_experiments)}/{len(rel_thres_values)}\n")
        if failed_experiments:
            f.write(f"Experimentos que falharam:\n")
            for rt in failed_experiments:
                f.write(f"  - rel_thres = {rt:.2f}\n")
    
    print(f"\n📄 Resumo salvo/atualizado em: {summary_file}")


if __name__ == "__main__":
    main()
