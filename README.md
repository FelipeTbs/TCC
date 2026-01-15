# Análise da Sensibilidade do Limiar de Relevância em Regressão Desbalanceada

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-TCC%202026-red.svg)](docs/TCC_COMPLETO_ATUALIZADO.pdf)

> Investigação sistemática do impacto do parâmetro `rel_thres` (relevance threshold) em problemas de regressão desbalanceada através de **561.600 experimentos** em **30 datasets**.

**Autor:** Felipe Basto Tabosa, Juscimara Gomes Avelino
**Instituição:** 1Centro de Informática –  Universidade Federal de Pernambuco (UFPE)  
**Ano:** 2026

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Principais Descobertas](#-principais-descobertas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Reproduzindo o Experimento](#-reproduzindo-o-experimento)


---

## 🎯 Visão Geral

Este repositório contém o código, dados e análises do trabalho de conclusão de curso que investiga sistematicamente a sensibilidade do parâmetro **relevance threshold (rel_thres)** em problemas de regressão desbalanceada.

### Motivação

Em regressão desbalanceada, o `rel_thres` define quais valores são considerados "raros" ou "relevantes" (φ(y) > rel_thres) e, portanto, devem receber tratamento especial pelos algoritmos de balanceamento. Apesar de sua importância teórica, existiam lacunas significativas sobre seu impacto real no desempenho preditivo.

### Escopo Experimental

- **30 datasets** de domínios diversos
- **26 valores de rel_thres** (0.50 a 1.00, incrementos de 0.02)
- **6 estratégias de balanceamento** (GN, RO, SMT, RU, SG, WC)
- **6 modelos de regressão** (XGBoost, Random Forest, Bagging, Decision Tree, MLP, SVR)
- **20 folds** de validação cruzada (10 folds × 2 repetições)
- **Total: 561.600 experimentos**

---

## 🔬 Principais Descobertas

### 1. Impacto Global do rel_thres: Não Significativo

```
Teste Kruskal-Wallis: H=11.09, p=0.988
```

**Conclusão:** O rel_thres **não apresenta impacto estatisticamente significativo** no desempenho médio.

### 2. Heterogeneidade Entre Datasets

| Classe de Sensibilidade | Datasets | Percentual | Característica |
|--------------------------|----------|------------|----------------|
| **ALTA** (RV > 2.5) | 3 | 10.0% | Pequenos, alta assimetria |
| **MODERADA** (1.0 < RV ≤ 1.5) | 2 | 6.7% | Tamanho médio |
| **MODERADA-BAIXA** (0.7 < RV ≤ 1.0) | 6 | 20.0% | Diversos |
| **BAIXA** (RV ≤ 0.7) | 19 | 63.3% | Grandes, bem representados |

**RV = Razão de Variabilidade:** (SERA_max - SERA_min) / SERA_mediana

### 3. Estratégias de Balanceamento: Altamente Significativas

```
Teste Kruskal-Wallis: H=146.98, p<0.001
```

| Ranking | Estratégia | SERA Médio | vs 1º |
|---------|------------|------------|-------|
| 1º | **Gaussian Noise (GN)** | 1.65×10⁹ | — |
| 2º | Random Oversampling (RO) | 1.81×10⁹ | +9.7% |
| 3º | SmoteR (SMT) | 1.83×10⁹ | +10.9% |
| 4º | Random Undersampling (RU) | 2.05×10⁹ | +24.2% |
| 5º | SMOGN (SG) | 2.15×10⁹ | +30.3% |
| 6º | WERCS (WC) | 4.49×10⁹ | +172% |

### 4. Modelos: XGBoost Domina

| Ranking | Modelo | SERA Médio | vs XGBoost |
|---------|--------|------------|------------|
| 1º | **XGBoost** | 3.51×10⁸ | — |
| 2º | Random Forest | 6.61×10⁸ | +88% |
| 3º | Bagging | 6.80×10⁸ | +94% |
| 4º | Decision Tree | 7.03×10⁸ | +100% |
| 5º | MLP | 5.16×10⁹ | +1,370% |
| 6º | SVR | 6.43×10⁹ | +1,733% |

### 5. Valores Extremos: Problemáticos

Datasets com **rel_thres ≥ 0.88** apresentam:
- Outliers **24.3% mais severos**
- Falhas silenciosas em algoritmos de balanceamento
- Pior caso: rel_thres=0.98 → SERA=8.23×10¹¹

### 6. Não Há Configuração Universal Ótima

Análise das melhores combinações (rel_thres × estratégia × modelo) por dataset:
- **30 datasets → 30 configurações diferentes**
- Random Forest ótimo em **43.3%** dos casos individuais
- SmoteR ótimo em **40.0%** dos casos
- rel_thres ótimo concentra-se em **0.50-0.70** (60%)

**Paradoxo:** GN e XGBoost são melhores na **média global**, mas raramente são ótimos **individualmente**.
- **Explicação:** Trade-off entre consistência (GN/XGBoost) vs excelência pontual (SMT/RF)

---

## 📁 Estrutura do Projeto
```
TCC/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências Python
├── LICENSE                             # Licença MIT
├── .gitignore                          # Arquivos ignorados
│
├── data/                               # Datasets originais
│   ├── raw/                            # Dados brutos
│   └── processed/                      # Dados processados
│
├── experiments/                        # Resultados dos experimentos
│   ├── rel_thres_0.50/                 # Resultados por threshold
│   │   └── appendices/
│   │       ├── GN/                     # Resultados por estratégia
│   │       │   └── data/
│   │       │       └── boston/         # Resultados por dataset
│   │       │           └── XGBRegressor/  # Resultados por modelo
│   │       │               ├── Test1_GN_XGBRegressor.csv
│   │       │               ├── Pred1_GN_XGBRegressor.csv
│   │       │               └── ... (20 folds × 2 arquivos)
│   │       ├── RO/
│   │       ├── SMT/
│   │       ├── RU/
│   │       ├── SG/
│   │       └── WC/
│   ├── rel_thres_0.52/
│   ├── rel_thres_0.54/
│   └── ... (até rel_thres_1.00)
│
├── logs/                               # Logs de execução
│
├── resultados_configs/                 # Resultados de análises
│   ├── tabela_melhores_configuracoes.csv
│   ├── tabela_configuracoes_completa.csv
│   └── padroes_configuracoes.csv
│
├── temp_train_files/                   # Arquivos temporários de treino
│
├── test_output/                        # Saídas de teste
│
├── venv/                               # Ambiente virtual Python
│
├── analysis_script.py                  # Script de análise principal
├── analyze_best_configurations.py      # Análise de melhores configurações
├── analyze_datasets_sensitivity.py     # Análise de sensibilidade por dataset
├── resampling_apuana.py                # Script de reamostragem
└── run_experiments.py                  # Orquestração de experimentos
```
---

## 🚀 Instalação

### Requisitos

- Python 3.8+
- 8 GB RAM (mínimo)
- 50 GB espaço em disco (para dados completos)

### Passo 1: Clone o repositório

```bash
git clone https://github.com/FelipeTbs/TCC.git
cd TCC
```

### Passo 2: Crie ambiente virtual

**Opção A: venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

**Opção B: conda**
```bash
conda env create -f environment.yml
conda activate rel-thres
```

### Passo 3: Instale dependências

```bash
pip install -r requirements.txt
```

**Bibliotecas principais:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
tqdm>=4.64.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
```

---

## 💻 Uso

### Análise Rápida (Dados Pré-computados)

Se você tem acesso aos **resultados já computados**, pode analisar diretamente:

#### 1. Análise por rel_thres

```bash
python src/analysis/analyze_rel_thres.py data/results/ -o results/
```

**Gera:**
- `tabela_rel_thres.csv` - SERA por threshold
- `tabela_estrategias.csv` - Por estratégia
- `tabela_modelos.csv` - Por modelo
- `estatisticas.txt` - Testes estatísticos

#### 2. Análise por dataset

```bash
python src/analysis/analyze_datasets.py data/results/ -o results/
```

**Gera:**
- `tabela_datasets_sensibilidade.csv` - RV por dataset
- `tabela_datasets_completa_artigo.csv` - Tabela 5 do artigo
- `estatisticas_por_classe.csv` - Agregação por classe

#### 3. Melhores configurações

```bash
python src/analysis/analyze_best_configs.py data/results/ -o results/
```

**Gera:**
- `tabela_melhores_configuracoes.csv` - Melhor config por dataset
- `padroes_configuracoes.csv` - Padrões agregados

**⏱️ Tempo:** 5-60 minutos dependendo do hardware

### Reproduzindo Experimentos

**⚠️ ATENÇÃO:** Experimentos completos requerem **semanas** de computação!

```bash
python src/experiments/run_experiments.py \
    --datasets data/raw/ \
    --rel-thres-start 0.50 \
    --rel-thres-end 1.00 \
    --rel-thres-step 0.02 \
    --strategies GN RO SMT RU SG WC \
    --models XGBRegressor RandomForestRegressor BaggingRegressor \
             DecisionTreeRegressor MLPRegressor SVR \
    --folds 10 \
    --repeats 2 \
    --output data/results/
```

**Recursos necessários:**
- Cluster computacional ou
- GPU (recomendado para XGBoost/MLP)
- Tempo estimado: 2-4 semanas

---

## 📊 Resultados

### Tabelas Principais

Todas as tabelas estão disponíveis em `results/tables/`:

1. **Tabela 1:** Comparação rel_thres baixo vs alto
2. **Tabela 2:** Comportamento em rel_thres ≥ 0.88
3. **Tabela 3:** Ranking de estratégias
4. **Tabela 4:** Ranking de modelos
5. **Tabela 5:** Sensibilidade por dataset (30 datasets)
6. **Tabela 6:** Melhores configurações por dataset

### Visualizações

```bash
python src/analysis/visualization.py results/tables/ -o results/figures/
```

**Gráficos gerados:**
- Distribuição de RV (sensibilidade)
- SERA por rel_thres (boxplot)
- Comparação de estratégias (barplot)
- Comparação de modelos (barplot)
- Heatmap de interações

---

### Artigo Completo

O TCC completo está disponível em [`docs/TCC_COMPLETO_ATUALIZADO.pdf`](docs/TCC_COMPLETO_ATUALIZADO.pdf).

**Seções principais:**
1. Introdução
2. Fundamentação Teórica
3. Metodologia
4. Resultados
   - 4.2. Impacto do rel_thres
   - 4.3. Comparação de Estratégias
   - 4.4. Comparação de Modelos
   - 4.5. Sensibilidade por Dataset
   - 4.6. Interações Triplas
5. Discussão
6. Conclusão

---

## 📚 Referências Principais

1. **Avelino, J. G., Cavalcanti, G. D. C., & Cruz, R. M. O. (2024).** Resampling strategies for imbalanced regression: a survey and empirical analysis. *Artificial Intelligence Review*, 57(82).

2. **Branco, P., Torgo, L., & Ribeiro, R. P. (2019).** Pre-processing approaches for imbalanced distributions in regression. *Neurocomputing*, 343, 76-99.

3. **Ribeiro, R. P., & Moniz, N. (2020).** Imbalanced regression and extreme value prediction. *Machine Learning*, 109(9), 1803-1835.

4. **Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).** SMOTE for regression. *Portuguese Conference on Artificial Intelligence*, 378-389.
