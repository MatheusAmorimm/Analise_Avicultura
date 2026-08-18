<h1 align="center">🐔 Análise Preditiva de Conversão Alimentar em Avicultura de Corte</h1>

<p align="center">
  <em>Modelos de regressão aplicados a dados reais de uma integradora avícola</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Colab" />
  <img src="https://img.shields.io/badge/Projeto-Extensionista-534AB7?style=flat" alt="Extensionista" />
</p>

---

## 📌 Sobre o projeto

Projeto **extensionista** desenvolvido para a matéria de **Negócios**, com foco em **modelos de regressão e predição**.

Diferente de um exercício com base sintética, este trabalho foi construído sobre uma **base de dados real de uma empresa do ramo avícola** — 5.813 lotes de frango de corte, com 40 variáveis que cobrem todo o ciclo produtivo: alojamento, nutrição, genética, mortalidade semanal, consumo de ração e abate.

O objetivo é responder a uma pergunta que vale dinheiro para a integradora:

> **É possível prever a eficiência alimentar de um lote a partir das suas características produtivas — e, mais importante, quais fatores mais impactam essa eficiência?**

---

## 🎯 O problema de negócio

A **Conversão Alimentar (CA)** é o principal indicador de eficiência da avicultura de corte:

```
CA = kg de ração consumida / kg de peso vivo produzido
```

Quanto **menor** a CA, **mais eficiente** o lote. Na base analisada, a CA média é **1,762** — ou seja, são necessários ~1,76 kg de ração para produzir 1 kg de frango.

Como a ração representa a maior fatia do custo de produção, **cada centésimo de CA economizado se traduz em toneladas de ração** na escala da integradora. Um lote típico da base produz ~92.000 kg de frango consumindo ~162.000 kg de ração.

---

## 🗂️ A base de dados

| | |
|---|---|
| **Registros** | 5.813 lotes (5.796 após limpeza) |
| **Variáveis** | 40 originais → 99 preditores após one-hot encoding |
| **Variável-resposta** | `CA` (Conversão Alimentar) |
| **Origem** | Dados operacionais reais de integradora avícola (anonimizados) |

> ⚠️ **A base não está versionada neste repositório.** Por se tratar de dados operacionais reais de uma empresa, o arquivo `Base_Dados_Modelos_racao_peso.xlsx` está no `.gitignore`. O notebook está completo e com todas as saídas preservadas — é possível acompanhar toda a análise, gráficos e resultados sem executar o código.

**Grupos de variáveis:**

- **Cadastrais** — Produtor, Região, Técnico responsável, Tipo de Instalação
- **Manejo** — Área do galpão, Densidade (aves/m²), Vazio sanitário, Datas de alojamento e abate
- **Insumos** — Programa Nutricional, Linhagem Genética, Origem do pintinho, Peso do pintinho
- **Desempenho semanal** — Peso aos 7/14/21/28/35/42/49 dias, Mortalidade acumulada (%Mort7 … %Mort49)
- **Resultado** — Aves alojadas/abatidas, Peso abatido, Ração por fase (Pré, Inicial, Cres1, Cres2, Final), Sobra de ração, **CA**

---

## 🔬 Metodologia

```
Base bruta (5.813 lotes)
        │
        ├─ 1. EDA — distribuições, outliers, correlações, análise por categoria
        │
        ├─ 2. Limpeza — remoção de nulos (1) e outliers de CA (16 lotes, Q3 + 3×IQR)
        │
        ├─ 3. Feature engineering — criação de `Dias_Alojamento` (Data Abate − Data Alojamento)
        │
        ├─ 4. Encoding — one-hot em Região, Técnico, Instalação, Nutrição, Genética, Origem
        │
        ├─ 5. Split 80/20 + StandardScaler (fit só no treino)
        │
        └─ 6. Modelagem
               ├─ KNN Regressor  → escolha de k por validação cruzada 5-fold
               └─ Gradiente Descendente → implementação manual + SGDRegressor
```

### Os dois modelos

**🔵 KNN Regressor** — prevê a CA de um lote pela média dos *k* lotes mais parecidos na base de treino (distância euclidiana). A normalização é obrigatória aqui: sem ela, `Ração Total` (~170.000) dominaria completamente `Peso Pintinho` (~0,04).

**🟣 Gradiente Descendente** — ajusta os coeficientes iterativamente na direção que reduz o erro:

$$\beta^{(k+1)} = \beta^{(k)} - \alpha \cdot \nabla J(\beta^{(k)})$$

Implementado de **duas formas**: uma versão manual em NumPy (50 épocas, com curva de convergência do MSE, para fins didáticos) e o `SGDRegressor` do scikit-learn para as métricas formais.

---

## 📊 Resultados

### Comparação final

| Modelo | R² Treino | R² Teste | MSE Teste | MAE Teste | ΔR² (overfitting) |
|---|---:|---:|---:|---:|---:|
| KNN (k=25) | 0,2777 | 0,2429 | 0,0020 | 0,0344 | 0,0348 ✅ |
| **Gradiente Descendente (SGD)** | **0,5679** | **0,5599** | **0,0012** | **0,0264** | **0,0080** ✅ |

**O modelo linear venceu.** O SGD explica **56% da variação da CA** contra 24% do KNN — e com menos overfitting. Isso sugere que a relação entre as variáveis produtivas e a CA é predominantemente **linear e aditiva**, um cenário em que a busca por vizinhos em um espaço de 99 dimensões perde eficácia (maldição da dimensionalidade).

### Escolha do k no KNN

Em vez de escolher o k com maior R² no teste (que costuma estar em overfitting), o critério foi **o menor k com ΔR² ≤ 0,05**:

| k | R² Treino | R² Teste | R² CV (5-fold) | ΔR² |
|---:|---:|---:|---:|---:|
| 5 | 0,4793 | 0,2352 | 0,1842 | 0,2441 ❌ |
| 10 | 0,3763 | 0,2376 | 0,2160 | 0,1387 ❌ |
| 20 | 0,3009 | 0,2411 | 0,2092 | 0,0598 ⚠️ |
| **25** | **0,2777** | **0,2429** | **0,2038** | **0,0348** ✅ |
| 50 | 0,2273 | 0,2271 | 0,1829 | 0,0002 |
| 100 | 0,1853 | 0,1907 | 0,1570 | −0,0053 |

### Traduzindo o erro para linguagem de negócio

Um MAE de 0,0264 não diz nada sozinho. Aplicado a um lote típico (~92.000 kg de frango, ~162.000 kg de ração):

| Modelo | MAE | Erro absoluto | % do consumo do lote |
|---|---:|---:|---:|
| KNN (k=25) | 0,0344 | ≈ 3.165 kg de ração | 1,95% |
| SGD | 0,0264 | ≈ 2.429 kg de ração | **1,50%** |

---

## 💡 Principais insights para o cliente

### 1. Mortalidade tardia é o maior driver da CA

| Variável | Correlação com CA |
|---|---:|
| **%Mort49** (mortalidade acumulada até 49 dias) | **+0,501** |
| Idade Média ao abate | +0,208 |
| %Mort42 | +0,156 |
| %Mort35 | +0,154 |
| Aves Abatidas | −0,137 |

A mortalidade acumulada até o 49º dia é, isolada, o fator mais correlacionado com a CA (r = 0,50) — bem à frente de qualquer outro. E o padrão é claro: a correlação **cresce ao longo do ciclo** (%Mort21 ≈ 0,01 → %Mort49 = 0,50).

**Por quê?** Uma ave que morre na semana 7 já consumiu quase toda a ração do ciclo sem entregar peso no abate. O prejuízo de eficiência é máximo justamente no fim.

> **► Recomendação:** priorizar biosseguridade e monitoramento sanitário nas **semanas 5–7**. É a alavanca de maior impacto sobre a eficiência alimentar — mais do que ajustes de nutrição ou genética, que apresentaram variação bem menor entre categorias.

### 2. Ciclos mais longos pioram a conversão

`Idade Média ao abate` é o segundo maior driver (r = +0,21). Aves mais velhas convertem ração em peso com menos eficiência — cada dia extra de alojamento tem custo marginal crescente.

### 3. O que **não** explicou a CA

Variáveis de consumo bruto de ração (`Ração Total`, r = 0,02) praticamente não têm poder explicativo isolado — o que faz sentido, já que estão embutidas no próprio cálculo da CA como numerador e escalam junto com o tamanho do lote.

---

## 🧭 Decisões metodológicas

Cada escolha do notebook tem justificativa técnica documentada:

| Decisão | Por quê |
|---|---|
| Excluir `Produtor` | 333 categorias únicas → centenas de dummies sem ganho preditivo |
| Excluir `Num_Galpao` | Identificador numérico sem significado ordinal |
| Outlier: Q3 + 3×IQR | Critério conservador — remove só os 16 casos extremos (CA > 2,0), preservando a variabilidade natural |
| `StandardScaler` antes do KNN | Distância euclidiana exige escalas comparáveis |
| Split 80/20, `random_state=42` | Reprodutibilidade — mesmos lotes de treino/teste em qualquer execução |
| Faixa de k de 5 a 100 | k baixo (≤15) causava ΔR² > 0,15; foi preciso explorar k alto para achar a região saudável |
| k escolhido por ΔR² ≤ 0,05 | Menor k que generaliza bem, em vez do maior R² de teste |
| Validação cruzada 5-fold | Confirma estabilidade do resultado em 5 divisões diferentes do treino |
| `weights='uniform'` no KNN | `weights='distance'` levava a overfitting extremo (R² treino = 1,0) |
| `alpha=1e-4` no GD | Com 99 features, taxas maiores fazem os pesos divergirem |
| `learning_rate='constant'` no SGD | `invscaling`/`adaptive` com eta0 alto também divergiam |

> **Regra de ouro seguida em todo o pipeline:** todo pré-processamento é ajustado **apenas com os dados de treino** (`fit` no treino, `transform` em treino e teste), evitando vazamento de informação.

---

## 🚀 Como executar

### Só ler os resultados

O notebook está commitado **com todas as saídas** — tabelas, métricas e gráficos. Basta abri-lo aqui pelo GitHub para ver a análise inteira, sem precisar rodar nada nem ter a base em mãos.

### Reproduzir a análise

Como a base não é versionada (ver seção anterior), reproduzir o notebook exige ter o arquivo `Base_Dados_Modelos_racao_peso.xlsx` localmente.

**Google Colab** — foi como o projeto foi desenvolvido:

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Execute a **Etapa 1** — será aberto o seletor de arquivos
3. Faça upload da base
4. Rode as células restantes em ordem

**Localmente:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
jupyter notebook Analise_Avicultura_CA_Eric.ipynb
```

Rodando local, substitua a célula de upload do Colab por:

```python
df_raw = pd.read_excel('Base_Dados_Modelos_racao_peso.xlsx')
```

---

## 📁 Estrutura

```
.
├── Analise_Avicultura_CA_Eric.ipynb    # Notebook completo (EDA → modelagem → insights)
├── .gitignore
└── README.md

# Base_Dados_Modelos_racao_peso.xlsx    ← não versionada (dados da empresa)
```

---

## 🔭 Limitações e próximos passos

- **R² de 0,56 é um teto modesto** — boa parte da variação da CA depende de fatores não capturados na base (clima, qualidade do lote de ração, manejo diário).
- **Modelos lineares e baseados em distância** foram o escopo da matéria. Modelos de árvore (Random Forest, XGBoost) provavelmente capturariam interações que o SGD não vê.
- **Sem análise temporal** — a data de alojamento foi usada apenas para derivar a duração do ciclo, mas efeitos de sazonalidade (calor no verão prejudica a conversão) ficaram fora.
- **Variáveis pós-abate no modelo** — algumas features só são conhecidas ao final do ciclo, o que limita o uso preditivo *antes* do abate. Uma versão futura poderia restringir o modelo a variáveis disponíveis no dia do alojamento.

---

## 🛠️ Tecnologias

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg" height="40" alt="pandas" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="numpy" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original-wordmark.svg" height="40" alt="matplotlib" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" height="40" alt="jupyter" />
</div>

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` (KNeighborsRegressor, SGDRegressor, StandardScaler, cross_val_score)

---

## 👤 Autor

**Matheus Amorim** — [@MatheusAmorimm](https://github.com/MatheusAmorimm)

Projeto extensionista · Matéria de Negócios · Ciência de Dados e Inteligência Artificial

---

<p align="center">
  <sub>Os dados desta base são operacionais reais e foram fornecidos de forma anonimizada — produtores, técnicos, regiões e origens aparecem apenas como códigos (P147, Tecnico_1, Regiao_2, Origem_15).</sub>
</p>
