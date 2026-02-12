# 🤖 Trabalho AV1 - Classificação e Regressão com Numpy

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Numpy](https://img.shields.io/badge/Numpy-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

Implementação de modelos de **Classificação** e **Regressão** utilizando apenas **Numpy** (sem bibliotecas de Machine Learning), desenvolvido como parte da AV1 de Inteligência Artificial Computacional.

## 📋 Sobre o Projeto

Este projeto implementa do zero algoritmos clássicos de Machine Learning, demonstrando profundo entendimento dos fundamentos matemáticos e estatísticos por trás dos modelos, sem depender de bibliotecas como Scikit-learn.

### 🎯 Objetivos

- ✅ Implementar algoritmos de ML usando apenas Numpy
- ✅ Compreender a matemática por trás dos modelos
- ✅ Comparar diferentes abordagens de classificação e regressão
- ✅ Validar modelos usando validação Monte Carlo
- ✅ Análise de desempenho e métricas

## 🧮 Modelos Implementados

### 📊 Regressão
1. **MQO (Mínimos Quadrados Ordinários)**
   - Regressão linear clássica
   - Solução analítica via álgebra linear
   - Estimativa de parâmetros β

### 🏷️ Classificação
1. **Naive Bayes**
   - Classificador probabilístico baseado no Teorema de Bayes
   - Assunção de independência entre features
   
2. **Gauss Tradicional**
   - Classificador baseado em distribuição gaussiana
   - Estimativa de máxima verossimilhança (MLE)
   
3. **Gauss Regularizado**
   - Versão melhorada com regularização da matriz de covariância
   - Previne problemas de singularidade
   - Matriz de identidade com fator de regularização (1e-8)

## 🗂️ Estrutura do Projeto

```
Trabalho-AV1-Classificacao-Regressao/
│
├── classificacao_numpy.py          # Implementação dos classificadores
├── regressao_numpy.py              # Implementação de regressão
├── EMGsDataset.csv                 # Dataset de sinais EMG
├── aerogenerador.dat               # Dataset de aerogerador
├── Relatorio_IA_AV1_FINAL.pdf     # Relatório técnico completo
└── README.md                       # Documentação
```

## 📊 Datasets

### 1. EMGsDataset.csv
- **Descrição**: Sinais de eletromiografia (EMG)
- **Uso**: Classificação de padrões
- **Features**: Múltiplos canais de sinais EMG
- **Classes**: Diferentes gestos/movimentos

### 2. aerogenerador.dat
- **Descrição**: Dados de aerogeradores
- **Uso**: Regressão para predição
- **Features**: Variáveis físicas do sistema
- **Target**: Variável de saída a ser predita

## 🚀 Como Executar

### Pré-requisitos

```bash
Python 3.13 ou superior
Numpy
Matplotlib (para visualizações)
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/LucNath/Trabalho-AV1-Classificacao-Regressao-Inteligencia-Artificial.git
cd Trabalho-AV1-Classificacao-Regressao-Inteligencia-Artificial

# Instale as dependências
pip install numpy matplotlib
```

### Executando Classificação

```python
# Execute o script de classificação
python classificacao_numpy.py
```

### Executando Regressão

```python
# Execute o script de regressão
python regressao_numpy.py
```

## 📈 Métricas e Avaliação

### Classificação
- **Acurácia**: Percentual de predições corretas
- **Validação Monte Carlo**: R = 500 iterações
- **Matriz de Confusão**: Análise de erros por classe

### Regressão
- **MSE (Mean Squared Error)**: Erro quadrático médio
- **R² (Coeficiente de Determinação)**: Qualidade do ajuste
- **Validação Monte Carlo**: R = 500 iterações

## 🧪 Metodologia

### Validação Monte Carlo
Implementada para garantir robustez dos resultados:
1. **500 iterações** de treinamento/teste
2. **Split aleatório** em cada iteração
3. **Média das métricas** para resultado final
4. **Desvio padrão** para análise de estabilidade

### Tratamento de Dados
- Normalização de features quando necessário
- Adição de intercepto (bias term)
- Tratamento de valores faltantes
- Split treino/teste aleatório

## 🔬 Fundamentos Matemáticos

### Mínimos Quadrados Ordinários (MQO)

```
β = (X^T X)^(-1) X^T y
```

Onde:
- `β`: Vetor de parâmetros
- `X`: Matriz de features
- `y`: Vetor target

### Naive Bayes

```
P(C|X) = P(X|C) * P(C) / P(X)
```

Classificação por máxima probabilidade a posteriori (MAP).

### Distribuição Gaussiana

```
P(x|C) = (1/√(2π|Σ|)) * exp(-½(x-μ)^T Σ^(-1) (x-μ))
```

Onde:
- `μ`: Média da classe
- `Σ`: Matriz de covariância
- `Σ_reg = Σ + λI`: Covariância regularizada

## 📊 Resultados

### Classificação (EMGsDataset)
| Modelo | Acurácia | Tempo |
|--------|----------|-------|
| Naive Bayes | -% | - ms |
| Gauss Tradicional | -% | - ms |
| Gauss Regularizado | -% | - ms |

### Regressão (Aerogerador)
| Métrica | Valor |
|---------|-------|
| MSE | - |
| R² | - |
| RMSE | - |

*Nota: Execute os scripts para obter os resultados atualizados*

## 🛠️ Tecnologias Utilizadas

- **Python 3.13** - Linguagem principal
- **Numpy** - Operações matriciais e vetoriais
- **Matplotlib** - Visualização de resultados
- **Pandas** (opcional) - Leitura de dados CSV

## 💡 Conceitos Aplicados

### Álgebra Linear
- Multiplicação de matrizes
- Inversão de matrizes
- Decomposição de autovalores
- Determinantes

### Estatística
- Estimação de máxima verossimilhança
- Distribuições de probabilidade
- Teorema de Bayes
- Correlação e covariância

### Machine Learning
- Aprendizado supervisionado
- Classificação multiclasse
- Regressão linear
- Validação cruzada
- Overfitting e regularização

## 🎓 Aprendizados

Este projeto proporcionou:
- ✅ Compreensão profunda dos algoritmos
- ✅ Domínio de operações com Numpy
- ✅ Implementação sem bibliotecas prontas
- ✅ Análise crítica de resultados
- ✅ Debugging de implementações matemáticas

## 📝 Relatório Técnico

O arquivo `Relatorio_IA_AV1_FINAL.pdf` contém:
- Fundamentação teórica completa
- Descrição detalhada dos algoritmos
- Análise de resultados
- Gráficos e visualizações
- Conclusões e discussões

## 🔍 Possíveis Melhorias

- [ ] Implementar validação cruzada k-fold
- [ ] Adicionar mais modelos (SVM, KNN)
- [ ] Grid search para hiperparâmetros
- [ ] Visualização interativa dos resultados
- [ ] Análise de feature importance
- [ ] Pipeline de pré-processamento

## 🤝 Contribuindo

Este é um projeto acadêmico, mas sugestões são bem-vindas!

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/melhoria`)
3. Commit suas mudanças (`git commit -m 'Adicionar melhoria'`)
4. Push para a branch (`git push origin feature/melhoria`)
5. Abra um Pull Request

## 👨‍💻 Autor

**Lucas Nathan**

- GitHub: [@LucNath](https://github.com/LucNath)
- LinkedIn: [Lucas Nathan][https://linkedin.com/in/-](https://www.linkedin.com/in/lucas-nathan-de-moraes-gomes-a83418242/)

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- **Professor André** - Orientação e conhecimento transmitido
- **UNIFOR** - Estrutura e apoio
- **Comunidade Python/Numpy** - Documentação excelente

---

<div align="center">

### 📚 Desenvolvido como parte da AV1 de Inteligência Artificial Computacional

**UNIFOR - Universidade de Fortaleza**

⭐ Se este projeto foi útil para você, considere dar uma estrela!

</div>

---

**Última atualização:** Fevereiro 2026
