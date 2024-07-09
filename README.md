# Rede Neural Feedforward com Backpropagation em Rust

## Visão Geral

Este projeto implementa uma rede neural feedforward com o algoritmo de backpropagation usando a linguagem de programação Rust. O projeto é um exemplo simples de como redes neurais podem ser construídas e treinadas para resolver problemas de aprendizado supervisionado, como classificação e regressão.

## Objetivos

O objetivo deste projeto é fornecer uma implementação didática de uma rede neural utilizando Rust, demonstrando conceitos importantes de aprendizado de máquina, como:

- Estrutura de uma rede neural (neurônios, camadas e conexões).
- Função de ativação sigmoidal e sua derivada.
- Forward pass: propagação dos dados de entrada através da rede.
- Backward pass: ajuste dos pesos e bias utilizando o algoritmo de backpropagation.
- Função de custo quadrático para avaliação do desempenho.

## Estrutura do Projeto

A rede neural é composta por várias camadas de neurônios. Cada neurônio possui um bias e um conjunto de pesos que conectam os neurônios de uma camada aos neurônios da próxima camada.

### Componentes Principais

1. **Neurônio (Neuron):** Estrutura que representa um neurônio, contendo um bias e uma lista de pesos.

2. **Camada (Layer):** Estrutura que representa uma camada de neurônios. Cada camada contém uma lista de neurônios.

3. **Função de Ativação Sigmoidal:** Função usada para introduzir não-linearidade na rede neural. A função sigmoidal é definida como:
   \[
   \text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
   \]
   e sua derivada é:
   \[
   \text{sigmoid\_prime}(z) = \text{sigmoid}(z) \times (1 - \text{sigmoid}(z))
   \]

4. **Forward Pass:** Processo de propagação dos dados de entrada através da rede para calcular as ativações dos neurônios em cada camada.

5. **Backward Pass:** Processo de ajuste dos pesos e bias dos neurônios utilizando o algoritmo de backpropagation. A função de custo utilizada é a função de custo quadrático:
   \[
   \text{cost\_derivative} = (\text{output\_activations} - y)
   \]

### Processo de Treinamento

1. **Inicialização:** Definição das camadas da rede neural com seus respectivos neurônios, pesos e bias.

2. **Forward Pass:** Propagação dos dados de entrada através da rede para calcular as ativações dos neurônios.

3. **Backward Pass:** Ajuste dos pesos e bias utilizando o algoritmo de backpropagation para minimizar a função de custo.

4. **Atualização dos Pesos e Bias:** Os pesos e bias são atualizados com base nos gradientes calculados durante o backward pass.

5. **Iteração:** O processo de forward pass e backward pass é repetido por um número definido de épocas para treinar a rede neural.

## Exemplo de Uso

Neste exemplo, a rede neural é treinada com um conjunto de dados de entrada e uma saída desejada (target). A estrutura da rede é definida com três camadas, onde a primeira e a segunda camadas são camadas ocultas, e a terceira camada é a camada de saída.

Ao final do treinamento, a saída da rede neural é impressa para avaliar o desempenho da rede no conjunto de dados fornecido.