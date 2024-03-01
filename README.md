# Pytorch

Framework de código aberto para machine learning e deep learning. Ele fornece uma estrutura flexível e eficiente para a implementação de algoritmos de aprendizado de máquina e redes neurais. 

## Instalação

```bash
python -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt
python main.py
```

## Mudanças realizadas

A partir do código divulgado pelo professor, tivemos que realizar algumas alterações para que o mesmo
funcione corretamente. Percebemos que o matplot não estava instalado na máquina, por isso instalamos a
biblioteca segundo a documentação oficial.
Acrescentamos uma importação "import torch.nn.functional as F" pois "F" não tinha sido definido.

## MNIST
O conjunto de dados MNIST (Modified National Institute of Standards and Technology) é uma grande base de dados de dígitos manuscritos que é normalmente utilizada para treinar vários sistemas de processamento de imagem e modelos de aprendizagem automática.

Encontramos um repo no git [MNIST](https://github.com/mbornet-hl/MNIST/tree/master) que fornece algumas imagens para treinamento.

## Funcionamento do código
A lógica do código é iniciada importando as imagens previamente instaladas do MNIST.
![Carregando conjunto](/assets/carregandoConjuntos.png)

Com as imagens já carregadas, função "imshow" exibe uma tupla/conjunto de treinamento.
![Exibição](/assets/visualizando.png.png)