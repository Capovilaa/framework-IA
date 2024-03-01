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
![Exibição](/assets/visualizando.png)

## Definindo arquitetura
Definimos uma rede neural convolucional (CNN) em PyTorch para classificar dígitos do conjunto MNIST. A arquitetura inclui duas camadas convolucionais, duas camadas totalmente conectadas e utiliza funções de ativação ReLU. Após a definição, a rede é instanciada e pronta para treinamento e previsão.
![Net](/assets/net.png)

## Treinamento da rede neural
Após instanciar a classe, o treinamento é iniciado. Esse que enumerado por 2 índices, calculando a perda (como o modelo está performando a tarefa).
Após esse treinamento é criado um arquivo à parte do tipo pth.
![Treinando](/assets/treinando.png)

## Realiza os teste
Com todos os parâmetros prontos, nessa etapa que os testes são iniciados, ele exibe com a ajuda do matplot alguns números que ele irá tentar reconhecer. As previsões são printadas.
![Teste](/assets/testando.png)

## Mostra acurácia
Após cada previsão, o código printa também uma porcentagem de certeza daquela previsão. Isso ajuda para o usuário ter uma noção da propabilidade daquilo estar correto.
![Previ](/assets/previ.png)