import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np

# Definindo transformações de pré-processamento
transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))])

# Carregando o conjunto de dados MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Definindo as classes
classes = tuple(str(i) for i in range(10))

# Função para mostrar imagens
def imshow(img):
    img = img / 2 + 0.5 # Desfazendo a normalização
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Obtendo alguns dados de treinamento para visualização
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Visualizando imagens de treinamento
imshow(torchvision.utils.make_grid(images))
print('Labels:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Definindo a arquitetura da rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(
        -
        1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Criando a rede neural
net = Net()

# Definindo a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Treinando a rede neural
for epoch in range(2): # Loop de treinamento em duas épocas    
    running_loss = 0.0

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 2000 == 1999: # Imprimir a cada 2000 lotes
        print(f'Época {epoch + 1}, Lote {i + 1}: perda {running_loss / 2000:.3f}')
        running_loss = 0.0
print('Treinamento concluído')

# Salvando o modelo treinado
PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

# Testando o modelo com dados de teste
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('Rótulos verdadeiros:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Carregando o modelo treinado
net = Net()
net.load_state_dict(torch.load(PATH))

# Fazendo previsões com o modelo
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Previsões:', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# Avaliando a acurácia do modelo em todo o conjunto de teste
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f'Acurácia da rede neural na MNIST: {100 * correct / total:.2f}%')