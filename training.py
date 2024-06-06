import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from notebook.ViT import ViT 

def train_model(net, trainloader, criterion, optimizer, device, num_epochs):
    net.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  
            outputs = net(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            running_loss += loss.item()
            if i % 100 == 99: 
                print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print("Done Boi")
    torch.save(net.state_dict(), 'vit_model.pth')



def test_model(net, testloader, device):
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    image_size, in_channels = 32, 3
    patch_size = 4
    embed_dim = 192
    num_layers = 12
    mlp_dim = 768
    num_heads = 3
    drop_p = 0.5
    num_classes = 10

    net = ViT(image_size=image_size,
              in_channels=in_channels,
              patch_size=patch_size,
              num_layers=num_layers,
              embed_dim=embed_dim,
              mlp_dim=mlp_dim,
              num_heads=num_heads,
              drop_p=drop_p,
              num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    num_epochs = 10
    train_model(net, trainloader, criterion, optimizer, device, num_epochs)

    test_model(net, testloader, device)