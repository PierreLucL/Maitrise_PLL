import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class PostersNet(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel simple
    permettant de classifier les posters de films.
    """

    def __init__(self):
        super().__init__()
        #300
        self.C1 = nn.Conv2d(3, 9, kernel_size=5 )
        #296
        self.C1_2 = nn.Conv2d(9, 9, kernel_size=5 )
        #292
        self.P2 = nn.MaxPool2d(2,stride=2)
        #146
        self.S3 = nn.BatchNorm2d(9)
        self.C4 = nn.Conv2d(9, 12, kernel_size=5 )
        #142
        self.P5 = nn.MaxPool2d(2,stride=2)
        #71
        self.S6 = nn.BatchNorm2d(12)
        self.C7 = nn.Conv2d(12, 12, kernel_size=3 )
        #69
        self.S8 = nn.BatchNorm2d(12)
        self.C9 = nn.Conv2d(12, 12, kernel_size=3 )
        #67
        self.S10 = nn.BatchNorm2d(12)
        self.C11 = nn.Conv2d(12, 12, kernel_size=3 )
        #65
        self.S12 = nn.BatchNorm2d(12)
        self.P13 = nn.MaxPool2d(2, stride=2)
        #32
        self.C14 = nn.Conv2d(12,12,kernel_size=3 )
        #30
        self.S14 = nn.BatchNorm2d(12)
        self.C15 = nn.Conv2d(12,12,kernel_size=3 )
        #28
        self.S15 = nn.BatchNorm2d(12)
        self.C16 = nn.Conv2d(12,12,kernel_size=3 )
        #26
        self.S16 = nn.BatchNorm2d(12)
        self.C17 = nn.Conv2d(12,12,kernel_size=3 )
        #24
        self.S18 = nn.BatchNorm2d(12)
        self.P18 = nn.AvgPool2d(2, stride=2)
        #12
        self.C19 = nn.Conv2d(12,12,kernel_size=3 )
        #10
        self.S19 = nn.BatchNorm2d(12)
        self.C20 = nn.Conv2d(12,12,kernel_size=3 )
        #8
        self.S20 = nn.BatchNorm2d(12)
        self.C21 = nn.Conv2d(12,12,kernel_size=3 )
        self.S21 = nn.BatchNorm2d(12)
        #6
        self.C22 = nn.Conv2d(12,12,kernel_size=3 )
        self.S22 = nn.BatchNorm2d(12)
        #4
        self.C23 = nn.Conv2d(12,12,kernel_size=3 )
        #2
        self.S24 = nn.BatchNorm2d(12)
        self.P25 = nn.AvgPool2d(2, stride=2)
        #1
        self.L26 = nn.Linear(12, 25)
        self.output = nn.Sigmoid()


    def forward(self, x):
        y = F.relu(self.C1(x))
        y = F.relu(self.S3(self.C1_2(y)))
        y = self.P2(y)
        y = F.relu(self.S6(self.C4(y)))
        y = self.P5(y)
        y = F.relu(self.S8(self.C7(y)))
        y = F.relu(self.S10(self.C9(y)))
        y = F.relu(self.S12(self.C11(y)))
        y = self.P13(y)
        y = F.relu(self.S14(self.C14(y)))
        y = F.relu(self.S15(self.C15(y)))
        y = F.relu(self.S16(self.C16(y)))
        y = F.relu(self.S18(self.C17(y)))
        y = self.P18(y)
        y = F.relu(self.S19(self.C19(y)))
        y = F.relu(self.S20(self.C20(y)))
        y = F.relu(self.S21(self.C21(y)))
        y = F.relu(self.S22(self.C22(y)))
        y = F.relu(self.S24(self.C23(y)))
        y=  self.P25(y)
        y = y.view(-1,12)
        y = self.L26(y)
        return self.output(y)

class PostersNet2(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel simple
    permettant de classifier les posters de films.
    """

    def __init__(self, largeur):
        super().__init__()
        self.largeur=largeur
        #300
        self.C1 = nn.Conv2d(3, 30, kernel_size=5, stride=2 )
        self.S2 = nn.BatchNorm2d(30)
        #148
        self.C3 = nn.Conv2d(30, self.largeur, kernel_size=5, stride=2 )
        self.S4 = nn.BatchNorm2d(self.largeur)
        #72
        self.P5 = nn.MaxPool2d(3,stride=3)
        #24
        self.C6 = nn.Conv2d(self.largeur, self.largeur, kernel_size=5, stride=2, padding=(2,2))
        self.S7 = nn.BatchNorm2d(self.largeur)
        #12
        self.P8 = nn.MaxPool2d(2,stride=2)
        #6
        self.C9 = nn.Conv2d(self.largeur, self.largeur, kernel_size=3)
        self.S10 = nn.BatchNorm2d(self.largeur)
        #4
        self.C11 = nn.Conv2d(self.largeur, self.largeur, kernel_size=3)
        self.S12 = nn.BatchNorm2d(self.largeur)
        #2
        self.P13 = nn.AvgPool2d(2)
        #1
        self.L14 = nn.Linear(self.largeur, 25)
        self.output = nn.Sigmoid()


    def forward(self, x):
        y = F.relu(self.S2(self.C1(x)))
        y = F.relu(self.S4(self.C3(y)))
        y = self.P5(y)
        y = F.relu(self.S7(self.C6(y)))
        y = self.P8(y)
        y = F.relu(self.S10(self.C9(y)))
        y = F.relu(self.S12(self.C11(y)))
        y = self.P13(y)
        y = y.view(-1,self.largeur)
        y = self.L14(y)
        print(y.shape)
        return self.output(y)

class AlexNet(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel simple
    permettant de classifier les posters de films.
    """

    def __init__(self):
        super().__init__()
        # 300
        self.C1 = nn.Conv2d(3, 96, kernel_size=(11,11), stride=(5,5))
        self.S2 = nn.BatchNorm2d(96)
        # 58
        self.P3 = nn.MaxPool2d(3, stride=2)
        # 28
        self.C4 = nn.Conv2d(96, 256, kernel_size=(5,5), padding=2)
        self.S5 = nn.BatchNorm2d(256)
        # 28
        self.P6 = nn.MaxPool2d(2,stride=2)
        # 14
        self.C7 = nn.Conv2d(256, 384, kernel_size=(3,3))
        self.S8 = nn.BatchNorm2d(384)
        # 12
        self.C9 = nn.Conv2d(384, 384, kernel_size=(3, 3),padding=1)
        self.S10 = nn.BatchNorm2d(384)
        # 12
        self.C11 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.S12 = nn.BatchNorm2d(256)
        # 12
        self.P13 = nn.MaxPool2d(3,stride=2)
        # 5
        self.L14 = nn.Linear(6400, 4096)
        # 6400
        self.dropout = nn.Dropout(p=0.5)
        # 4096
        self.L15 = nn.Linear(4096, 4096)
        self.L16 = nn.Linear(4096, 25)
        # 25
        self.output = nn.Sigmoid()


    def forward(self, x):
        y = F.relu(self.S2(self.C1(x)))
        y = self.P3(y)
        y = F.relu(self.S5(self.C4(y)))
        y = self.P6(y)
        y = F.relu(self.S8(self.C7(y)))
        y = F.relu(self.S10(self.C9(y)))
        y = F.relu(self.S12(self.C11(y)))
        y = self.P13(y)
        y = y.view(-1,6400)
        y = F.relu(self.L14(y))
        y = self.dropout(y)
        y = F.relu(self.L15(y))
        y = self.dropout(y)
        y = self.L16(y)
        return self.output(y)


class ResNet18(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        # On crée le réseau de neurone pré-entraîné
        self.model = resnet18(pretrained=pretrained, progress=False)

        # On récupère le nombre de neurones avant la couche de classement
        dim_before_fc = self.model.fc.in_features

        # Changer la dernière couche pleinement connecté pour avoir le bon
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.output = nn.Sigmoid()
        self.model.fc = nn.Linear(dim_before_fc, 25)

    def forward(self, x):
        # Appeler la fonction forward du réseau préentraîné (resnet18) de LegoNet
        x = self.model.forward(x)
        x = self.output(x)
        return x