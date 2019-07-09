import torch
import torch.nn as nn
import torch.nn.functional as F
NOISE_DIM = 10

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim + 2 + 3, 256*4*4)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 512)
#         self.fc4 = nn.Linear(512, 20736)
        
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
#         self.conv1_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.dropout = nn.Dropout(p=0.1)
        self.up = nn.Upsample(scale_factor=2)
        self.max = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

#         self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
#         self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv6 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv7 = nn.Conv2d(8, 1, 3)
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        x = F.leaky_relu(self.fc1(
            torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        ))
        EnergyDeposit = x.view(-1, 256, 4, 4)
        EnergyDeposit = self.up(F.leaky_relu(self.conv1(EnergyDeposit)))
#         EnergyDeposit = self.up(self.dropout(F.leaky_relu(self.conv1_2(EnergyDeposit))))
        
        EnergyDeposit = F.leaky_relu(self.conv2(EnergyDeposit))
        EnergyDeposit = self.up(self.dropout(F.leaky_relu(self.conv2_2(EnergyDeposit))))

#         EnergyDeposit = F.leaky_relu(self.conv3(EnergyDeposit))
#         EnergyDeposit = self.up(self.dropout(F.leaky_relu(self.conv3_2(EnergyDeposit))))
        
        EnergyDeposit = self.up(F.leaky_relu(self.conv4(EnergyDeposit)))
        EnergyDeposit = self.up(self.dropout(F.leaky_relu(self.conv4_2(EnergyDeposit))))
        
        EnergyDeposit = F.leaky_relu(self.conv5(EnergyDeposit))
        EnergyDeposit = self.max(self.dropout(F.leaky_relu(self.conv5_2(EnergyDeposit))))
        
        EnergyDeposit = self.dropout(F.leaky_relu(self.conv6(EnergyDeposit)))
        EnergyDeposit = self.conv7(EnergyDeposit)

        return EnergyDeposit