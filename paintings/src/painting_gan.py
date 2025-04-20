import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

data_dir = "data"
print(os.listdir(data_dir))

epochs = 100
batch_size = 200
image_size = 64
learning_rate = 0.0002
sample_dir = "samples"
latent_size = 100


if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


transformed_data = transforms.Compose([
    transforms.Resize((image_size)),
    transforms.CenterCrop((image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])



dataset = datasets.ImageFolder(data_dir, transform=transformed_data)


dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

real_images= next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_images[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()




class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size,64 * 8,4,1,0,bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64 * 8, 64 * 4,4,2,1,bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64 * 4, 64 * 2,4,2,1,bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(64 * 2, 64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
            )
        
    def forward(self,input):
        return self.main(input)
            
            

G = Generator().to(device)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64 ,64 * 2, 4,2,1,bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64 * 2, 64 * 4, 4,2,1,bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64 * 4,64 * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2,inplace=True),
            
            
            nn.Conv2d(64 * 8, 1,4,1,0,bias=False),
            nn.Sigmoid()
            )
        
    def forward(self,input):
        return self.main(input)
    
D = Discriminator().to(device)






D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.999))


real_label = 1
fake_label = 0


def denorm(x):
    out = (x+1) / 2
    return out
    

criterion = nn.BCELoss()


img_list = []
G_losses = []
D_losses = []
iters = 0



for epoch in range(epochs):

    for i, data in enumerate(dataloader, 0):




        
        D.zero_grad()
  
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = D(real_cpu).view(-1)
  
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()


        noise = torch.randn(b_size, latent_size, 1, 1, device=device)

        fake = G(noise)
        label.fill_(fake_label)

        output = D(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        D_optimizer.step()


        G.zero_grad()
        label.fill_(real_label)  
       
        output = D(fake).view(-1)

        errG = criterion(output, label)

        errG.backward()
        D_G_z2 = output.mean().item()

        G_optimizer.step()


        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


        G_losses.append(errG.item())
        D_losses.append(errD.item())



        iters += 1
  
        
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
        
        
        
real_batch = next(iter(dataloader))


plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

fake_images = denorm(fake)


grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(grid.cpu(), (1, 2, 0))) 
plt.show()


fake_images = fake.reshape(fake.size(0),3, 64, 64)
save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
        
plt.imshow(fake_images[:10])
plt.show()
        
        
        
        
        
        










