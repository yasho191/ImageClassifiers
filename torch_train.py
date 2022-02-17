import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch_models import Models

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of the model. Must be one of: 1. AlexNet 2. DenseNet 3. InceptionV3 4. ResNet 5. VGG", type=str)
parser.add_argument("classes", help="Number of classes", type=int)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=16)

args = parser.parse_args()

MODEL_NAME = args.model
CLASSES = args.classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

if MODEL_NAME not in ["AlexNet", "DenseNet", "InceptionV3", "ResNet", "VGG"]:
    print(f"Invalid argument for model: {MODEL_NAME}")
    exit(-1)

# training loop that can be used for classifiers
# print the seed value
seed = torch.initial_seed()
print('Used seed : {}'.format(seed))

model = Models(MODEL_NAME, CLASSES)
# if you have a cuda enabled gpu use
model = model.ret_model()
if torch.cuda.is_available():
    model = model.to(device=0)

print(model)
print('Model created')

# create training dataset and data loader
train_dataset = datasets.ImageFolder('data/train', transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
print('Training Dataset created')

train_dataloader = data.DataLoader(
    train_dataset,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True,
    batch_size=BATCH_SIZE)
print('Training Dataloader created')

# create validation dataset and data loader
validation_dataset = datasets.ImageFolder('data/validate', transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
print('Validation Dataset created')

validation_dataloader = data.DataLoader(
    validation_dataset,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True,
    batch_size=BATCH_SIZE)
print('Validation Dataloader created')

# create optimizer
# the one that WORKS
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
print('Optimizer created')

# multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print('LR Scheduler created')

total_steps = 1
last_epoch=0

# if checkpoints exists: Load Model from ckpt
try:
    ckpts = os.listdir(f'ckpt/{MODEL_NAME}')
    ckpts = [int(i[i.rfind('e')+1:-4]) for i in ckpts if i.endswith('.pkl')]
    E = max(ckpts)
    checkpoint = torch.load(f'ckpt/{MODEL_NAME}/{MODEL_NAME}_states_e{E}.pkl')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    total_steps = checkpoint['total_steps']
    print(f'Checkpoint found: Training restored from EPOCH: {E}')
except:
    print('No previous models found in ckpt directory')

# start training!!
print('Starting training...')
for epoch in range(last_epoch, EPOCHS):
    lr_scheduler.step()
    model.train()
    train_acc = 0
    train_loss = 0
    for imgs, classes in train_dataloader:
        # for GPU: 
        if torch.cuda.is_available():
            imgs, classes = imgs.to(device=0), classes.to(device=0)

        # calculate the loss
        output = model(imgs)
        loss = F.cross_entropy(output, classes)

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training loss and Accuracy
        train_loss += loss.item()
        with torch.no_grad():
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == classes)
            train_acc += accuracy.item() / classes.size(0)
        
        total_steps += 1

    # validation
    val_loss = 0
    val_acc = 0
    model.eval()     # Optional when not using Model Specific layer
    for imgs, classes in validation_dataloader:
        if torch.cuda.is_available():
            imgs, classes = output.to(device=0), classes.to(device=0)
        
        output = model(imgs)
        loss = F.cross_entropy(output, classes)
        val_loss += loss.item()

        _, preds = torch.max(output, 1)
        accuracy = torch.sum(preds == classes)
        val_acc += accuracy.item() / classes.size(0)

    print(f'Epoch {epoch+1} \tSteps: {total_steps} \tLoss: {train_loss/len(train_dataloader)} \tAcc: {train_acc/len(train_dataloader)} \tValidation Loss: {val_loss / len(validation_dataloader)} \tVal Acc: {val_acc / len(validation_dataloader)}')
               
    # save checkpoints after every 5 epochs
    if epoch % 5 == 0:
        CHECK_FOLDER = os.path.isdir(f'ckpt/{MODEL_NAME}')
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.mkdir(f'ckpt/{MODEL_NAME}')
        checkpoint_path = os.path.join('ckpt', f'{MODEL_NAME}', f'{MODEL_NAME}_states_e{epoch + 1}.pkl')
        state = {
            'epoch': epoch+1,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)
        print(f'Checkpoint Saved at EPOCH: {epoch+1}')