import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os

# training loop that can be used for classifiers
# print the seed value
seed = torch.initial_seed()
print('Used seed : {}'.format(seed))

# create model
model = ModelClass(classes='NUM_CLASSES').to('device')

print(model)
print('Model created')

# create dataset and data loader
dataset = datasets.ImageFolder('TRAIN_IMG_DIR', transforms.Compose([
    transforms.CenterCrop('IMAGE_DIM'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
print('Dataset created')

dataloader = data.DataLoader(
    dataset,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
    batch_size='BATCH_SIZE')
print('Dataloader created')

# create optimizer
# the one that WORKS
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

print('Optimizer created')

# multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print('LR Scheduler created')

# start training!!
print('Starting training...')
total_steps = 1
for epoch in range('NUM_EPOCHS'):
    lr_scheduler.step()
    for imgs, classes in dataloader:
        imgs, classes = imgs.to('device'), classes.to('device')

        # calculate the loss
        output = model(imgs)
        loss = F.cross_entropy(output, classes)

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the information and add to tensorboard
        if total_steps % 10 == 0:
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                    .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                
        # print out gradient values and parameter average values
        if total_steps % 100 == 0:
            with torch.no_grad():
                # print and save the grad of the parameters
                # also print and save parameter values
                print('*' * 10)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print('\t{} - grad_avg: {}'.format(name, avg_grad))
                        
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print('\t{} - param_avg: {}'.format(name, avg_weight))
                        
        total_steps += 1

    # save checkpoints
    checkpoint_path = os.path.join('CHECKPOINT_DIR', 'model_states_e{}.pkl'.format(epoch + 1))
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'seed': seed,
    }
    torch.save(state, checkpoint_path)
