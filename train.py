import torch
from torch.utils.data import DataLoader

from model import AudioConvNet
from dataset import AudioDataset
from tqdm import tqdm
import utils_gz


# Data parameters
data_folder = 'data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = None  # 'checkpoint_noisy.pth.tar'

epochs = 70
batch_size = 4
workers = 2
lr = 1e-3
decay_lr_to = 0.1
weight_decay = 5e-4

total_train_loss = []
total_valid_loss = []


def main():
    """
    Основная функция для тренировки модели
    """
    global epochs, checkpoint, total_train_loss, total_valid_loss

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = AudioConvNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        total_train_loss = checkpoint['train_loss']
        total_valid_loss = checkpoint['valid_loss']

    # Перемещение на рабочее устройство
    model = model.to(device)
    criterion = torch.nn.MSELoss().to(device)

    # Создаём Datasets и DataLoaders
    train_dataset = AudioDataset(data_folder)
    valid_dataset = AudioDataset(data_folder, mode='val')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               collate_fn=valid_dataset.collate_fn)

    decay_lr_at = [int(epochs * 0.7), int(epochs * 0.9)]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            utils_gz.adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Validation
        valid(valid_loader=valid_loader,
              model=model,
              criterion=criterion,
              epoch=epoch)

        # Save checkpoint
        if epoch % 2 == 0:
            filename = 'checkpoint_noisy.pth.tar'
            if epoch % 10 == 0:
                filename = f'checkpoint_noisy{epoch}.pth.tar'
            utils_gz.save_checkpoint(epoch, model, optimizer, filename, total_train_loss, total_valid_loss)
            print('Model saved')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Тренировка одной эпохи
    """
    global total_train_loss
    model.train()

    train_loss = 0.0

    # Batches
    for clean, noisy in tqdm(train_loader):

        # Move to default device
        clean = clean.to(device)  # (batch_size (N), 3, 300, 300)
        noisy = noisy.to(device)  # (batch_size (N), 3, 300, 300)

        # Forward prop.
        predicted_clean = model(noisy)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(clean, predicted_clean)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    total_train_loss.append(avg_train_loss)

    print(f'Epoch: {epoch}\t'
          f'Loss: {avg_train_loss:.4f}')

    del clean, noisy, predicted_clean


def valid(valid_loader, model, criterion, epoch=None):

    global total_valid_loss

    model.eval()
    valid_loss = 0.0

    # Batches
    for clean, noisy in tqdm(valid_loader):
        # Move to default device
        clean = clean.to(device)  # (batch_size (N), 3, 300, 300)
        noisy = noisy.to(device)  # (batch_size (N), 3, 300, 300)

        # Forward prop.
        predicted_clean = model(noisy)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(clean, predicted_clean)  # scalar
        valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    total_valid_loss.append(avg_valid_loss)

    if epoch is None:
        print(f'MSE for test dataset: {avg_valid_loss:.4f}')
    else:
        print(f'Epoch: {epoch}\t'
              f'Loss: {avg_valid_loss:.4f}')

    del clean, noisy, predicted_clean


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Прервано пользователем!')
