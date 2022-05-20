import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
from dataset import DetectNoisyDataset
from model import AudioConvNet
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(  # каталог с данными
    '-i', '--input', required=True,
    help='Путь к данным'
)
parser.add_argument(  # режим
    '-m', '--mode', default='one',
    help='Режим работы модели'
)
args = vars(parser.parse_args())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = AudioConvNet()
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
criterion = torch.nn.MSELoss().to(device)


def test(test_loader, model, criterion):

    model.eval()
    true_pos = 0

    # Batches
    for (mel, target) in tqdm(test_loader):
        # Move to default device
        mel = mel.to(device)

        # Forward prop.
        predicted_mel = model(mel)

        # MSE
        mse = criterion(mel, predicted_mel)
        predict = 1 if mse > 0.034 else 0
        if predict == target:
            true_pos += 1

    acc = true_pos / len(test_loader)
    print(f'ACCURACY for test dataset: {acc:.4f}')


if __name__ == '__main__':

    data_folder = args['input']
    mode = args['mode']

    if mode != 'one':
        test_dataset = DetectNoisyDataset(data_folder, mode=mode)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  collate_fn=test_dataset.collate_fn)
        with torch.no_grad():
            test(test_loader, model, criterion)
    else:
        mel = torch.from_numpy(np.load(data_folder))
        add_mel = torch.zeros((int(mel.shape[0] / 50) + 1) * 50, 80)

        for i in range(mel.shape[0]):
            add_mel[i] = mel[i]

        add_mel = add_mel.reshape(-1, 50, 80).permute(0, 2, 1).to(device)
        with torch.no_grad():
            predict_mel = model(add_mel)
        mse = criterion(add_mel, predict_mel)
        predict = 'DETECTED NOISY' if mse > 0.034 else 'no noise'

        print('Результат обработки mel-спектрограммы : ', predict)
