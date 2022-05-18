import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
from dataset import AudioDataset
from train import valid
from model import AudioConvNet


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


if __name__ == '__main__':

    data_folder = args['input']
    mode = args['mode']

    if mode != 'one':
        criterion = torch.nn.MSELoss().to(device)
        test_dataset = AudioDataset(data_folder, mode=mode)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=4,
                                                  collate_fn=test_dataset.collate_fn)
        valid(test_loader, model, criterion)
    else:
        audio = torch.from_numpy(np.load(data_folder))
        audio_len = audio.shape[0]
        add_audio = torch.zeros((int(audio.shape[0] / 50) + 1) * 50, 80)

        for i in range(audio.shape[0]):
            add_audio[i] = audio[i]

        add_audio = add_audio.reshape(-1, 50, 80).permute(0, 2, 1)
        clean_audio = model(add_audio)
        clean_audio = clean_audio.permute(0, 2, 1).reshape(-1, 80)
        clean_audio = clean_audio[:audio_len].detach().numpy()
        np.save(r'output\cleaned_mel.npy', clean_audio)
        print(r'Сохранено: output\cleaned_mel.npy')
