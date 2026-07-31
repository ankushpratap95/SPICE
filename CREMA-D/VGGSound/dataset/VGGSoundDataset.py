import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class VGGSound(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.use_pre_frame = 1
        train_video_data = []
        train_audio_data = []
        test_video_data  = []
        test_audio_data  = []
        train_label = []
        test_label  = []
        train_class = []
        test_class  = []

        with open('./data/vggsound.csv') as f:
            csv_reader = csv.reader(f)

            for item in csv_reader:
                if item[3] == 'train':
                    video_dir = os.path.join('./data/video/', 'train-set-img', 'Image-{:02d}-FPS'.format(self.args.fps), item[0]+ '_{:06d}'.format(int(item[1])) + '.mp4')
                    audio_dir = os.path.join('./data/video/', 'train-audios/train-set', item[0] + '_{:06d}'.format(int(item[1]))+'.wav')
                   
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: train_class.append(item[2])
                        train_label.append(item[2])

                if item[3] == 'test':
                    video_dir = os.path.join('./data/video/', 'test-set-img', 'Image-{:02d}-FPS'.format(self.args.fps), item[0]+ '_{:06d}'.format(int(item[1])) + '.mp4')
                    audio_dir = os.path.join('./data/video/', 'test-audios/test-set', item[0] + '_{:06d}'.format(int(item[1]))+'.wav')
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: test_class.append(item[2])
                        test_label.append(item[2])

        assert len(train_class) == len(test_class)
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]


    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=35400, mono=True)

        # new_sample = np.tile(sample, 20)[:22050 * 20]

        if len(sample) == 0:
            sample = np.array([0])
        while len(sample) / rate < 20.:
            sample = np.tile(sample, 2)
        start_point = 0
        new_sample = sample[start_point:start_point + rate * 20]

        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.
        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.video[idx])
        if self.mode == 'train':
            select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
            select_index.sort()
        else:
            select_index = [idx for idx in range(0, len(image_samples), len(image_samples) // self.use_pre_frame)]
        images = torch.zeros((self.use_pre_frame, 3, 384, 384))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img
        images = images.permute((1, 0, 2, 3))
        # label

        label = self.label[idx]
        
        return spectrogram, images, label


