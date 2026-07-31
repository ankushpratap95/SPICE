import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torch
import numpy as np
import librosa
from torchvision import transforms
import glob


class VADataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.use_pre_frame = 3

        train_video_data, train_audio_data, train_label, train_class = [], [], [], []
        test_video_data, test_audio_data, test_label, test_class = [], [], [], []
        weight_video_data, weight_audio_data, weight_label, weight_class = [], [], [], []
        root = args.data_root

        train_file = os.path.join(root, 'annotations', 'train.csv')
        data = pd.read_csv(train_file)
        self.labels = data['label']
        self.files = data['youtube_id']
        self.label_map = dict(zip(data['youtube_id'], data['label']))

        train_img_root = os.path.join(root, 'train_img', 'Image-01-FPS')
        train_wav_root = os.path.join(root, 'train_wav')

        for folder in os.listdir(train_img_root):
            video_dir = os.path.join(train_img_root, folder)


            youtube_id = folder.rsplit('_', 2)[0]

            audio_path = os.path.join(train_wav_root, folder + '.wav')

            if not os.path.exists(audio_path):
                audio_matches = glob.glob(os.path.join(train_wav_root, f"{youtube_id}*"))
                if len(audio_matches) == 0:
                    continue
                audio_path = audio_matches[0]

            if len(os.listdir(video_dir)) > 3:
                train_video_data.append(video_dir)
                train_audio_data.append(audio_path)

                label = self.label_map[youtube_id]

                if label not in train_class:
                    train_class.append(label)

                train_label.append(label)

        
        test_file = os.path.join(root, 'annotations', 'test.csv')
        data = pd.read_csv(test_file)
        self.labels = data['label']
        self.files = data['youtube_id']
        self.label_map = dict(zip(data['youtube_id'], data['label']))

        test_img_root = os.path.join(root, 'test_img', 'Image-01-FPS')
        test_wav_root = os.path.join(root, 'test_wav')

        for folder in os.listdir(test_img_root):
            video_dir = os.path.join(test_img_root, folder)

            youtube_id = folder.rsplit('_', 2)[0]

            audio_path = os.path.join(test_wav_root, folder + '.wav')

            if not os.path.exists(audio_path):
                audio_matches = glob.glob(os.path.join(test_wav_root, f"{youtube_id}*"))
                if len(audio_matches) == 0:
                    continue
                audio_path = audio_matches[0]

            if len(os.listdir(video_dir)) > 3:
                test_video_data.append(video_dir)
                test_audio_data.append(audio_path)

                label = self.label_map[youtube_id]

                if label not in test_class:
                    test_class.append(label)

                test_label.append(label)


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
        if len(sample) == 0:
            sample = np.array([0])
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)
            
        start_point = 0
        new_sample = sample[start_point:start_point + rate * 10]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.
        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
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
            
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = images.permute((1, 0, 2, 3))
        

        label = self.label[idx]
        return spectrogram, images, label
