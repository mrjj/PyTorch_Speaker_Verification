import argparse
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from utils import mfccs_and_spec


class EmbedSet(Dataset):
    def __init__(self, audio_path):
        self.path = audio_path

        self.audio_list = []
        for root, dirs, files in os.walk(audio_path):
            for file_name in files:
                if os.path.splitext(file_name)[-1] != '.wav':
                    continue
                file_path = os.path.join(root, file_name)
                self.audio_list.append(file_path)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_file = self.audio_list[idx]
        label = audio_file.split('/')[-3]

        _, mel_db, _ = mfccs_and_spec(audio_file, wav_process=True)
        return torch.Tensor(mel_db), label


def infer(model_path, audio_path, embedding_size, cuda, batch_size=1):
    inference_dataset = EmbedSet(audio_path=audio_path)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    embedder_net = SpeechEmbedder(embedding_size=embedding_size)
    if cuda:
        embedder_net.cuda()

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    
    for batch_id, (mel_db_batch, label_batch) in tqdm(enumerate(inference_loader)):
        if cuda:
            # label_batch = label_batch.cuda()
            mel_db_batch = mel_db_batch.cuda()
        # label_batch = Variable(label_batch)
        mel_db_batch = Variable(mel_db_batch)

        print('>>>>>>>>>>>>>>>> shape', mel_db_batch.size())
        out = embedder_net(mel_db_batch)
        features = out.detach().cpu().numpy()
        labels = np.array(list(label_batch))
        print('>>>>>>>>>>>>>>> features', features)
        print('>>>>>>>>>>>>>>> features.shape', features.shape)
        print('>>>>>>>>>>>>>>>>>>>>>>>> labels', labels)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
    parser.add_argument('--audio-path',
                        type=str,
                        default='/data5/xin/voxceleb/raw_data/test/id00017/01dfn2spqyE/',
                        help='path to dataset')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        metavar='PATH',
                        required=True,
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--batch-size', type=int, default=10, metavar='BS',
    #                     help='input batch size for training (default: 512)')
    parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
                        help='Dimensionality of the embedding')
    # parser.add_argument('--num-classes', type=int, default=5994, metavar='ES',
    #                     help='Number of classes')

    args = parser.parse_args()
    args.cuda = not args.no_cuda


    # TODO(xin): load embeddings size from checkpoint

    # # TODO(xin): Support batching
    # args.batch_size = 1

    return args


def main():
    args = parse_arguments()
    infer(model_path=args.checkpoint, audio_path=args.audio_path, embedding_size=args.embedding_size, cuda=args.cuda)


if __name__ == '__main__':
    main()
