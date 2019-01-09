import argparse
import glob
import math
import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from utils import mfccs_and_spec

random.seed(12345)

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
        random.shuffle(self.audio_list)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_file = self.audio_list[idx]
        # label = audio_file.split('/')[-3]

        _, mel_db, _ = mfccs_and_spec(audio_file, wav_process=True)
        # return torch.Tensor(mel_db), label
        return torch.Tensor(mel_db), audio_file


def infer(model_path, audio_path, embedding_size, cuda, batch_size=1):
    inference_dataset = EmbedSet(audio_path=audio_path)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    embedder_net = SpeechEmbedder(embedding_size=embedding_size)
    if cuda:
        embedder_net.cuda()

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    # features_full, labels_full = np.empty((0,embedding_size), float), np.empty((0,), str)
    for batch_id, (mel_db_batch, audio_files_batch) in tqdm(enumerate(inference_loader), total=math.ceil(len(inference_dataset) / batch_size)):
        if cuda:
            mel_db_batch = mel_db_batch.cuda()
        mel_db_batch = Variable(mel_db_batch)
        # print('>>>>>>>>>>>>>>>> mel_db_batch shape', mel_db_batch.size())
        out = embedder_net(mel_db_batch)
        features = out.detach().cpu().numpy()
        audio_files = list(audio_files_batch)

        for idx in range(len(audio_files)):
            npz_path = audio_files[idx].replace('raw_data/', 'ge2e_npz_embed{}/'.format(embedding_size)).replace('.wav', '.npz')
            # skip existed
            if os.path.exists(npz_path):
                continue
            if not os.path.isdir(os.path.dirname(npz_path)):
                os.makedirs(os.path.dirname(npz_path))
            label = audio_files[idx].split('/')[-3]

            np.savez_compressed(npz_path, train_sequence=features[idx,:].reshape((1, -1)).astype(float), train_cluster_id=np.array([label], dtype=str))
        # print('>>>>>>>>>>>>>>> features', features)
        # print('>>>>>>>>>>>>>>> features.shape', features.shape)
        # print('>>>>>>>>>>>>>>>>>>>>>>>> labels', labels)

        # features_full = np.append(features_full, features, axis=0)
        # labels_full = np.append(labels_full, labels, axis=0)
    # return features_full, labels_full

def infer_pairs():
    raise Exception('Not supported yet!!')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
    parser.add_argument('--audio-path',
                        type=str,
                        default='/data5/xin/voxceleb/raw_data/test/',
                        help='path to dataset')
    # parser.add_argument('--npz-path',
    #                     type=str,
    #                     default='./xin.npz',
    #                     help='output path for npz file')
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
    parser.add_argument('--batch-size', type=int, default=256, metavar='BS',
                        help='input batch size for training (default: 256)')
    # TODO(xin): load embeddings size from checkpoint
    parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
                        help='Dimensionality of the embedding')
    parser.add_argument('--mode', type=str, default='dev', choices=['both', 'dev', 'test'],
                        help='Packing mode')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    return args


def main():
    args = parse_arguments()
    if args.mode in ['both', 'dev']:
        infer(model_path=args.checkpoint,
              audio_path=args.audio_path,
              embedding_size=args.embedding_size,
              cuda=args.cuda,
              batch_size=args.batch_size)

        # np.savez_compressed('tmp.npz', train_sequence=features_full, train_cluster_id=labels_full)
    if args.mode in ['both', 'test']:
        infer_pairs()

if __name__ == '__main__':
    main()
