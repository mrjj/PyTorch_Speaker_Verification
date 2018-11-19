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
# from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
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

        # if hp.training:
        #     self.path = hp.data.train_path_unprocessed
        #     self.utterance_number = hp.train.M
        # else:
        #     self.path = hp.data.test_path_unprocessed
        #     self.utterance_number = hp.test.M
        # self.speakers = glob.glob(os.path.dirname(self.path))
        # self.speakers = glob.glob(self.path.split('*')[0] + '/*/')
        # shuffle(self.speakers)
        
    def __len__(self):
        return len(self.audio_list)
        # return len(self.speakers)

    def __getitem__(self, idx):
        audio_file = self.audio_list[idx]
        label = audio_file.split('/')[-3]

        _, mel_db, _ = mfccs_and_spec(audio_file, wav_process=True)
        return torch.Tensor(mel_db), label

        # speaker = self.speakers[idx]
        # # wav_files = glob.glob(speaker+'/*.WAV')
        # wav_files = glob.glob(speaker+'/*/*.wav')
        # shuffle(wav_files)
        # # wav_files = wav_files[0:self.utterance_number]
        
        # mel_dbs = []
        # for f in wav_files:
        #     _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
        #     mel_dbs.append(mel_db)
        # return torch.Tensor(mel_dbs)


def infer(model_path, audio_path, embedding_size, cuda, batch_size=1):
    
    # if hp.data.data_preprocessed:
    #     test_dataset = SpeakerDatasetTIMITPreprocessed()
    # else:
    #     test_dataset = SpeakerDatasetTIMIT()

    inference_dataset = EmbedSet(audio_path=audio_path)
    # test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    embedder_net = SpeechEmbedder(embedding_size=embedding_size)
    if cuda:
        embedder_net.cuda()

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    
    # batch_avg_EER = 0
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


        # assert hp.test.M % 2 == 0
        # enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
        
        # enrollment_batch = torch.reshape(enrollment_batch, (int(hp.test.N*hp.test.M/2), enrollment_batch.size(2), enrollment_batch.size(3)))
        # verification_batch = torch.reshape(verification_batch, (int(hp.test.N*hp.test.M/2), verification_batch.size(2), verification_batch.size(3)))
        
        # perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
        # unperm = list(perm)
        # for i,j in enumerate(perm):
        #     unperm[j] = i

    #     verification_batch = verification_batch[perm]
    #     enrollment_embeddings = embedder_net(enrollment_batch)
    #     verification_embeddings = embedder_net(verification_batch)
    #     verification_embeddings = verification_embeddings[unperm]
        
    #     enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, int(hp.test.M/2), enrollment_embeddings.size(1)))
    #     verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, int(hp.test.M/2), verification_embeddings.size(1)))
        
    #     enrollment_centroids = get_centroids(enrollment_embeddings)

    #     sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

    #     # calculating EER
    #     diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
        
    #     for thres in [0.01*i+0.5 for i in range(50)]:
    #         sim_matrix_thresh = sim_matrix>thres
            
    #         FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
    #         /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)

    #         FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
    #         /(float(hp.test.M/2))/hp.test.N)
            
    #         # Save threshold when FAR = FRR (=EER)
    #         if diff> abs(FAR-FRR):
    #             diff = abs(FAR-FRR)
    #             EER = (FAR+FRR)/2
    #             EER_thresh = thres
    #             EER_FAR = FAR
    #             EER_FRR = FRR
    #     batch_avg_EER += EER
    #     print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
    # avg_EER += batch_avg_EER/(batch_id+1)

    # avg_EER = avg_EER / hp.test.epochs
    # print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
    # return avg_EER

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

    # """ TODO(xin)
    # - Right now embedding_size is hardcoded in log_dir name
    # # LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    # #     .format(args.optimizer, args.n_triplets, args.lr, args.wd,
    # #             args.margin,args.embedding_size)
    # - Should move to model state_dict
    # """
    # args.embedding_size = int(os.path.dirname(args.checkpoint).split('-')[-3].split('embeddings')[-1].strip())

    # # TODO(xin): Support batching
    # args.batch_size = 1

    # # TODO(xin): Add num_classes to state_dict
    # args.num_classes = 5994

    return args


def main():
    args = parse_arguments()
    infer(model_path=args.checkpoint, audio_path=args.audio_path, embedding_size=args.embedding_size, cuda=args.cuda)


if __name__ == '__main__':
    main()
