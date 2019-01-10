# import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CSV_PATH = '/data5/xin/voxceleb/ge2e_npz_embed256/test.csv'
# CSV_PATH = '/data5/xin/voxceleb/triplet_npz_embed512/test.csv'
TRAIN_SEQUENCE = 'train_sequence'
TRAIN_CLUSTER = 'train_cluster_id'

def load():
    with open(CSV_PATH, 'r') as f:
        records = pd.read_csv(CSV_PATH, names=['npz_path'])['npz_path'].values.tolist()

    # IMPORTANT: dont overwhelm the machine
    records = records[:2000]
    features, speakers = [], []
    for p in tqdm(records):
        data = np.load(p)
        features.append(data[TRAIN_SEQUENCE].flatten())
        speakers.append(data[TRAIN_CLUSTER])
    features = np.array(features)
    speakers = np.array(speakers)

    return features, speakers

def pca_preprocess(features):
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    return features_reduced

def plot(df):
    # sns.set_context("notebook", font_scale=1.1)
    # sns.set_style("ticks")

    # Create scatterplot of dataframe
    sns.lmplot(x='dim1',
            y='dim2',
            data=df,
            fit_reg=False,
            legend=True,
            size=9,
            hue='speaker')
            # scatter_kws={"s":200, "alpha":0.3})

    plt.title('TSNE', weight='bold').set_fontsize('14')
    plt.xlabel('Prin Comp 1', weight='bold').set_fontsize('10')
    plt.ylabel('Prin Comp 2', weight='bold').set_fontsize('10')
    plt.savefig('tmp.png')

def tsne(features_reduced, speakers):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features_reduced)
    print(type(tsne_results))

    df = pd.DataFrame(data=tsne_results, columns=['dim1', 'dim2'])
    df['speaker'] = speakers
    plot(df)

def main():
    features, speakers = load()
    print('==> Finish loading')
    # features_reduced = pca_preprocess(features)
    features_reduced = features
    print('==> Finish pca')
    tsne(features_reduced, speakers)
    print('==> Finish tsne')


if __name__ == '__main__':
    main()
