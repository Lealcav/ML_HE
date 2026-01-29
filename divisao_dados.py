import tensorflow_datasets as tfds
import numpy as np

class DivisaoDados:
    def __init__(self):
        self.treino=None
        self.teste=None
        self.info=None

    def carregar_dados(self):
        (self.treino, self.teste), self.info = tfds.load(
                                                'mnist',
                                                split=['train', 'test'],
                                                shuffle_files=True,
                                                as_supervised=True,
                                                with_info=True,
                                            )

    # Convert to numpy arrays
    def tfds_to_numpy(self, dataset):
        images = []
        labels = []
        for image, label in tfds.as_numpy(dataset):
            images.append(image)
            labels.append(label)
        return np.array(images), np.array(labels) 

    def concatenar(self, treino, teste):
        return np.concatenate([treino, teste], axis=0)
    
    def shuffle_data(self, x, y):
        indices = np.random.permutation(len(x))
        
        return x[indices], y[indices]
    
    def split_data_target_shadow(self, dados):
        return dados[:30000], dados[30000:60000]
    
    def split_data_target(self, dados):
        split_idx = int(0.75 * len(dados))
        # Adicione ":" após o split_idx para retornar a fatia completa
        return dados[:split_idx], dados[split_idx:]