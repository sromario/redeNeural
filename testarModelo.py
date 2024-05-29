import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carregar o modelo
model = tf.keras.models.load_model('modelo.h5')

# Parâmetros de pré-processamento de imagens
image_weight = 160
image_height = 160
image_size = (image_weight, image_height)

# Parâmetros do treinamento
batch_size = 32

# Classes de saída
class_name = ['cat', 'dog']

# Diretório base
dataset_dir = os.path.join(os.getcwd(), 'dataset/split_dataset')


dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_train_cats_len = len(os.listdir(os.path.join(dataset_train_dir,'cat')))
dataset_train_dogs_len = len(os.listdir(os.path.join(dataset_train_dir,'dog')))

# Contar o número de imagens de validação em cada classe
dataset_validation_dir = os.path.join(dataset_dir, 'validation')
dataset_validation_cats_len = len(os.listdir(os.path.join(dataset_validation_dir, 'cat')))
dataset_validation_dogs_len = len(os.listdir(os.path.join(dataset_validation_dir, 'dog')))





# Pré-processar o conjunto de validação
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Pré-processar novo train com parâmetros
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5

dataset_teste = dataset_validation.take(dataset_validation_batches)

def plot_dataset_predictions(dataset):
    for features, labels in dataset.take(1):
        predictions = model.predict_on_batch(features).flatten()
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Labels:      %s' % labels)
        print('Predictions: %s' % predictions.numpy())

        plt.gcf().clear()
        plt.figure(figsize=(15, 15))

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis('off')
            plt.imshow(features[i].numpy().astype('uint8'))  
            plt.title(class_name[predictions[i]])

# Plotar previsões no conjunto de dados de teste
plot_dataset_predictions(dataset_teste)
plt.show()

# Avaliar o modelo no conjunto de dados de teste
loss, accuracy = model.evaluate(dataset_teste)


