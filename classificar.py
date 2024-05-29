from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os 
import matplotlib.pyplot as plt
import tensorflow as tf

# Diretório base
dataset_dir = os.path.join(os.getcwd(), 'dataset/split_dataset')

# Contagem dos dados
dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_validation_dir = os.path.join(dataset_dir, 'validation')

# Visualização da quantidade de imagens
def quantidade(train_cats_len, train_dogs_len, validation_cats_len, validation_dogs_len):
    print('Train cats:', train_cats_len)
    print('Train dogs:', train_dogs_len)
    print('Validation cats:', validation_cats_len)
    print('Validation dogs:', validation_dogs_len)

dataset_train_cats_len = len(os.listdir(os.path.join(dataset_train_dir, 'cat')))
dataset_train_dogs_len = len(os.listdir(os.path.join(dataset_train_dir, 'dog')))
dataset_validation_cats_len = len(os.listdir(os.path.join(dataset_validation_dir, 'cat')))
dataset_validation_dogs_len = len(os.listdir(os.path.join(dataset_validation_dir, 'dog')))

quantidade(dataset_train_cats_len, dataset_train_dogs_len, dataset_validation_cats_len, dataset_validation_dogs_len)

# Pré-processamento das imagens
image_size = (160, 160)
batch_size = 32

# Dataset de treinamento e validação
dataset_train = image_dataset_from_directory(
    dataset_train_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

dataset_validation = image_dataset_from_directory(
    dataset_validation_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2)
])

# Aplicação da data augmentation ao dataset de treinamento
def augment_data(images, labels):
    return data_augmentation(images), labels

dataset_train = dataset_train.map(augment_data)

# Transfer learning com MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Modelo
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(160, 160, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Treinamento do modelo
history = model.fit(
    dataset_train,
    validation_data=dataset_validation,
    epochs=32,
    callbacks=[early_stopping, reduce_lr]
)
# Salvar modelo
model.save('modelo.h5')
# Função para plotar os resultados
def plot_model(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(accuracy))

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_model(history)
