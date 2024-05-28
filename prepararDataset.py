import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split

# criar dataset de train e validation
dataset_dir = 'dataset'
base_dir = 'dataset/Split_dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Criar os diretórios base, de treino e de validação
os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

#copia de divide as img em train e validação
def splitCopy(class_name):
    class_dir = os.path.join(dataset_dir, class_name)
    all_files = os.listdir(class_dir)
    
    train_files, validation_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    train_class_dir = os.path.join(train_dir, class_name)
    validation_class_dir = os.path.join(validation_dir, class_name)
    
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(validation_class_dir, exist_ok=True)
    
    for file_name in train_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(train_class_dir, file_name)
        shutil.copyfile(src, dst)
    
    for file_name in validation_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(validation_class_dir, file_name)
        shutil.copyfile(src, dst)

classes = ['cat', 'dog']
for class_name in classes:
    splitCopy(class_name)
print("Processo concluído!")