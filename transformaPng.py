from PIL import Image
import os

def convert_folder_png_to_jpeg(folder_path, output_folder):
    # Verificar se o caminho de saída existe, caso contrário, criar
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Listar todos os arquivos na pasta de entrada
    files = os.listdir(folder_path)
    
    # Iterar sobre os arquivos na pasta de entrada
    for file in files:
        if file.endswith(".png"):  # Verificar se o arquivo é um PNG
            png_path = os.path.join(folder_path, file)
            jpeg_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".jpeg")
            convert_png_to_jpeg(png_path, jpeg_path)

def convert_png_to_jpeg(png_path, jpeg_path):
    try:
        # Abrir a imagem PNG
        img = Image.open(png_path)
        
        # Salvar como JPEG
        img.save(jpeg_path, "JPEG")
        print(f"{png_path} convertido para {jpeg_path}")
        
        # Excluir o arquivo PNG original
        os.remove(png_path)
        print(f"Arquivo {png_path} excluído.")
    except Exception as e:
        print(f"Erro ao converter a imagem: {e}")

# Caminho para a pasta de entrada contendo os arquivos PNG
pasta_entrada = "dataset/Split_dataset/train/cat"

# Caminho para a pasta de saída para os arquivos JPEG convertidos
pasta_saida = "dataset/split_dataset/train/cat"

# Converter todos os arquivos PNG na pasta de entrada para JPEG na pasta de saída
convert_folder_png_to_jpeg(pasta_entrada, pasta_saida)

