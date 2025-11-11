import numpy as np
import os
from PIL import Image
from pathlib import Path

CAMINHO_DIRETORIO_ESPECTROGRAMAS = Path("r\seu\caminhoparaespectrogramas")
CAMINHO_LISTA_FILTRADA_WINDOWS = Path("r\seu\caminhoparalistaW")
CAMINHO_LISTA_FILTRADA_GDRIVE = Path("r\seu\caminhoparacriarlistaGD")

def is_usable_spectrogram(img_path, min_contrast, max_contrast, max_energy, max_black):
    #Função de filtro
    try:
        img = Image.open(img_path).convert('L')

        img_cropped = img.crop((316, 60, 2181, 986))

        img_array = np.array(img_cropped, dtype=np.float32) / 255.0

        contrast = np.std(img_array) * 255
        mean_energy = np.mean(img_array)
        black_pixels = np.sum(img_array < 0.1) / img_array.size

        if contrast < min_contrast:
            print(f"Rejected {img_path}: Low contrast {contrast:.1f} < {min_contrast}")
            return False

        if contrast > max_contrast:
            print(f"Rejected {img_path}: High contrast {contrast:.1f} > {max_contrast}")
            return False

        if mean_energy > max_energy:
            print(f"Rejected {img_path}: Overexposed {mean_energy:.3f} > {max_energy}")
            return False

        if black_pixels > max_black:
            print(f"Rejected {img_path}: {black_pixels * 100:.1f}% black > {max_black * 100}%")
            return False

        return True

    except Exception as e:
        print(f"Error checking {img_path}: {str(e)}")
        return False

def create_filtered_filelist(root_dir):
    output_dir = root_dir.parent / "filtered_lists"
    output_dir.mkdir(exist_ok=True)

    for class_dir in os.listdir(root_dir):
        class_path = root_dir / class_dir
        if class_path.is_dir():
            output_file = output_dir / f"filtered_files_{class_dir}.txt"

            with open(output_file, 'w') as f:
                for root, dirs, files in os.walk(class_path):
                    for img_file in files:
                        img_path = Path(root) / img_file
                        if is_usable_spectrogram(str(img_path), min_contrast=18, max_contrast=55, max_energy=0.55,
                                                 max_black=0.75):
                            f.write(f"{img_path},{class_dir}\n")

            print(f"Created filtered list for {class_dir}: {output_file}")

def create_simple_google_drive_filelist(original_file, output_file):
    with open(original_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            img_path, _ = line.split(',')

            path_parts = img_path.split('\\')
            species_name = path_parts[-2]

            filename = os.path.basename(img_path)

            new_path = f"/content/drive/MyDrive/Espectrogramas5a20/Ramphastidae/{species_name}/{filename}"

            f_out.write(f"{new_path},{species_name}\n")

    print(f"Created corrected Google Drive filelist")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    create_filtered_filelist(CAMINHO_DIRETORIO_ESPECTROGRAMAS)
    create_simple_google_drive_filelist(CAMINHO_LISTA_FILTRADA_WINDOWS, CAMINHO_LISTA_FILTRADA_GDRIVE)
