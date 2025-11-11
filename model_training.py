from pathlib import Path
from mutagen import File
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import time

#Windows usa "\" para separar diretórios. O "r" antes da String é necessário para funcionamento do programa.
CAMINHO_PARA_SONS = Path(r"Seu\Caminho")
CAMINHO_PARA_SALVAR_SONS = Path(r"Seu\Caminho\Salvar")

DURACAO_MINIMA = 5
DURACAO_MAXIMA = 20

#Variáveis do espectrograma
SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
TICKS = np.array([250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["250", "500", "1k", "2k", "4k", "8k"])
LIMITE_FREQUENCIA_INFERIOR = 250
LIMITE_FREQUENCIA_SUPERIOR = 8000

def get_duration(file_path):
    try:
        audio = File(file_path)
        if audio is not None:
            return audio.info.length
        return None
    except:
        return None

#Função para plotar e salvar os espectrogramas
def plot_spectrogram_and_save(signal, fs, numero_arquivo, output_path: Path, fft_size=2048, hop_size=None, window_size=None):
    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 8

    stft = librosa.stft(
        signal,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        center=False,
    )
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        x_axis="time",
        sr=fs,
        hop_length=hop_size,
        cmap="inferno",
        fmin=LIMITE_FREQUENCIA_INFERIOR,
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(TICKS, TICK_LABELS)
    plt.ylim(LIMITE_FREQUENCIA_INFERIOR, LIMITE_FREQUENCIA_SUPERIOR)
    plt.colorbar(format="%+2.f dBFS")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path.with_stem(
            f"EspectrogramaArquivo{numero_arquivo}"
        ),
        **SAVE_PARAMS,
    )
    plt.close()


def main():
    tempo_inicial = time.time()
    numero_erros = 0
    lista_arquivos_com_erro = []
    for root, dirs, files in os.walk(CAMINHO_PARA_SONS):
        relative_path = os.path.relpath(root, CAMINHO_PARA_SONS)
        output_dir = Path(CAMINHO_PARA_SALVAR_SONS) / relative_path

        if files:
            nome_especie = os.path.basename(root)
            print(f"Criando espectrogramas de {nome_especie}")

            os.makedirs(output_dir, exist_ok=True)

            for filename in files:
                nome_arquivo = filename
                file_path = Path(root) / filename
                duration = get_duration(file_path)
                if duration is not None:
                    if DURACAO_MINIMA <= duration <= DURACAO_MAXIMA:
                        try:
                            plt.rcParams.update({"font.size": 20})
                            signal, sample_rate = sf.read(file_path)
                            print(f"Sample rate: {sample_rate}")
                            signal_mono = librosa.to_mono(signal.T)
                            numero_arquivo = int(
                                nome_arquivo.replace(".mp3", "").replace(".wav", "").replace(".ogg", ""))
                            output_path = output_dir / f"EspectrogramaArquivo{numero_arquivo}.png"
                            plot_spectrogram_and_save(signal_mono, sample_rate, numero_arquivo, output_path)
                            print(f"Arquivo número {numero_arquivo} transformado em Espectrograma")
                        except Exception as e:
                            print(f"Erro processando {file_path}: {str(e)}")
                            numero_erros += 1
                            lista_arquivos_com_erro.append(nome_arquivo)
    tempo_final = time.time()
    tempo_total = tempo_final - tempo_inicial
    print(f"Houve {numero_erros} arquivo(s) com erro(s). O(s) arquivo(s) com erro(s) foi|foram"
          f" {lista_arquivos_com_erro}.")
    print(f"Tempo de processo: {tempo_total:.2f} seconds")

if __name__ == "__main__":
    main()