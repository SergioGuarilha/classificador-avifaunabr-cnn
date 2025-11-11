import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import os

# Modelo deve ser igual ao de treinamento
class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(8),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 256, 512)
            features = self.features(dummy_input)
            feature_size = features.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

CAMINHO_MODELO = r"D:\ModelosCNN\CrossVal5Folds\Turdidae\TurdidaeCrossValFold3(3).pth"
DIR_IMAGEM_PARA_TESTE = r"D:\TesteEspectro\Turdidae"
ARQUIVO_RESULTADOS = "D:\Predicoes\predictionsT(teste).csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo treinado
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = BirdCNN(num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint['class_names']


# Transformações devem ser iguais as usadas pelo modelo
def get_test_transform():
    return v2.Compose([
        v2.Lambda(lambda x: x.crop((316, 60, 2181, 986))),
        v2.Grayscale(),
        v2.Resize((256, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])


# Função de previsão
def predict_image(image_path, model, transform, class_names):
    try:
        img = Image.open(image_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        return {
            'filename': os.path.basename(image_path),
            'predicted_class': class_names[pred.item()],
            'confidence': conf.item() * 100,
            'all_predictions': {class_names[i]: f"{prob.item() * 100:.1f}%"
                                for i, prob in enumerate(probs)}
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def run_tests():
    # Carrega modelo e nomes de classe
    model, class_names = load_model(CAMINHO_MODELO)
    transform = get_test_transform()

    # Processa todas as imagens no diretório de testes
    results = []
    image_files = [f for f in os.listdir(DIR_IMAGEM_PARA_TESTE)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"\nTesting {len(image_files)} spectrograms...")
    for img_file in image_files:
        img_path = os.path.join(DIR_IMAGEM_PARA_TESTE, img_file)
        result = predict_image(img_path, model, transform, class_names)
        if result:
            results.append(result)
            print(f"{img_file}: {result['predicted_class']} ({result['confidence']:.1f}%)")

    # Salva os resultados num csv
    if results:
        import csv
        with open(ARQUIVO_RESULTADOS, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {ARQUIVO_RESULTADOS}")


if __name__ == "__main__":
    run_tests()