import torch
import numpy as np
from PIL import Image
from model import EyeGazeCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, input_channels=1, output_dim=3):
    model = EyeGazeCNN(input_channels, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, ...]
    tensor = torch.from_numpy(img).to(DEVICE)

    with torch.no_grad():
        gaze = model(tensor).cpu().numpy()[0]

    return gaze


if __name__ == '__main__':
    model = load_model('final_model.pt', output_dim=3)
    gaze = predict_image(model, 'test_eye.png')
    print('Predicted gaze:', gaze)
