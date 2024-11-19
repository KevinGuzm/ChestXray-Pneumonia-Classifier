import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from utils import load_model
import os

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = load_model('model_weights.pth', device)

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Función de predicción
def predict(image):
    image = transform(image).unsqueeze(0).to(device)  # Preprocesar imagen
    model.eval()
    with torch.no_grad():
        output = model(image).item()
    confidence = output if output > 0.5 else 1 - output
    result = "Pneumonia" if output > 0.5 else "Normal"
    return f"Prediction: {result} (Confidence: {confidence:.2f})"

# Directorio con imágenes de ejemplo
example_images_dir = "./examples/"
example_images = [
    [os.path.join(example_images_dir, img)]
    for img in os.listdir(example_images_dir)
    if img.lower().endswith(('jpg', 'jpeg', 'png'))
]

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Pneumonia Detection")
    gr.Markdown("Upload a chest X-ray image or select one of the example images below.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Chest X-ray")
            predict_button = gr.Button("Predict")
        with gr.Column():
            result_output = gr.Textbox(label="Result")

    gr.Examples(
        examples=example_images,
        inputs=image_input,
        label="Example Images"
    )

    predict_button.click(fn=predict, inputs=[image_input], outputs=[result_output])

if __name__ == "__main__":
    demo.launch(share=True)