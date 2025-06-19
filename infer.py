import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.deblur_unet import DeblurNet
from utils.model import DeblurAdvancedUnet

def load_model(checkpoint_path, device):
    base_model=DeblurNet()
    model = DeblurAdvancedUnet.load_from_checkpoint(checkpoint_path,model=base_model,lr=2e-5)
    model.to(device)
    model.eval()
    return model

def infer(model, test_dir, output_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    to_pil = transforms.ToPILImage()

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(image_files, desc="Deblurring images"):
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0).cpu()
        
        output_image = to_pil(output_tensor.clamp(0, 1))
        output_image.save(os.path.join(output_dir, img_name))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deblur a folder of test images.")
    parser.add_argument("--test_dir", type=str, default="data/test/blur", help="Path to the test images folder (blurred).")
    parser.add_argument("--checkpoint", type=str, default="model_ckp/best.ckpt", help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--output_dir", type=str, default="data/test/outputs", help="Where to save the deblurred images.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    infer(model, args.test_dir, args.output_dir, device)
