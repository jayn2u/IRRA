import torch
import torch.onnx
from model.build import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
import os
import argparse

def export_onnx_icfg():
    config_path = "/home/jayn2u/IRRA/irra_icfg/configs.yaml"
    model_path = "/home/jayn2u/IRRA/irra_icfg/best.pth"
    
    print(f"Loading configuration from {config_path}")
    args = load_train_configs(config_path)

    args.loss_names = "sdm+mlm" 
    
    print("Building model...")
    # Enable FP16 conversion by setting use_fp16=True (default)
    model = build_model(args, num_classes=100)
    
    print(f"Loading checkpoint from {model_path}")
    checkpointer = Checkpointer(model)
    checkpointer.load(f=model_path)
    
    device = "cuda"
    model.to(device)
    model.eval()
    
    batch_size = 1
    # Use float16 for dummy image since model is FP16
    dummy_image = torch.randn(batch_size, 3, 384, 128, dtype=torch.float32).to(device)
    dummy_text = torch.randint(0, 49408, (batch_size, 77), dtype=torch.int32).to(device)

    output_dir = "/home/jayn2u/IRRA/irra_icfg"

    print("\nAttempting to export Image Encoder...")
    
    class ImageEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, image):
            return self.model.encode_image(image)
    
    img_enc = ImageEncoderWrapper(model)
    
    img_onnx_path = os.path.join(output_dir, "irra_icfg_image_encoder.onnx")
    try:
        torch.onnx.export(
            img_enc,
            (dummy_image,),
            img_onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['image_features'],
            dynamic_axes={'image': {0: 'batch_size'}, 'image_features': {0: 'batch_size'}}
        )
        print(f"Success: Image Encoder exported to {img_onnx_path}")
    except Exception as e:
        print(f"Failed to export Image Encoder: {e}")

    print("\nAttempting to export Text Encoder...")

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, text):
            return self.model.encode_text(text)

    txt_enc = TextEncoderWrapper(model)

    txt_onnx_path = os.path.join(output_dir, "irra_icfg_text_encoder.onnx")
    try:
        torch.onnx.export(
            txt_enc,
            (dummy_text,),
            txt_onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['text'],
            output_names=['text_features'],
            dynamic_axes={'text': {0: 'batch_size'}, 'text_features': {0: 'batch_size'}}
        )
        print(f"Success: Text Encoder exported to {txt_onnx_path}")
    except Exception as e:
        print(f"Failed to export Text Encoder: {e}")

if __name__ == "__main__":
    export_onnx_icfg()
