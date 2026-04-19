import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import os
from model import CurlClassifier

def export_to_torchscript(model, output_path, example_input=None):
    """
    Export the model to mobile-optimized TorchScript format for Lite Interpreter.
    """
    if example_input is None:
        example_input = torch.randn(1, 6, 100)
    
    model.eval()
    # 1. Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # 2. Optimize for mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # 3. Save for Lite Interpreter
    optimized_model._save_for_lite_interpreter(output_path)
    
    print(f"Model exported and optimized for mobile: {output_path}")

def export_quantized_model(model, output_path):
    """
    Apply dynamic INT8 quantization and save the model state dict.
    Note: Dynamic quantization is suitable for CPU inference.
    """
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model state dict saved: {output_path}")

if __name__ == "__main__":
    model = CurlClassifier()
    
    os.makedirs("models_saved", exist_ok=True)
    
    export_to_torchscript(model, "models_saved/curl_classifier_scripted.pt")
    export_quantized_model(model, "models_saved/curl_classifier_quant.pth")
