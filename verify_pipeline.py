import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from utils import load_dataset
    from model import CurlClassifier
    from export_mobile import export_to_torchscript, export_quantized_model
    
    print("Dependencies and imports: OK")
    
    # 1. Load Data
    base_path = 'data'
    X, y = load_dataset(base_path)
    print(f"Data loading: OK. Loaded {len(X)} windows.")
    
    if len(X) == 0:
        print("Error: No data loaded. Check data directory.")
        sys.exit(1)
        
    # 2. Test Model
    X_torch = torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)
    
    model = CurlClassifier()
    output = model(X_torch[:5])
    print(f"Model forward pass: OK. Output shape: {output.shape}")
    
    # 3. Quick Training (1 epoch)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    optimizer.zero_grad()
    outputs = model(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    print(f"Training loop (1 epoch): OK. Loss: {loss.item():.4f}")
    
    # 4. Save and Export
    os.makedirs('models_saved', exist_ok=True)
    torch.save(model.state_dict(), 'models_saved/curl_classifier.pth')
    
    export_to_torchscript(model, 'models_saved/curl_classifier_scripted.pt')
    export_quantized_model(model, 'models_saved/curl_classifier_quant.pth')
    
    print("Verification pipeline: COMPLETED SUCCESSFULLY")

except Exception as e:
    print(f"Verification pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
