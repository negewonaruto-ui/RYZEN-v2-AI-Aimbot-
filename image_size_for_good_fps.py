from ultralytics import YOLO

# 1. Load your original PyTorch model
# Ensure 'best.pt' is in the same directory as this script.
try:
    model = YOLO('best.pt') 
except FileNotFoundError:
    print("Error: 'best.pt' not found. Make sure the model file is in the same folder.")
    exit()

# 2. Export the model to ONNX format with the corrected input size (you want 300 is good but 224 for fps boost but use 300 if you have good cpu)
# This will overwrite your existing best.onnx file.
print(f"Exporting model to ONNX with input size 224...")
model.export(
    format='onnx', 
    imgsz=200,        # The new size to match your configuration
    simplify=True,    # Recommended for faster inference
    # Assuming 'best.onnx' is your target file name
    # The default 'best.onnx' is usually correct if you don't specify a 'name'
) 

print("✅ Model successfully re-exported to best.onnx with 224 input size.")
