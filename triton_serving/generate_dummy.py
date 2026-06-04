import os
import torch
import torch.nn as nn

# Define a minimal linear network for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Flattened input size: 3 channels * 224 height * 224 width
        self.fc = nn.Linear(3 * 224 * 224, 1000)

    def forward(self, x):
        # Flatten the input tensor from [B, C, H, W] to [B, C*H*W] before the linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Initialize the model and set it to evaluation mode
model = SimpleModel()
model.eval()

# Create a dummy input tensor matching the dimensions specified in config.pbtxt [1, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224)

# Set the target output directory and path for the ONNX model
output_dir = os.path.expanduser("~/triton_test/model_repository/basic_onnx/1")
output_path = os.path.join(output_dir, "model.onnx")

# Ensure the target directory exists and export the model to ONNX format
os.makedirs(output_dir, exist_ok=True)
torch.onnx.export(
    model, 
    dummy_input, 
    output_path, 
    input_names=['input_0'], 
    output_names=['output_0']
)

print(f"Success! Dummy model has been generated at: {output_path}")
