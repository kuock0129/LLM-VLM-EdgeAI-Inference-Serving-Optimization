import os
import onnx
from onnx import helper
from onnx import TensorProto

# Define input and output tensor dimensions matching config.pbtxt
input_shape = [1, 3, 224, 224]
output_shape = [1, 1000]

# Create input and output nodes (Tensor Information)
input_tensor = helper.make_tensor_value_info('input_0', TensorProto.FLOAT, input_shape)
output_tensor = helper.make_tensor_value_info('output_0', TensorProto.FLOAT, output_shape)

# Define the target shape as a constant initializer tensor for Reshape
shape_tensor = helper.make_tensor(
    'shape_tensor',
    TensorProto.INT64,
    [2],
    [1, 1000]
)

# Create a Reshape node to transform input_0 to output_0 shape
node_def = helper.make_node(
    'Reshape',
    inputs=['input_0', 'shape_tensor'],
    outputs=['output_0']
)

# Create the graph definition
graph_def = helper.make_graph(
    [node_def],
    'simple_test_graph',
    [input_tensor],
    [output_tensor],
    initializer=[shape_tensor]
)

# CRITICAL FIX 1: Explicitly specify an older official Opset version (Opset 21) 
# instead of letting it default to the experimental Opset 26.
opset_imports = [helper.make_opsetid("ai.onnx", 21)]

# Create the final ONNX model with the explicit opset
model_def = helper.make_model(graph_def, producer_name='triton-test', opset_imports=opset_imports)

# CRITICAL FIX 2: Explicitly downgrade the IR version to match Triton's expectation
# ir_version 7 corresponds to ONNX official stable Release 1.10/1.11
model_def.ir_version = 7

# Set the target output directory and path
output_dir = os.path.expanduser("~/triton_test/model_repository/basic_onnx/1")
output_path = os.path.join(output_dir, "model.onnx")

# Ensure directory exists and save the model
os.makedirs(output_dir, exist_ok=True)
onnx.save(model_def, output_path)

print(f"Success! Backwards-compatible ONNX dummy model (Opset 21, IR 7) generated at: {output_path}")
