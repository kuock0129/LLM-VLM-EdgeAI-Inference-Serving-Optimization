#!/usr/bin/env python3
"""
Quantization for pruned ONNX models with empty initializers.

Root cause: pruning leaves 84 initializer descriptors with no data
(no external ref, no raw_data). The quantizer crashes on these.

Fix: patch the proto to remove empty initializers and redirect any
MatMul nodes that referenced them to use a zeros constant instead,
then save using onnx's external-data chunked writer (avoids 2GB limit)
and quantize from that patched file.
"""

import sys
import subprocess
from pathlib import Path
import platform

print("=" * 80)
print("QUANTIZATION — patching empty initializers before quantizing")
print("=" * 80)

INPUT_DIR  = Path("/Users/sashalai/Documents/UW/26sp/eep564/FinalProject/model_quantization/model_quantization_outputs/pruned/Llama-3.2-3B-Instruct_pruned_30pct_onnx")
OUTPUT_DIR = Path("/Users/sashalai/Documents/UW/26sp/eep564/FinalProject/model_quantization/model_quantization_outputs/quantized/Llama-3.2-3B-Instruct_pruned_30pct_int8_optimum")
PATCHED_DIR = Path("/tmp/llama_patched_onnx")

input_model = INPUT_DIR / "model.onnx"
if not input_model.exists():
    print(f"❌ model.onnx not found in {INPUT_DIR}"); sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATCHED_DIR.mkdir(parents=True, exist_ok=True)

print("\nInstalling dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "onnxruntime", "onnx", "numpy"], check=True)

import onnx, numpy as np
from onnx import numpy_helper, TensorProto
from onnx.external_data_helper import load_external_data_for_model
import onnxruntime, gc

print(f"✓ onnxruntime {onnxruntime.__version__} / onnx {onnx.__version__}\n")

# ── Step 1: Find empty initializers ──────────────────────────────────────────
print("=" * 80)
print("STEP 1: IDENTIFYING EMPTY INITIALIZERS")
print("=" * 80)

model = onnx.load(str(input_model), load_external_data=False)

empty_names = set()
for init in model.graph.initializer:
    is_external = (init.data_location == TensorProto.EXTERNAL)
    has_data    = (len(init.raw_data) > 0 or len(init.float_data) > 0
                   or len(init.int32_data) > 0 or len(init.int64_data) > 0)
    if not is_external and not has_data:
        empty_names.add(init.name)

print(f"Found {len(empty_names)} empty initializers (pruned-away weights)")

# ── Step 2: Load external data for the 171 real initializers ─────────────────
print("\n" + "=" * 80)
print("STEP 2: LOADING EXTERNAL DATA FOR REAL WEIGHTS (~7 GB, takes a minute)")
print("=" * 80)
load_external_data_for_model(model, str(INPUT_DIR))
print("✓ External data loaded")

# ── Step 3: Patch — replace empty initializers with explicit zeros ────────────
print("\n" + "=" * 80)
print("STEP 3: PATCHING EMPTY INITIALIZERS → ZEROS")
print("=" * 80)

# Build a map of initializer name → index for fast lookup
init_map = {init.name: i for i, init in enumerate(model.graph.initializer)}

patched = 0
for name in empty_names:
    idx = init_map.get(name)
    if idx is None:
        continue
    init = model.graph.initializer[idx]
    # dims may be empty on fully-pruned tensors; default to scalar zero
    dims = list(init.dims) if init.dims else [1]
    dtype = init.data_type or TensorProto.FLOAT
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)
    zeros = np.zeros(dims, dtype=np_dtype)
    new_init = numpy_helper.from_array(zeros, name=name)
    model.graph.initializer[idx].CopyFrom(new_init)
    patched += 1

print(f"✓ Patched {patched} empty initializers with zeros tensors")

# ── Step 4: Save patched model using chunked external-data writer ─────────────
# Cannot use onnx.save() — proto > 2 GB hits protobuf limit.
# onnx.save_model with all_tensors_to_one_file writes a small .onnx + data file.
print("\n" + "=" * 80)
print("STEP 4: SAVING PATCHED MODEL (chunked external data)")
print("=" * 80)

patched_model_path = PATCHED_DIR / "model.onnx"

# Move all initializers to external storage to stay under protobuf 2GB limit
onnx.save_model(
    model,
    str(patched_model_path),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="model.onnx_data",
    size_threshold=1024,
)
del model; gc.collect()
print(f"✓ Patched model saved to {PATCHED_DIR}")

# Verify no empty initializers remain
verify = onnx.load(str(patched_model_path), load_external_data=False)
still_empty = sum(
    1 for i in verify.graph.initializer
    if i.data_location != TensorProto.EXTERNAL
    and len(i.raw_data) == 0
    and len(i.float_data) == 0
)
print(f"✓ Verification: {still_empty} empty initializers remaining (expect 0)")
del verify; gc.collect()

if still_empty > 0:
    print("❌ Patch incomplete — some initializers are still empty"); sys.exit(1)

# ── Step 5: Quantize ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 5: INT8 QUANTIZATION (arm64 safe: per_channel=False)")
print("=" * 80)
print("⏳ 20-40 minutes...\n")

from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    quantize_dynamic(
        model_input=str(patched_model_path),
        model_output=str(OUTPUT_DIR / "model.onnx"),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=False,       # True segfaults ORT on arm64
        reduce_range=False,      # ditto
        use_external_data_format=True,
    )
    print("✓ Quantization complete!")

except Exception as e:
    print(f"\n❌ Quantization failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Size report ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
for f in sorted(OUTPUT_DIR.glob("model*")):
    gb = f.stat().st_size / 1024**3
    print(f"  {f.name}: {gb:.3f} GB")

orig = (INPUT_DIR / "model.onnx_data").stat().st_size / 1024**3
quant_f = OUTPUT_DIR / "model.onnx_data"
if quant_f.exists():
    quant = quant_f.stat().st_size / 1024**3
    print(f"\nCompression: {(1-quant/orig)*100:.1f}%  ({orig:.2f} GB → {quant:.2f} GB)")

import shutil; shutil.rmtree(PATCHED_DIR, ignore_errors=True)
print(f"✓ Cleaned up {PATCHED_DIR}")
print("\n✓ Done!")