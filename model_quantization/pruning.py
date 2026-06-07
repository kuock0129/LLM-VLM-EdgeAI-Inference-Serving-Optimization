"""
Model Pruning Module

This module handles:
1. Loading HuggingFace models
2. Applying Wanda pruning (pruning by weights and activations)
3. Exporting pruned models to ONNX format
4. Measuring compression and sparsity
"""

import os
import torch
import torch.nn as nn
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json
import numpy as np
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelPruner:
    """Handles pruning of PyTorch models and ONNX export"""

    # Models that require special handling
    MULTIMODAL_MODELS = ['lmms-lab/LLaVA-OneVision-1.5-4B-Instruct']

    def __init__(self, base_output_dir: str):
        """
        Initialize the pruner

        Args:
            base_output_dir: Base directory for all model outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.pruned_dir = self.base_output_dir / "pruned"
        self.pruned_dir.mkdir(parents=True, exist_ok=True)
        self.quantized_dir = self.base_output_dir / "quantized"
        (self.quantized_dir / "INT8").mkdir(parents=True, exist_ok=True)
        (self.quantized_dir / "INT4").mkdir(parents=True, exist_ok=True)
        (self.quantized_dir / "FP16").mkdir(parents=True, exist_ok=True)

    def get_pruned_output_dir(self, model_name: str, sparsity: float) -> Path:
        """Get the output directory for a pruned model"""
        model_short_name = model_name.split('/')[-1]
        sparsity_str = f"{int(sparsity * 100)}pct"
        return self.pruned_dir / f"{model_short_name}_pruned_{sparsity_str}"

    def load_wikitext2_calibration_data(
        self,
        tokenizer,
        num_samples: int = 128,
        seq_length: int = 128
    ) -> torch.Tensor:
        """
        Load WikiText-2 calibration data for Wanda pruning

        Args:
            tokenizer: HuggingFace tokenizer
            num_samples: Number of calibration samples
            seq_length: Sequence length for each sample

        Returns:
            Tensor of input_ids for calibration
        """
        logger.info("Loading WikiText-2 calibration data...")

        try:
            # Try loading from datasets library
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            logger.info(f"Loaded WikiText-2 from HuggingFace datasets")
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.info("Using fallback calibration data")
            # Fallback: use simple text
            dataset = [{"text": "This is a sample text for calibration. " * 50}] * num_samples

        # Prepare calibration samples
        calibration_data = []
        text_samples = []

        # Extract text from dataset
        if hasattr(dataset, '__iter__'):
            for item in dataset:
                if isinstance(item, dict) and 'text' in item:
                    text = item['text'].strip()
                    if len(text) > 50:  # Only use non-empty samples
                        text_samples.append(text)
                        if len(text_samples) >= num_samples * 2:  # Get extra for filtering
                            break

        logger.info(f"Collected {len(text_samples)} text samples")

        # Set padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added new pad_token: [PAD]")

        # Tokenize samples
        for text in text_samples[:num_samples]:
            try:
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=seq_length,
                    truncation=True,
                    padding="max_length"
                )
                calibration_data.append(tokens.input_ids)

                if len(calibration_data) >= num_samples:
                    break
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                continue

        if not calibration_data:
            logger.error("No calibration data could be prepared!")
            raise ValueError("Failed to prepare calibration data")

        # Stack into batch
        calibration_tensor = torch.cat(calibration_data, dim=0)
        logger.info(f"Calibration data shape: {calibration_tensor.shape}")

        return calibration_tensor

    def collect_activation_statistics(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activation statistics for Wanda pruning

        Args:
            model: PyTorch model
            calibration_data: Input calibration data
            device: Device to run on

        Returns:
            Dictionary mapping layer names to activation norms
        """
        logger.info("Collecting activation statistics...")

        model.eval()
        model = model.to(device)
        calibration_data = calibration_data.to(device)

        activation_stats = {}
        hooks = []

        def get_activation_hook(name):
            def hook(module, input, output):
                # Compute activation norms from INPUT (not output) for Wanda pruning
                # Wanda importance: |weight[i,j]| * ||input[j]||
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input

                if hasattr(input_tensor, 'abs'):
                    # Compute average absolute activation per input feature
                    # For Linear layer: input shape is [batch, seq_len, in_features]
                    act_norm = input_tensor.abs().mean(dim=(0, 1)) if input_tensor.dim() > 2 else input_tensor.abs().mean(dim=0)

                    if name not in activation_stats:
                        activation_stats[name] = []
                    activation_stats[name].append(act_norm.cpu())

            return hook

        # Register hooks for Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)

        # Run calibration data through model
        try:
            with torch.no_grad():
                batch_size = 8
                num_batches = (len(calibration_data) + batch_size - 1) // batch_size
                logger.info(f"Running {num_batches} batches through model (this may take 5-10 minutes on CPU)...")

                for batch_idx, i in enumerate(range(0, len(calibration_data), batch_size)):
                    batch = calibration_data[i:i+batch_size]
                    try:
                        # Log progress every few batches
                        if batch_idx % 4 == 0 or batch_idx == num_batches - 1:
                            logger.info(f"  Processing batch {batch_idx + 1}/{num_batches} ({(batch_idx + 1) / num_batches * 100:.1f}%)...")

                        _ = model(batch)
                    except Exception as e:
                        logger.warning(f"Forward pass error on batch {batch_idx + 1}: {e}")
                        continue

                logger.info("✓ Activation collection complete!")
        except Exception as e:
            logger.error(f"Error during activation collection: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Average activation statistics across batches
        avg_activation_stats = {}
        for name, stats_list in activation_stats.items():
            if stats_list:
                avg_activation_stats[name] = torch.stack(stats_list).mean(dim=0)

        logger.info(f"Collected activation stats for {len(avg_activation_stats)} layers")

        return avg_activation_stats

    def apply_wanda_pruning(
        self,
        model: nn.Module,
        tokenizer,
        sparsity: float = 0.5,
        num_calibration_samples: int = 64,
        device: str = "cpu"
    ) -> nn.Module:
        """
        Apply Wanda pruning (pruning by weights and activations)

        Wanda pruning computes importance scores as: importance = |weight| * |activation|
        and prunes weights with lowest importance scores.

        Args:
            model: PyTorch model
            tokenizer: HuggingFace tokenizer for calibration data
            sparsity: Target sparsity (0.0 to 1.0)
            num_calibration_samples: Number of calibration samples (default: 64 for speed)
            device: Device to run on

        Returns:
            Pruned model
        """
        logger.info(f"Applying Wanda pruning with {sparsity*100:.0f}% sparsity...")

        # Load calibration data
        calibration_data = self.load_wikitext2_calibration_data(
            tokenizer,
            num_samples=num_calibration_samples
        )

        # Collect activation statistics
        activation_stats = self.collect_activation_statistics(
            model,
            calibration_data,
            device=device
        )

        # Apply pruning to each Linear layer
        pruned_params = 0
        total_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activation_stats:
                weight = module.weight.data

                # Get activation statistics for this layer
                act_norm = activation_stats[name].to(weight.device)

                # Compute importance scores: |weight| * |input_activation|
                # act_norm has shape [in_features], weight has shape [out_features, in_features]
                # Wanda pruning: importance[i,j] = |weight[i,j]| * ||input[j]||
                importance = weight.abs() * act_norm.unsqueeze(0)  # [in_features] -> [1, in_features]

                # Flatten importance scores
                importance_flat = importance.flatten()

                # Compute threshold for pruning
                num_params = importance_flat.numel()
                num_to_prune = int(sparsity * num_params)

                if num_to_prune > 0:
                    # Find threshold value
                    threshold = torch.kthvalue(importance_flat, num_to_prune).values

                    # Create mask: keep weights with importance > threshold
                    mask = (importance > threshold).float()

                    # Apply mask
                    module.weight.data *= mask

                    # Track pruning stats
                    pruned_params += (mask == 0).sum().item()
                    total_params += num_params

                    logger.debug(f"Pruned layer {name}: {(mask == 0).sum().item()}/{num_params} params")

        achieved_sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Wanda pruning complete: {achieved_sparsity*100:.2f}% sparsity achieved")

        return model

    def calculate_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate sparsity statistics for a model

        Args:
            model: PyTorch model

        Returns:
            Dictionary with sparsity metrics
        """
        logger.info("Calculating model sparsity...")
        total_params = 0
        zero_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        overall_sparsity = zero_params / total_params if total_params > 0 else 0

        logger.info(f"✓ Sparsity calculation complete")

        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'overall_sparsity': overall_sparsity,
            'compression_ratio': 1 - overall_sparsity
        }

    def apply_unstructured_pruning(
        self,
        model: nn.Module,
        sparsity: float = 0.3,
        layers_to_prune: Optional[list] = None
    ) -> nn.Module:
        """
        Apply magnitude-based unstructured pruning to model

        Args:
            model: PyTorch model
            sparsity: Target sparsity (0.0 to 1.0)
            layers_to_prune: List of layer names to prune (None = all linear layers)

        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune

        logger.info(f"\nApplying unstructured pruning with {sparsity*100:.1f}% sparsity...")
        logger.info("This may take 1-3 minutes...")

        # Track pruned layers
        pruned_layers = []
        total_modules = sum(1 for _ in model.named_modules())
        processed = 0

        for name, module in model.named_modules():
            processed += 1

            # Log progress every 100 modules
            if processed % 100 == 0:
                logger.info(f"  Progress: {processed}/{total_modules} modules processed...")

            # Target linear layers and embeddings
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Skip if specific layers specified and this isn't one
                if layers_to_prune and name not in layers_to_prune:
                    continue

                # Apply L1 unstructured pruning
                try:
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    # Make pruning permanent
                    prune.remove(module, 'weight')
                    pruned_layers.append(name)
                except Exception as e:
                    logger.warning(f"Could not prune layer {name}: {e}")

        logger.info(f"✓ Pruned {len(pruned_layers)} layers")

        # Calculate and log sparsity
        sparsity_stats = self.calculate_sparsity(model)
        logger.info(f"✓ Achieved sparsity: {sparsity_stats['overall_sparsity']*100:.2f}%")

        return model

    def apply_structured_pruning(
        self,
        model: nn.Module,
        sparsity: float = 0.3,
        dim: int = 0
    ) -> nn.Module:
        """
        Apply structured pruning (prune entire neurons/channels)

        Args:
            model: PyTorch model
            sparsity: Target sparsity (0.0 to 1.0)
            dim: Dimension to prune (0 for rows, 1 for columns)

        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune

        logger.info(f"\nApplying structured pruning with {sparsity*100:.1f}% sparsity (dim={dim})...")
        logger.info("This may take 1-3 minutes...")

        pruned_layers = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    # Apply structured pruning on specified dimension
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=sparsity,
                        n=2,  # L2 norm
                        dim=dim
                    )
                    # Make pruning permanent
                    prune.remove(module, 'weight')
                    pruned_layers.append(name)
                except Exception as e:
                    logger.warning(f"Could not prune layer {name}: {e}")

        logger.info(f"✓ Pruned {len(pruned_layers)} layers")

        # Calculate and log sparsity
        sparsity_stats = self.calculate_sparsity(model)
        logger.info(f"✓ Achieved sparsity: {sparsity_stats['overall_sparsity']*100:.2f}%")

        return model

    def load_model(self, model_name: str) -> Optional[Tuple[Any, Any]]:
        """
        Load HuggingFace model and tokenizer

        Args:
            model_name: HuggingFace model name

        Returns:
            Tuple of (model, tokenizer) or None if failed
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model {model_name}...")
            logger.info("This may take 2-5 minutes depending on model size...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model in float32 for pruning
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            logger.info(f"✓ Successfully loaded {model_name}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    def save_pytorch_model(self, model: Any, tokenizer: Any, output_dir: Path):
        """
        Save pruned PyTorch model

        Args:
            model: PyTorch model
            tokenizer: HuggingFace tokenizer
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving pruned model to {output_dir}...")

        try:
            # Update generation_config to include pad_token_id if set
            # This prevents warnings during ONNX export with optimum-cli
            if hasattr(model, 'generation_config') and tokenizer.pad_token_id is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Updated generation_config with pad_token_id: {tokenizer.pad_token_id}")

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def export_to_gguf(
        self,
        pytorch_model_path: Path,
        quantization: str = "q4_k_m",
    ) -> bool:
        """
        Export pruned HF model to GGUF and quantize with llama.cpp.

        Quantization options:
          q4_k_m  — INT4, best quality/size tradeoff (recommended)
          q8_0    — INT8, near-lossless, 2x smaller than FP16
          q4_0    — INT4, smallest, lowest quality
          f16     — FP16, no quantization (baseline)

        Requires llama.cpp installed:
          git clone https://github.com/ggerganov/llama.cpp
          cmake -B build && cmake --build build --config Release -j
        """
        # Route to quantized/INT8, quantized/INT4, or quantized/FP16
        quant_lower = quantization.lower()
        if quant_lower == "f16":
            precision_dir = self.quantized_dir / "FP16"
        elif quant_lower in ("q8_0", "q8_1"):
            precision_dir = self.quantized_dir / "INT8"
        else:
            precision_dir = self.quantized_dir / "INT4"

        model_name = pytorch_model_path.name
        gguf_dir = precision_dir / model_name
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # F16 GGUF lives in FP16 folder and is reused across quantizations
        f16_dir   = self.quantized_dir / "FP16" / model_name
        f16_dir.mkdir(parents=True, exist_ok=True)
        f16_path  = f16_dir / "model_f16.gguf"
        quant_path = gguf_dir / f"model_{quantization}.gguf"

        # Locate llama.cpp tools
        convert_script  = self._find_llama_cpp_tool("convert_hf_to_gguf.py")
        quantize_binary = self._find_llama_cpp_tool("llama-quantize")

        if not convert_script:
            logger.error("convert_hf_to_gguf.py not found. Clone llama.cpp and build it first.")
            logger.error("  git clone https://github.com/ggerganov/llama.cpp && cmake -B build && cmake --build build -j")
            return False

        # Step 1: HF safetensors → GGUF F16
        logger.info(f"Converting {pytorch_model_path} → GGUF F16...")
        result = subprocess.run(
            ["python3", convert_script, str(pytorch_model_path),
             "--outfile", str(f16_path), "--outtype", "f16"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"GGUF conversion failed:\n{result.stderr}")
            return False
        logger.info(f"✓ F16 GGUF saved: {f16_path}")

        # Step 2: quantize GGUF → INT4/INT8
        if quantization.lower() == "f16":
            logger.info("Skipping quantization (f16 requested)")
            return True

        if not quantize_binary:
            logger.error("llama-quantize binary not found. Build llama.cpp first.")
            return False

        quant_type = quantization.upper()
        logger.info(f"Quantizing GGUF to {quant_type}...")
        result = subprocess.run(
            [quantize_binary, str(f16_path), str(quant_path), quant_type],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"Quantization failed:\n{result.stderr}")
            return False

        size_gb = quant_path.stat().st_size / 1e9
        logger.info(f"✓ Quantized GGUF saved: {quant_path} ({size_gb:.2f} GB)")
        return True

    def _find_llama_cpp_tool(self, name: str) -> Optional[str]:
        """Search common locations for llama.cpp tools."""
        import shutil
        # Check PATH first
        found = shutil.which(name)
        if found:
            return found
        # Common local build paths
        candidates = [
            Path.home() / "llama.cpp" / name,
            Path.home() / "llama.cpp" / "build" / "bin" / name,
            Path("/usr/local/bin") / name,
            Path("./llama.cpp") / name,
            Path("./llama.cpp/build/bin") / name,
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def export_to_onnx(
        self,
        pytorch_model_path: Path,
        task: str = "text-generation",
        batch_size: int = 1,
        device: str = "cpu",
        num_threads: int = 4
    ) -> bool:
        """
        Export pruned PyTorch model to ONNX format

        Args:
            pytorch_model_path: Path to saved PyTorch model
            task: Model task
            batch_size: Batch size for export
            device: Device to use
            num_threads: Number of CPU threads

        Returns:
            True if successful, False otherwise
        """
        output_dir = pytorch_model_path.parent / f"{pytorch_model_path.name}_onnx"

        logger.info(f"Exporting pruned model to ONNX at {output_dir}...")

        # Prepare environment
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(num_threads)
        env['MKL_NUM_THREADS'] = str(num_threads)

        # Build command - use -m for model path
        cmd = [
            'optimum-cli', 'export', 'onnx',
            '-m', str(pytorch_model_path),  # Use -m flag (required)
            '--task', task,
            '--device', device,
            '--batch_size', str(batch_size),
            '--no-post-process',
            '--trust-remote-code',
            str(output_dir)
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("Successfully exported pruned model to ONNX")
            if result.stdout:
                logger.debug(f"Output: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export to ONNX")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Error output: {e.stderr}")
            if e.stdout:
                logger.error(f"Standard output: {e.stdout}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during ONNX export: {str(e)}")
            return False

    def process_model(
        self,
        model_name: str,
        sparsity: float = 0.5,
        pruning_method: str = "wanda",
        export_onnx: bool = True,
        export_gguf: bool = False,
        gguf_quantization: str = "q4_k_m",
        num_calibration_samples: int = 128,
        device: str = "cpu",
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pruning pipeline: load, prune, save, and export to ONNX

        CORRECT WORKFLOW:
        1. Load PyTorch model (FP32 for accurate calculations)
        2. Apply Wanda pruning with WikiText-2 calibration data
        3. Export pruned model to ONNX (using optimum-cli)
        4. Quantize the pruned ONNX model (done separately)

        Args:
            model_name: HuggingFace model name
            sparsity: Target sparsity (0.0 to 1.0), default 0.5 = 50%
            pruning_method: 'wanda', 'structured', or 'unstructured' (default: 'wanda')
            export_onnx: Whether to export to ONNX (enabled by default for correct workflow)
            num_calibration_samples: Number of WikiText-2 samples for Wanda calibration
            device: Device to run on ('cpu' or 'cuda')
            skip_existing: Skip steps if checkpoints exist (default: True)

        Returns:
            Dictionary with results and metrics
        """
        results = {
            'model_name': model_name,
            'sparsity_target': sparsity,
            'pruning_method': pruning_method,
            'success': False,
            'pytorch_model_path': None,
            'onnx_model_path': None,
            'sparsity_stats': None,
            'errors': [],
            'checkpoints_used': []
        }

        # Skip multimodal models
        if model_name in self.MULTIMODAL_MODELS:
            logger.warning(f"Skipping {model_name} - multimodal models require special handling")
            results['errors'].append("Multimodal model not supported")
            return results

        # Check for existing checkpoints
        output_dir = self.get_pruned_output_dir(model_name, sparsity)
        onnx_output_dir = output_dir.parent / f"{output_dir.name}_onnx"

        # Check for pruned model - look for config.json and any .safetensors or .bin files
        pruned_exists = False
        if output_dir.exists():
            has_config = (output_dir / "config.json").exists()
            has_model_files = (
                list(output_dir.glob("*.safetensors")) or
                list(output_dir.glob("*.bin")) or
                list(output_dir.glob("pytorch_model*.bin"))
            )
            # Also check that model files are reasonably sized (>1MB to avoid incomplete saves)
            if has_config and has_model_files:
                total_size = sum(f.stat().st_size for f in has_model_files)
                if total_size > 1_000_000:  # At least 1MB
                    pruned_exists = True
                    logger.debug(f"Found pruned checkpoint with {len(has_model_files)} model files, total size: {total_size / (1024**3):.2f} GB")

        onnx_exists = onnx_output_dir.exists() and (onnx_output_dir / "model.onnx").exists()

        # Step 1-5: Load, prune, and save PyTorch model
        if skip_existing and pruned_exists:
            logger.info("="*80)
            logger.info("CHECKPOINT DETECTED: Pruned model already exists")
            logger.info("="*80)
            logger.info(f"✓ Found existing pruned model at: {output_dir}")
            logger.info("  Skipping pruning step (set skip_existing=False to force rerun)")

            # Try to load sparsity stats from saved JSON (fast!)
            sparsity_stats_path = output_dir / "sparsity_stats.json"
            if sparsity_stats_path.exists():
                try:
                    with open(sparsity_stats_path, 'r') as f:
                        final_sparsity = json.load(f)
                    results['sparsity_stats'] = final_sparsity
                    logger.info(f"✓ Loaded sparsity stats from cache - Sparsity: {final_sparsity['overall_sparsity']*100:.2f}%")
                    logger.info(f"  (Avoided loading {final_sparsity['total_params']:,} parameters - much faster!)")
                except Exception as e:
                    logger.warning(f"Could not load sparsity stats JSON: {e}")
                    # Fall back to estimated stats
                    results['sparsity_stats'] = {
                        'overall_sparsity': sparsity,
                        'total_params': 0,
                        'zero_params': 0,
                        'compression_ratio': 1 - sparsity
                    }
                    logger.info(f"✓ Using target sparsity: {sparsity*100:.0f}%")
            else:
                # No saved stats, use target sparsity
                logger.info(f"✓ No sparsity stats cache found, using target sparsity: {sparsity*100:.0f}%")
                results['sparsity_stats'] = {
                    'overall_sparsity': sparsity,
                    'total_params': 0,
                    'zero_params': 0,
                    'compression_ratio': 1 - sparsity
                }

            results['pytorch_model_path'] = str(output_dir)
            results['checkpoints_used'].append('pruned_pytorch')
        else:
            # Run full pruning pipeline
            logger.info("Running pruning pipeline (no checkpoint found or skip_existing=False)")

            # Step 1: Load model
            model_data = self.load_model(model_name)
            if model_data is None:
                results['errors'].append("Failed to load model")
                return results

            model, tokenizer = model_data

            # Step 2: Calculate initial sparsity
            initial_sparsity = self.calculate_sparsity(model)
            logger.info(f"Initial sparsity: {initial_sparsity['overall_sparsity']*100:.2f}%")

            # Step 3: Apply pruning
            try:
                if pruning_method == "wanda":
                    logger.info(f"Applying Wanda pruning with WikiText-2 calibration ({num_calibration_samples} samples)...")
                    model = self.apply_wanda_pruning(
                        model,
                        tokenizer,
                        sparsity=sparsity,
                        num_calibration_samples=num_calibration_samples,
                        device=device
                    )
                elif pruning_method == "unstructured":
                    model = self.apply_unstructured_pruning(model, sparsity)
                elif pruning_method == "structured":
                    model = self.apply_structured_pruning(model, sparsity)
                else:
                    logger.error(f"Unknown pruning method: {pruning_method}")
                    results['errors'].append(f"Unknown pruning method: {pruning_method}")
                    return results
            except Exception as e:
                logger.error(f"Pruning failed: {str(e)}")
                results['errors'].append(f"Pruning failed: {str(e)}")
                return results

            # Step 4: Calculate final sparsity
            final_sparsity = self.calculate_sparsity(model)
            results['sparsity_stats'] = final_sparsity

            # Step 5: Save pruned model
            logger.info(f"\nSaving pruned model to: {output_dir}")
            self.save_pytorch_model(model, tokenizer, output_dir)
            results['pytorch_model_path'] = str(output_dir)

            # Save sparsity stats for checkpoint reuse
            sparsity_stats_path = output_dir / "sparsity_stats.json"
            try:
                with open(sparsity_stats_path, 'w') as f:
                    json.dump(final_sparsity, f, indent=2)
                logger.info(f"✓ Sparsity stats saved to {sparsity_stats_path}")
            except Exception as e:
                logger.warning(f"Could not save sparsity stats: {e}")

        # Step 6: Export to ONNX (required for quantization workflow)
        if export_onnx:
            if skip_existing and onnx_exists:
                logger.info("\n" + "="*80)
                logger.info("CHECKPOINT DETECTED: ONNX model already exists")
                logger.info("="*80)
                logger.info(f"✓ Found existing ONNX model at: {onnx_output_dir}")
                logger.info("  Skipping ONNX export (set skip_existing=False to force rerun)")

                results['onnx_model_path'] = str(onnx_output_dir)
                results['checkpoints_used'].append('onnx')
            else:
                logger.info("\n" + "-"*80)
                logger.info("Step 6: Exporting pruned model to ONNX...")
                logger.info("This will take 20-30 minutes (required for quantization workflow)")
                logger.info("-"*80 + "\n")

                onnx_success = self.export_to_onnx(output_dir)
                if onnx_success:
                    results['onnx_model_path'] = str(onnx_output_dir)
                    logger.info(f"✓ ONNX export successful: {onnx_output_dir}")
                else:
                    results['errors'].append("ONNX export failed")
        else:
            logger.info("\nSkipping ONNX export. Set export_onnx=True for full workflow.")

        # Step 7: Export to GGUF (optional, requires llama.cpp)
        if export_gguf and results['pytorch_model_path']:
            logger.info("\n" + "-"*80)
            logger.info(f"Step 7: Exporting pruned model to GGUF ({gguf_quantization.upper()})...")
            logger.info("-"*80)
            gguf_success = self.export_to_gguf(
                Path(results['pytorch_model_path']),
                quantization=gguf_quantization,
            )
            if gguf_success:
                quant_lower = gguf_quantization.lower()
                if quant_lower == "f16":
                    precision_dir = self.quantized_dir / "FP16"
                elif quant_lower in ("q8_0", "q8_1"):
                    precision_dir = self.quantized_dir / "INT8"
                else:
                    precision_dir = self.quantized_dir / "INT4"
                model_name = Path(results['pytorch_model_path']).name
                results['gguf_model_path'] = str(precision_dir / model_name / f"model_{gguf_quantization}.gguf")
                logger.info(f"✓ GGUF export successful: {results['gguf_model_path']}")
            else:
                results['errors'].append("GGUF export failed")

        results['success'] = True

        # Summary
        logger.info("\n" + "="*80)
        logger.info("PRUNING SUMMARY")
        logger.info("="*80)
        logger.info(f"Target sparsity: {sparsity*100:.0f}%")
        if results['sparsity_stats']:
            logger.info(f"Achieved sparsity: {results['sparsity_stats']['overall_sparsity']*100:.2f}%")
            if results['sparsity_stats']['total_params'] > 0:
                logger.info(f"Zero parameters: {results['sparsity_stats']['zero_params']:,} / {results['sparsity_stats']['total_params']:,}")
        logger.info(f"PyTorch model saved to: {output_dir}")
        if export_onnx and results['onnx_model_path']:
            logger.info(f"ONNX model saved to: {results['onnx_model_path']}")
        logger.info("="*80)

        return results


def main():
    """Example usage"""
    # Configuration
    OUTPUT_DIR = "/Users/sashalai/Documents/UW/26sp/eep564/Final Project/model_quantization/model_quantization_outputs"

    MODELS = [
        'google/gemma-2-2b',
        'meta-llama/Llama-3.2-3B-Instruct',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'lmms-lab/LLaVA-OneVision-1.5-4B-Instruct'
    ]

    # Pruning configurations
    SPARSITY_LEVELS = [0.3, 0.5]  # 30% and 50% sparsity

    # Initialize pruner
    pruner = ModelPruner(OUTPUT_DIR)

    # Process each model
    all_results = []
    for model_name in MODELS:
        for sparsity in SPARSITY_LEVELS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {model_name} with {sparsity*100:.0f}% sparsity")
            logger.info(f"{'='*60}")

            results = pruner.process_model(
                model_name,
                sparsity=sparsity,
                pruning_method="wanda",  # Wanda pruning with WikiText-2 calibration
                export_onnx=True,  # Required for correct workflow (20-30 min)
                num_calibration_samples=128,
                device="cpu"
            )

            all_results.append(results)

            # Print summary
            logger.info(f"\nResults for {model_name} ({sparsity*100:.0f}% sparsity):")
            logger.info(f"  Success: {'✓' if results['success'] else '✗'}")
            if results['sparsity_stats']:
                logger.info(f"  Achieved Sparsity: {results['sparsity_stats']['overall_sparsity']*100:.2f}%")
            if results['errors']:
                logger.warning(f"  Errors: {', '.join(results['errors'])}")

    # Save results summary
    summary_path = Path(OUTPUT_DIR) / "pruning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nPruning summary saved to {summary_path}")


if __name__ == "__main__":
    main()
