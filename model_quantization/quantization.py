"""
ONNX Export and Quantization Module

This module handles:
1. Exporting HuggingFace models to ONNX format
2. Applying INT8 and INT4 quantization to ONNX models
3. Saving quantized models
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handles ONNX export and quantization of models"""

    # Models that require special handling
    MULTIMODAL_MODELS = ['lmms-lab/LLaVA-OneVision-1.5-4B-Instruct']

    # Map to pre-exported ONNX models on HuggingFace (faster than exporting)
    ONNX_MODEL_MAPPING = {
        'google/gemma-4-E2B-it': 'onnx-community/gemma-4-E2B-it-ONNX',
        #'meta-llama/Llama-3.2-3B-Instruct': 'onnx-community/Llama-3.2-3B-Instruct-ONNX',
    }

    def __init__(
        self,
        base_output_dir: str,
        use_preexported: bool = True,
        enable_preprocessing: bool = True,
        preprocessing_size_threshold_gb: float = 5.0
    ):
        """
        Initialize the quantizer

        Args:
            base_output_dir: Base directory for all model outputs
            use_preexported: Use pre-exported ONNX models from HF (faster)
            enable_preprocessing: Enable preprocessing (shape inference, optimization) before quantization
            preprocessing_size_threshold_gb: Skip preprocessing for models larger than this (GB).
                                            Default 5.0 GB to avoid memory issues.
        """
        self.base_output_dir = Path(base_output_dir)
        self.onnx_base_dir = self.base_output_dir / "onnx"
        self.quantized_dir = self.base_output_dir / "quantized"
        self.onnx_base_dir.mkdir(parents=True, exist_ok=True)
        self.quantized_dir.mkdir(parents=True, exist_ok=True)
        self.use_preexported = use_preexported
        self.enable_preprocessing = enable_preprocessing
        self.preprocessing_size_threshold_gb = preprocessing_size_threshold_gb

    def get_model_output_dir(self, model_name: str) -> Path:
        """Get the output directory for a specific model"""
        model_short_name = model_name.split('/')[-1]
        return self.onnx_base_dir / model_short_name

    def check_onnx_exists(self, model_name: str) -> bool:
        """
        Check if ONNX base model already exists

        Args:
            model_name: HuggingFace model name

        Returns:
            True if ONNX model exists, False otherwise
        """
        output_dir = self.get_model_output_dir(model_name)

        # Check for common ONNX files
        if output_dir.exists():
            onnx_files = list(output_dir.glob("*.onnx"))
            if onnx_files:
                logger.info(f"Found existing ONNX model for {model_name} at {output_dir}")
                return True

        return False

    def download_preexported_onnx(self, model_name: str) -> bool:
        """
        Download pre-exported ONNX model from HuggingFace (faster than exporting)

        Args:
            model_name: HuggingFace model name

        Returns:
            True if download successful, False otherwise
        """
        onnx_model_name = self.ONNX_MODEL_MAPPING.get(model_name)

        if not onnx_model_name:
            logger.warning(f"No pre-exported ONNX version available for {model_name}")
            logger.info("Available pre-exported models:")
            for k, v in self.ONNX_MODEL_MAPPING.items():
                logger.info(f"  {k} → {v}")
            return False

        output_dir = self.get_model_output_dir(model_name)

        # Check if already exists
        if self.check_onnx_exists(model_name):
            logger.info(f"ONNX model already exists, skipping download")
            return True

        logger.info(f"Downloading pre-exported ONNX model from {onnx_model_name}...")
        logger.info("This may take 2-5 minutes depending on model size...")

        try:
            from huggingface_hub import snapshot_download

            # Download the entire model repository
            downloaded_path = snapshot_download(
                repo_id=onnx_model_name,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False
            )

            logger.info(f"✓ Successfully downloaded ONNX model to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to download ONNX model: {str(e)}")
            logger.error("You may need to install huggingface_hub: pip install huggingface_hub")
            return False

    def export_to_onnx(
        self,
        model_name: str,
        task: str = "text-generation",
        batch_size: int = 1,
        device: str = "cpu",
        num_threads: int = 4
    ) -> bool:
        """
        Export a HuggingFace model to ONNX format using optimum-cli

        Args:
            model_name: HuggingFace model name
            task: Model task (e.g., 'text-generation')
            batch_size: Batch size for export
            device: Device to use (cpu/cuda)
            num_threads: Number of threads for CPU operations

        Returns:
            True if export successful, False otherwise
        """
        # Skip multimodal models
        if model_name in self.MULTIMODAL_MODELS:
            logger.warning(f"Skipping {model_name} - multimodal models require special handling")
            return False

        output_dir = self.get_model_output_dir(model_name)

        # Check if already exists
        if self.check_onnx_exists(model_name):
            logger.info(f"ONNX model already exists for {model_name}, skipping export")
            return True

        logger.info(f"Exporting {model_name} to ONNX format...")

        # Prepare environment variables
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(num_threads)
        env['MKL_NUM_THREADS'] = str(num_threads)

        # Build optimum-cli command
        cmd = [
            'optimum-cli', 'export', 'onnx',
            '-m', model_name,
            '--task', task,
            '--device', device,
            '--batch_size', str(batch_size),
            '--no-post-process',
            '--trust-remote-code',
            str(output_dir)
        ]

        try:
            # Run export command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Successfully exported {model_name} to ONNX")
            logger.debug(f"Export output: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export {model_name} to ONNX")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting {model_name}: {str(e)}")
            return False

    def cleanup_intermediate_files(
        self,
        keep_onnx: bool = True,
        clean_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up intermediate files and cache to free up disk space.

        Args:
            keep_onnx: Keep base ONNX models (default: True, needed for re-quantization)
            clean_cache: Clean HuggingFace cache (default: True)

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'space_freed_mb': 0,
            'files_deleted': 0,
            'directories_cleaned': []
        }

        logger.info("="*80)
        logger.info("CLEANING UP INTERMEDIATE FILES")
        logger.info("="*80)

        # Clean base ONNX models (be careful!)
        if not keep_onnx and self.onnx_base_dir.exists():
            logger.warning("⚠️  Removing base ONNX models (you'll need to re-export to re-quantize)")
            try:
                for item in self.onnx_base_dir.iterdir():
                    if item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        stats['space_freed_mb'] += size / (1024 * 1024)
                        stats['files_deleted'] += 1
                        stats['directories_cleaned'].append(str(item))
                logger.info(f"✓ Cleaned ONNX directory")
            except Exception as e:
                logger.error(f"Error cleaning ONNX directory: {e}")

        # Clean HuggingFace cache
        if clean_cache:
            hf_cache_dir = Path.home() / ".cache" / "huggingface"
            if hf_cache_dir.exists():
                logger.info(f"Cleaning HuggingFace cache: {hf_cache_dir}")
                logger.info("Note: This only removes cached model downloads, not installed models")
                try:
                    # Get size before cleaning
                    cache_size = sum(f.stat().st_size for f in hf_cache_dir.rglob('*') if f.is_file())

                    # Clean hub cache (downloaded models)
                    hub_cache = hf_cache_dir / "hub"
                    if hub_cache.exists():
                        # Remove only blobs and snapshots (cached downloads)
                        for subdir in ['blobs', 'snapshots']:
                            target = hub_cache / subdir
                            if target.exists():
                                shutil.rmtree(target)
                                stats['directories_cleaned'].append(str(target))

                    # Get size after cleaning
                    cache_size_after = sum(f.stat().st_size for f in hf_cache_dir.rglob('*') if f.is_file())
                    freed = (cache_size - cache_size_after) / (1024 * 1024)
                    stats['space_freed_mb'] += freed

                    logger.info(f"✓ Cleaned HuggingFace cache (freed {freed:.2f} MB)")
                except Exception as e:
                    logger.error(f"Error cleaning HuggingFace cache: {e}")

        logger.info("="*80)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*80)
        logger.info(f"Space freed: {stats['space_freed_mb']:.2f} MB ({stats['space_freed_mb']/1024:.2f} GB)")
        logger.info(f"Items deleted: {stats['files_deleted']}")
        logger.info(f"Directories cleaned: {len(stats['directories_cleaned'])}")

        return stats

    def quantize_with_genai_builder(
        self,
        model_name: str,
        quantization_type: str = "int4",
        execution_provider: str = "cpu"
    ) -> Optional[Path]:
        """
        Quantize using onnxruntime-genai builder (ONLY for original/non-pruned models)

        IMPORTANT: This method CANNOT be used with pruned models!
        The genai builder expects original PyTorch models and has hardcoded
        architecture templates (LlamaForCausalLM, Gemma2ForCausalLM, etc.).
        It cannot accept custom ONNX files as input.

        Use this ONLY for baseline comparisons with non-pruned models.
        For pruned models, use quantize_onnx_from_path() instead.

        Args:
            model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3.2-3B-Instruct')
                        Must be original model, NOT a pruned model path
            quantization_type: 'int4' or 'int8'
            execution_provider: 'cpu', 'cuda', or 'directml'

        Returns:
            Path to quantized model directory if successful, None otherwise
        """
        try:
            import subprocess

            model_short_name = model_name.split('/')[-1]
            output_dir = self.quantized_dir / f"{model_short_name}_{quantization_type}_genai"

            logger.info(f"Using onnxruntime-genai builder for {quantization_type.upper()} quantization...")
            logger.info("This is the recommended approach for LLM quantization")

            # Build command
            cmd = [
                "python", "-m", "onnxruntime_genai.models.builder",
                "-m", model_name,
                "-o", str(output_dir),
                "-p", quantization_type,
                "-e", execution_provider,
                "--extra_options", "int4_block_size=128"
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            logger.info("This may take 20-40 minutes...")

            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"✓ GenAI builder completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            return output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"GenAI builder failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error using GenAI builder: {str(e)}")
            return None

    def quantize_onnx_from_path(
        self,
        onnx_model_path: Path,
        quantization_type: str = "int8",
        skip_existing: bool = True,
        apply_preprocessing: bool = None
    ) -> Optional[Path]:
        """
        Quantize an existing ONNX model (USE THIS FOR PRUNED MODELS)

        This method accepts an already-exported ONNX file and applies quantization.
        It works with both original and pruned ONNX models.

        For pruned models, this is the ONLY option because onnxruntime-genai
        builder cannot process custom-pruned architectures.

        Args:
            onnx_model_path: Path to ONNX model directory or file
            quantization_type: Type of quantization ('int8' or 'int4')
            skip_existing: Skip if quantized model already exists (default: True)
            apply_preprocessing: Apply preprocessing before quantization (None=use default, True/False=override)

        Returns:
            Path to quantized model directory if successful, None otherwise
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            if quantization_type.lower() == "int4":
                # INT4 requires specialized quantizer
                try:
                    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
                except ImportError:
                    try:
                        from onnxruntime.quantization import MatMulNBitsQuantizer as MatMulNBitsQuantizer
                    except ImportError:
                        logger.error("INT4 quantization requires onnxruntime >= 1.16.0")
                        logger.error("Install with: pip install onnxruntime>=1.16.0")
                        return None
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return None

        # Handle both directory and file paths
        if onnx_model_path.is_file():
            output_dir = onnx_model_path.parent
            model_path = onnx_model_path
        else:
            output_dir = onnx_model_path
            # Find the main model file
            onnx_files = list(output_dir.glob("model*.onnx"))
            if not onnx_files:
                onnx_files = list(output_dir.glob("*.onnx"))

            if not onnx_files:
                logger.error(f"No ONNX files found in {output_dir}")
                return None
            model_path = onnx_files[0]

        if not output_dir.exists():
            logger.error(f"ONNX model path not found: {output_dir}")
            return None

        # Create quantized output directory in the quantized/ folder
        # Extract model name from the ONNX directory
        model_name = output_dir.name.replace("_onnx", "")
        quant_dir = self.quantized_dir / f"{model_name}_{quantization_type}"
        quant_model_path = quant_dir / model_path.name

        # Check for existing quantized model
        if skip_existing and quant_model_path.exists():
            logger.info("="*80)
            logger.info(f"CHECKPOINT DETECTED: Quantized {quantization_type.upper()} model already exists")
            logger.info("="*80)
            logger.info(f"✓ Found existing quantized model at: {quant_dir}")
            logger.info("  Skipping quantization (set skip_existing=False to force rerun)")

            # Get file sizes for reporting
            try:
                original_size = model_path.stat().st_size / (1024 * 1024)
                quantized_size = quant_model_path.stat().st_size / (1024 * 1024)
                compression_ratio = (1 - quantized_size / original_size) * 100
                logger.info(f"  Original size: {original_size:.2f} MB")
                logger.info(f"  Quantized size: {quantized_size:.2f} MB")
                logger.info(f"  Compression: {compression_ratio:.1f}%")
            except Exception:
                pass

            return quant_dir

        quant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Quantizing {output_dir.name} to {quantization_type}...")

        try:
            # Check if model uses external data format (large models)
            model_data_file = model_path.parent / f"{model_path.stem}.onnx_data"
            uses_external_data = model_data_file.exists()

            if uses_external_data:
                model_size_gb = model_data_file.stat().st_size / (1024**3)
                logger.info(f"⚠️  LARGE MODEL DETECTED: {model_size_gb:.1f} GB with external data")
                logger.info("="*80)
                logger.info("IMPORTANT: Quantization of large models (10+ GB) is VERY slow")
                logger.info("Expected time: 30-60 minutes per quantization type")
                logger.info("The process may appear stuck but is working - please be patient!")
                logger.info("="*80)
                estimated_minutes = int(model_size_gb * 3)  # ~3 minutes per GB
            else:
                logger.info("Applying quantization (this may take 2-5 minutes)...")
                estimated_minutes = 5

            import time
            start_time = time.time()

            # Apply preprocessing if enabled (inline, before quantization)
            should_preprocess = apply_preprocessing if apply_preprocessing is not None else self.enable_preprocessing

            if should_preprocess:
                # Skip preprocessing for very large models - too memory intensive
                model_size_gb = model_path.stat().st_size / (1024**3) if not uses_external_data else model_size_gb

                if model_size_gb > self.preprocessing_size_threshold_gb:
                    logger.warning("="*80)
                    logger.warning(f"⚠️  VERY LARGE MODEL: {model_size_gb:.1f} GB")
                    logger.warning(f"Skipping preprocessing (threshold: {self.preprocessing_size_threshold_gb} GB)")
                    logger.warning("Shape inference on very large models can crash due to memory consumption")
                    logger.warning("Quantization will proceed without preprocessing")
                    logger.warning("To enable preprocessing, increase preprocessing_size_threshold_gb parameter")
                    logger.warning("="*80)
                else:
                    try:
                        from onnxruntime.quantization.shape_inference import quant_pre_process

                        preprocessed_path = quant_dir / f"{model_path.stem}_preprocessed.onnx"
                        logger.info("Running pre-processing (shape inference + optimization)...")

                        # For models >2GB, skip optimization but run shape inference
                        if uses_external_data:
                            logger.info("  - Skipping optimization (model >2GB, ONNX Runtime limitation)")
                            logger.info("  - Running symbolic shape inference (helps with transformers)")
                            logger.info("  - Running ONNX shape inference (determines tensor shapes)")
                        else:
                            logger.info("  - Symbolic shape inference (helps with transformers)")
                            logger.info("  - Model optimization (fuses operators, removes redundancies)")
                            logger.info("  - ONNX shape inference (determines tensor shapes)")

                        logger.info(f"  Model size: {model_size_gb:.2f} GB (threshold: {self.preprocessing_size_threshold_gb} GB)")
                        logger.info("  Note: This may take 5-15 minutes for large models...")

                        preprocess_start = time.time()

                        quant_pre_process(
                            input_model_path=str(model_path),
                            output_model_path=str(preprocessed_path),
                            skip_optimization=uses_external_data,  # optimization can't output >2GB
                            skip_onnx_shape=False,
                            skip_symbolic_shape=False,
                        )

                        preprocess_elapsed = (time.time() - preprocess_start) / 60
                        logger.info(f"✓ Pre-processing complete in {preprocess_elapsed:.1f} minutes")

                        # Use preprocessed model going forward
                        model_path = preprocessed_path
                        logger.info(f"  Using preprocessed model for quantization")

                    except ImportError:
                        logger.warning("onnxruntime.quantization not available, skipping preprocessing")
                    except MemoryError as e:
                        logger.error("="*80)
                        logger.error("MEMORY ERROR during preprocessing!")
                        logger.error(f"Model size: {model_size_gb:.2f} GB is too large for available memory")
                        logger.error("Recommendation: Reduce preprocessing_size_threshold_gb or disable preprocessing")
                        logger.error("Continuing with original model (no preprocessing)")
                        logger.error("="*80)
                    except KeyboardInterrupt:
                        logger.warning("Preprocessing interrupted by user")
                        logger.warning("Continuing with original model (no preprocessing)")
                        raise  # Re-raise to allow proper cleanup
                    except Exception as e:
                        logger.error(f"Preprocessing failed with error: {type(e).__name__}: {str(e)}")
                        logger.warning("Continuing with original model (no preprocessing)")
                        # Don't fail the whole process, just skip preprocessing
                        import traceback
                        logger.debug(f"Full traceback:\n{traceback.format_exc()}")

            # Apply quantization based on type
            logger.info(f"Starting quantization... (estimated: {estimated_minutes} minutes)")

            if quantization_type.lower() == "int8":
                # INT8 Dynamic Quantization
                # Using QInt8 (signed) which is better for LLM weights centered around zero
                logger.info("Applying INT8 dynamic quantization (signed, better for LLM weights)...")
                quantize_dynamic(
                    model_input=str(model_path),
                    model_output=str(quant_model_path),
                    weight_type=QuantType.QInt8,  # Signed INT8 for zero-centered weights
                    use_external_data_format=uses_external_data  # Critical for >2GB models
                )
            # elif quantization_type.lower() == "int4":
            #     # INT4 Block-wise Quantization using MatMulNBitsQuantizer
            #     # TODO: Enable after INT8 testing is complete
            #     # This uses weight-only 4-bit quantization optimized for LLMs
            #     logger.info("Applying INT4 block-wise quantization...")
            #     logger.info("Using MatMulNBitsQuantizer with LLM-optimized parameters:")
            #     logger.info("  - block_size=128 (larger blocks preserve model quality)")
            #     logger.info("  - is_symmetric=False (asymmetric for better LLM accuracy)")
            #     logger.info("  - accuracy_level=4 (highest optimization level)")
            #
            #
            #     quantizer = MatMulNBitsQuantizer(
            #         model=str(model_path),  # ModelProto object or a path string
            #         bits=4,                  # 4-bit quantization
            #         block_size=128,          # Larger blocks (default, good for LLMs)
            #         is_symmetric=False,      # Asymmetric quantization (better accuracy)
            #         accuracy_level=4,        # Highest optimization level
            #         nodes_to_exclude=None    # Quantize all MatMul nodes
            #     )
            #
            #     # Process the model (applies quantization)
            #     quantizer.process()
            #
            #     # Save the quantized model
            #     # The quantizer's save method handles external data automatically
            #     quantizer.model.save_model_to_file(
            #         str(quant_model_path),
            #         use_external_data_format=uses_external_data
            #     )
            else:
                logger.error(f"Unsupported quantization type: {quantization_type}")
                logger.error(f"Currently only INT8 is enabled. INT4 is disabled for testing.")
                return None

            elapsed_time = (time.time() - start_time) / 60
            logger.info(f"✓ Quantization completed in {elapsed_time:.1f} minutes")

            # Copy config files
            import shutil
            for config_file in ['config.json', 'generation_config.json', 'tokenizer_config.json',
                               'tokenizer.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt']:
                src = output_dir / config_file
                if src.exists():
                    dst = quant_dir / config_file
                    shutil.copy2(src, dst)

            # Get file sizes
            original_size = model_path.stat().st_size / (1024 * 1024)
            quantized_size = quant_model_path.stat().st_size / (1024 * 1024)
            compression_ratio = (1 - quantized_size / original_size) * 100

            logger.info(f"✓ Successfully quantized to {quantization_type}")
            logger.info(f"  Original size: {original_size:.2f} MB")
            logger.info(f"  Quantized size: {quantized_size:.2f} MB")
            logger.info(f"  Compression: {compression_ratio:.1f}%")

            return quant_dir

        except Exception as e:
            logger.error(f"Failed to quantize: {str(e)}")
            return None

    def quantize_onnx_model(
        self,
        model_name: str,
        quantization_type: str = "int8"
    ) -> Optional[Path]:
        """
        Quantize an ONNX model using onnxruntime quantization (wrapper for backward compatibility)

        Args:
            model_name: HuggingFace model name
            quantization_type: Type of quantization ('int8' or 'int4')

        Returns:
            Path to quantized model if successful, None otherwise
        """
        output_dir = self.get_model_output_dir(model_name)

        if not output_dir.exists():
            logger.error(f"ONNX model not found for {model_name}. Export first.")
            return None

        # Use the new path-based quantization method
        return self.quantize_onnx_from_path(output_dir, quantization_type)

    def process_model(
        self,
        model_name: str,
        apply_int8: bool = True,
        apply_int4: bool = True,
        force_reexport: bool = False,
        apply_preprocessing: bool = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: export to ONNX, preprocess, and apply quantization

        Args:
            model_name: HuggingFace model name
            apply_int8: Whether to apply INT8 quantization
            apply_int4: Whether to apply INT4 quantization
            force_reexport: Force re-export even if ONNX exists
            apply_preprocessing: Apply preprocessing before quantization (None=use default setting)

        Returns:
            Dictionary with results and paths
        """
        results = {
            'model_name': model_name,
            'onnx_export': False,
            'int8_quantization': None,
            'int4_quantization': None,
            'errors': []
        }

        # Step 1: Get ONNX model (download pre-exported or export)
        if force_reexport or not self.check_onnx_exists(model_name):
            success = False

            # Try downloading pre-exported ONNX first (faster!)
            if self.use_preexported:
                logger.info("Attempting to download pre-exported ONNX model...")
                success = self.download_preexported_onnx(model_name)

            # Fall back to export if download fails or disabled
            if not success:
                logger.info("Falling back to ONNX export (this will take longer)...")
                success = self.export_to_onnx(model_name)

            results['onnx_export'] = success

            if not success:
                results['errors'].append("ONNX export/download failed")
                return results
        else:
            results['onnx_export'] = True

        # Step 2: Apply INT8 quantization
        if apply_int8:
            output_dir = self.get_model_output_dir(model_name)
            int8_path = self.quantize_onnx_from_path(output_dir, "int8", apply_preprocessing=apply_preprocessing)
            results['int8_quantization'] = str(int8_path) if int8_path else None
            if not int8_path:
                results['errors'].append("INT8 quantization failed")

        # Step 3: Apply INT4 quantization
        if apply_int4:
            output_dir = self.get_model_output_dir(model_name)
            int4_path = self.quantize_onnx_from_path(output_dir, "int4", apply_preprocessing=apply_preprocessing)
            results['int4_quantization'] = str(int4_path) if int4_path else None
            if not int4_path:
                results['errors'].append("INT4 quantization failed")

        return results


def main():
    """Example usage with preprocessing and cleanup"""
    # Configuration
    OUTPUT_DIR = "/Users/sashalai/Documents/UW/26sp/eep564/Final Project/model_quantization/model_quantization_outputs"

    MODELS = [
        'google/gemma-2-2b',
        'meta-llama/Llama-3.2-3B-Instruct',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'lmms-lab/LLaVA-OneVision-1.5-4B-Instruct'
    ]

    # Initialize quantizer with preprocessing enabled (default)
    # Set enable_preprocessing=False to skip preprocessing globally
    quantizer = ModelQuantizer(OUTPUT_DIR, enable_preprocessing=True)

    # Process each model
    all_results = []
    for model_name in MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*60}")

        results = quantizer.process_model(
            model_name,
            apply_int8=True,
            apply_int4=True,
            # apply_preprocessing=True  # Override per-model if needed
        )

        all_results.append(results)

        # Print summary
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  ONNX Export: {'✓' if results['onnx_export'] else '✗'}")
        logger.info(f"  INT8 Quantization: {'✓' if results['int8_quantization'] else '✗'}")
        logger.info(f"  INT4 Quantization: {'✓' if results['int4_quantization'] else '✗'}")
        if results['errors']:
            logger.warning(f"  Errors: {', '.join(results['errors'])}")

    # Save results summary
    summary_path = Path(OUTPUT_DIR) / "quantization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nQuantization summary saved to {summary_path}")

    # Optional: Clean up intermediate files to free space
    logger.info("\n" + "="*60)
    logger.info("CLEANUP STEP (Optional)")
    logger.info("="*60)
    cleanup_stats = quantizer.cleanup_intermediate_files(
        keep_onnx=True,   # Keep base ONNX models for re-quantization
        clean_cache=True  # Clean HuggingFace cache
    )

    logger.info(f"\nTotal space freed: {cleanup_stats['space_freed_mb']/1024:.2f} GB")


if __name__ == "__main__":
    main()
