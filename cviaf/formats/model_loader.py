"""
Model loading utilities for ONNX and PyTorch/TorchScript models.
Provides unified interface for feature extraction and inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ModelWrapper(ABC):
    """Abstract base for all model wrappers."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_digest = ""
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model from disk."""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference. Input shape depends on model."""
        pass

    @abstractmethod
    def get_intermediate_features(self, input_data: np.ndarray,
                                  layer_name: str = "") -> np.ndarray:
        """Extract intermediate layer activations (white-box only)."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters as numpy arrays (white-box only)."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Tuple:
        """Expected input shape."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""
        pass

    def compute_digest(self) -> str:
        """Compute SHA-256 digest of the model file."""
        h = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self.model_digest = h.hexdigest()
        return self.model_digest

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_path": self.model_path,
            "model_digest": self.model_digest or self.compute_digest(),
            "format": self.__class__.__name__,
            "loaded": self._loaded,
        }


class ONNXModelWrapper(ModelWrapper):
    """Wrapper for ONNX models using onnxruntime."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._session = None
        self._input_name = ""
        self._input_shape: Tuple = ()
        self._output_names: List[str] = []

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime"
            )

        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        shape = inputs[0].shape
        # Replace dynamic dims with defaults
        self._input_shape = tuple(
            d if isinstance(d, int) else 1 for d in shape
        )
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.compute_digest()
        self._loaded = True

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self.load()
        results = self._session.run(
            self._output_names,
            {self._input_name: input_data.astype(np.float32)}
        )
        return results[0]

    def get_intermediate_features(self, input_data: np.ndarray,
                                  layer_name: str = "") -> np.ndarray:
        """
        ONNX intermediate feature extraction.
        If layer_name is provided and the model exposes that tensor, extract it.
        Falls back to last output.
        """
        if not self._loaded:
            self.load()

        if layer_name and layer_name in [o.name for o in self._session.get_outputs()]:
            results = self._session.run(
                [layer_name],
                {self._input_name: input_data.astype(np.float32)}
            )
            return results[0]

        # Fall back to running all outputs and returning the last one
        results = self._session.run(
            None,
            {self._input_name: input_data.astype(np.float32)}
        )
        return results[-1]

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Extract ONNX model parameters (initializers)."""
        try:
            import onnx
            model = onnx.load(self.model_path)
            params = {}
            for initializer in model.graph.initializer:
                params[initializer.name] = np.frombuffer(
                    initializer.raw_data,
                    dtype=np.float32
                ).copy()
            return params
        except ImportError:
            raise ImportError(
                "onnx package required for parameter extraction. "
                "Install with: pip install onnx"
            )

    @property
    def input_shape(self) -> Tuple:
        if not self._loaded:
            self.load()
        return self._input_shape

    @property
    def num_classes(self) -> int:
        if not self._loaded:
            self.load()
        outputs = self._session.get_outputs()
        if outputs:
            shape = outputs[0].shape
            if shape and len(shape) >= 2:
                last_dim = shape[-1]
                if isinstance(last_dim, int):
                    return last_dim
        return -1


class PyTorchModelWrapper(ModelWrapper):
    """Wrapper for PyTorch (.pt, .pth) and TorchScript (.pt) models."""

    def __init__(self, model_path: str, model_class=None, num_classes_hint: int = -1):
        super().__init__(model_path)
        self._model = None
        self._is_torchscript = False
        self._model_class = model_class
        self._num_classes = num_classes_hint
        self._hooks = {}
        self._features = {}

    def load(self) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for .pt/.pth models. "
                "Install with: pip install torch"
            )

        # Try loading as TorchScript first
        try:
            self._model = torch.jit.load(self.model_path, map_location="cpu")
            self._is_torchscript = True
        except Exception:
            # Fall back to state_dict loading
            if self._model_class is not None:
                self._model = self._model_class()
                state_dict = torch.load(
                    self.model_path, map_location="cpu", weights_only=True
                )
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self._model.load_state_dict(state_dict, strict=False)
            else:
                # Try loading the entire object
                self._model = torch.load(
                    self.model_path, map_location="cpu", weights_only=False
                )

        if hasattr(self._model, "eval"):
            self._model.eval()
        self.compute_digest()
        self._loaded = True

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        import torch
        if not self._loaded:
            self.load()
        with torch.no_grad():
            tensor = torch.from_numpy(input_data).float()
            output = self._model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.cpu().numpy()

    def _hook_fn(self, name):
        def hook(module, input, output):
            if hasattr(output, 'detach'):
                self._features[name] = output.detach().cpu().numpy()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if hasattr(output[0], 'detach'):
                    self._features[name] = output[0].detach().cpu().numpy()
        return hook

    def get_intermediate_features(self, input_data: np.ndarray,
                                  layer_name: str = "") -> np.ndarray:
        import torch
        if not self._loaded:
            self.load()

        self._features.clear()

        if layer_name and hasattr(self._model, "named_modules"):
            for name, module in self._model.named_modules():
                if name == layer_name or (not layer_name and name):
                    handle = module.register_forward_hook(self._hook_fn(name))
                    self._hooks[name] = handle
                    break

        # If no specific layer, hook the second-to-last module
        if not self._hooks and hasattr(self._model, "named_modules"):
            modules = list(self._model.named_modules())
            if len(modules) > 2:
                name, module = modules[-2]
                handle = module.register_forward_hook(self._hook_fn(name))
                self._hooks[name] = handle

        with torch.no_grad():
            tensor = torch.from_numpy(input_data).float()
            self._model(tensor)

        # Clean up hooks
        for h in self._hooks.values():
            h.remove()
        self._hooks.clear()

        if self._features:
            key = layer_name if layer_name in self._features else list(self._features.keys())[0]
            return self._features[key]

        return np.array([])

    def get_parameters(self) -> Dict[str, np.ndarray]:
        if not self._loaded:
            self.load()
        params = {}
        if hasattr(self._model, "named_parameters"):
            for name, param in self._model.named_parameters():
                params[name] = param.detach().cpu().numpy()
        elif hasattr(self._model, "parameters"):
            for i, param in enumerate(self._model.parameters()):
                params[f"param_{i}"] = param.detach().cpu().numpy()
        return params

    @property
    def input_shape(self) -> Tuple:
        # PyTorch models don't always declare their input shape
        return (1, 3, 224, 224)

    @property
    def num_classes(self) -> int:
        if self._num_classes > 0:
            return self._num_classes
        # Try to infer from the last linear layer
        if hasattr(self._model, "named_modules"):
            import torch.nn as nn
            for name, module in reversed(list(self._model.named_modules())):
                if isinstance(module, nn.Linear):
                    return module.out_features
        return -1


def load_model(model_path: str, access_level: str = "white-box",
               **kwargs) -> ModelWrapper:
    """
    Load a model from disk with the appropriate wrapper.
    
    Args:
        model_path: Path to model file (.onnx, .pt, .pth, .torchscript)
        access_level: "white-box" or "black-box"
        **kwargs: Additional args passed to the model wrapper
    
    Returns:
        ModelWrapper instance
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".onnx":
        wrapper = ONNXModelWrapper(model_path)
    elif ext in (".pt", ".pth", ".bin"):
        wrapper = PyTorchModelWrapper(model_path, **kwargs)
    else:
        # Try ONNX first, fall back to PyTorch
        try:
            wrapper = ONNXModelWrapper(model_path)
            wrapper.load()
            return wrapper
        except Exception:
            wrapper = PyTorchModelWrapper(model_path, **kwargs)

    wrapper.load()
    return wrapper
