# ml/inference/optimizer.py
import torch
import numpy as np
from numba import jit, cuda
import onnx
import onnxruntime as ort
from typing import Dict, Any

class InferenceOptimizer:
    """Multi-level inference optimization"""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Convert to ONNX for CPU optimization
        self.onnx_session = self._convert_to_onnx()
        
        # Optimize with TensorRT if GPU available
        if torch.cuda.is_available():
            self.trt_model = self._convert_to_tensorrt()
    
    def _convert_to_onnx(self):
        """Convert LightGBM to ONNX for optimized inference"""
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        
        onnx_model = convert_lightgbm(
            self.model,
            name='RealEstatePredictor',
            initial_types=[('input', FloatTensorType([None, self.config['n_features']]))]
        )
        
        # Optimize ONNX graph
        from onnxoptimizer import optimize
        optimized_model = optimize(onnx_model)
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 4
        
        return ort.InferenceSession(
            optimized_model.SerializeToString(),
            sess_options,
            providers=['CPUExecutionProvider']
        )
    
    def _convert_to_tensorrt(self):
        """Convert to TensorRT for GPU acceleration"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Build TensorRT engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        # Add optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 50), (8, 50), (32, 50))
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        return engine
    
    @jit(nopython=True, parallel=True)
    def _batch_predict_numba(self, features: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch prediction for CPU"""
        n_samples = features.shape[0]
        results = np.zeros(n_samples, dtype=np.float64)
        
        for i in range(n_samples):
            # Apply tree traversal in parallel
            results[i] = self._predict_single(features[i])
        
        return results
    
    async def predict_optimized(self, features: np.ndarray) -> np.ndarray:
        """Route to optimal inference backend"""
        
        batch_size = features.shape[0]
        
        # Use TensorRT for large batches on GPU
        if batch_size > 100 and torch.cuda.is_available():
            return await self._predict_tensorrt(features)
        
        # Use ONNX for medium batches
        elif batch_size > 10:
            return self._predict_onnx(features)
        
        # Use Numba for small batches
        else:
            return self._batch_predict_numba(features)
    
    def _predict_onnx(self, features: np.ndarray) -> np.ndarray:
        """ONNX Runtime prediction"""
        ort_inputs = {self.onnx_session.get_inputs()[0].name: features.astype(np.float32)}
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        return ort_outputs[0].flatten()