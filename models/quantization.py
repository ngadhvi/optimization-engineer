import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from optimum.quanto import quantize, freeze, qint8, qint4, qint2, qfloat8
from enum import Enum
from typing import Tuple, Any, Optional

class QuantizationType(Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4" 
    INT2 = "int2"
    FLOAT8 = "float8"

class ModelLoader:
    """Handles model loading with different quantization strategies."""
    
    @staticmethod
    def load_standard(model_name: str, device: str) -> Tuple[Any, Any]:
        """Load model without quantization."""
        print(f"Loading {model_name} (standard)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            model = model.to(device)
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    @staticmethod  
    def load_quantized_transformers(model_name: str, quant_type: QuantizationType) -> Tuple[Any, Any]:
        """Load model using Transformers QuantoConfig integration."""
        print(f"Loading {model_name} with {quant_type.value} quantization (Transformers)")
        
        quant_config = QuantoConfig(weights=quant_type.value)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto", 
            quantization_config=quant_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    @staticmethod
    def load_quantized_direct(model_name: str, quant_type: QuantizationType, device: str) -> Tuple[Any, Any]:
        """Load model using direct quanto quantization API."""
        print(f"Loading {model_name} with {quant_type.value} quantization (Direct API)")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        # Apply quantization
        quant_map = {
            QuantizationType.INT8: qint8,
            QuantizationType.INT4: qint4,
            QuantizationType.INT2: qint2,
            QuantizationType.FLOAT8: qfloat8
        }
        
        quantize(model, weights=quant_map[quant_type])
        freeze(model)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer