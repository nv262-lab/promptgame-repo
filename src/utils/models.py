"""
Model Interfaces for LLM Access

Provides unified interface for different LLM backends:
- OpenAI GPT-4o
- Anthropic Claude 3
- HuggingFace LLaMA-3-70B
- HuggingFace Mistral-7B
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time


@dataclass
class ModelResponse:
    """Standardized model response"""
    content: str
    model: str
    latency_ms: float
    tokens_used: int
    finish_reason: str


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ModelResponse:
        """Generate response from model"""
        pass
    
    def __call__(
        self,
        system_prompt: str,
        user_input: str,
        **kwargs
    ) -> str:
        """Convenience method returning just content"""
        response = self.generate(system_prompt, user_input, **kwargs)
        return response.content


class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models (GPT-4o)"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ModelResponse:
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        latency = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=response.choices[0].message.content,
            model=self.model,
            latency_ms=latency,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason
        )


class AnthropicInterface(BaseModelInterface):
    """Interface for Anthropic models (Claude 3)"""
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        
        self.model = model
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ModelResponse:
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input}
            ],
            temperature=temperature
        )
        
        latency = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=response.content[0].text,
            model=self.model,
            latency_ms=latency,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason
        )


class HuggingFaceInterface(BaseModelInterface):
    """Interface for HuggingFace models (LLaMA-3, Mistral)"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        device: str = "auto",
        load_in_8bit: bool = True
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers package required: pip install transformers")
        
        self.model_name = model_name
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ModelResponse:
        import torch
        
        start_time = time.time()
        
        # Format prompt based on model type
        if "llama" in self.model_name.lower():
            prompt = self._format_llama_prompt(system_prompt, user_input)
        elif "mistral" in self.model_name.lower():
            prompt = self._format_mistral_prompt(system_prompt, user_input)
        else:
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),  # Avoid division by zero
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        latency = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=response_text.strip(),
            model=self.model_name,
            latency_ms=latency,
            tokens_used=outputs.shape[1],
            finish_reason="stop"
        )
    
    def _format_llama_prompt(self, system_prompt: str, user_input: str) -> str:
        """Format prompt for LLaMA-3 Instruct"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _format_mistral_prompt(self, system_prompt: str, user_input: str) -> str:
        """Format prompt for Mistral Instruct"""
        return f"""<s>[INST] {system_prompt}

{user_input} [/INST]"""


class MockModelInterface(BaseModelInterface):
    """Mock interface for testing"""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ModelResponse:
        # Simple mock response
        content = f"I understand you want help with: {user_input[:100]}. As a helpful assistant, I'll do my best to assist you while following my guidelines."
        
        return ModelResponse(
            content=content,
            model=self.model_name,
            latency_ms=10.0,
            tokens_used=len(content.split()),
            finish_reason="stop"
        )


def create_model_interface(
    model_name: str,
    **kwargs
) -> BaseModelInterface:
    """
    Factory function to create model interface.
    
    Args:
        model_name: Name of model to create interface for
        **kwargs: Additional arguments for the interface
    
    Returns:
        Configured model interface
    """
    model_lower = model_name.lower()
    
    if "gpt" in model_lower or "openai" in model_lower:
        return OpenAIInterface(model=model_name, **kwargs)
    
    elif "claude" in model_lower or "anthropic" in model_lower:
        return AnthropicInterface(model=model_name, **kwargs)
    
    elif "llama" in model_lower or "mistral" in model_lower:
        return HuggingFaceInterface(model_name=model_name, **kwargs)
    
    elif "mock" in model_lower:
        return MockModelInterface(model_name=model_name)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def model_factory(model_name: str):
    """
    Create a simple callable model function.
    
    Args:
        model_name: Name of model
    
    Returns:
        Callable that takes (system_prompt, user_input) and returns response
    """
    interface = create_model_interface(model_name)
    return lambda system_prompt, user_input: interface(system_prompt, user_input)
