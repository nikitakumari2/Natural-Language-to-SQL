"""
Custom wrapper for HuggingFace models to handle response format issues.
This module provides direct integration with the HuggingFace API for text generation
tasks, avoiding some of the issues with LangChain's built-in implementations.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("hf_token")

class CustomHuggingFaceModel(LLM):
    """
    Custom LLM wrapper for HuggingFace Inference API.
    Provides more reliable handling of various model response formats.
    """
    
    model_name: str
    api_token: str = HF_API_TOKEN
    temperature: float = 0.05
    max_new_tokens: int = 512
    task: str = "text-generation"  # or "text2text-generation"
    
    def __init__(self, model_name: str, task: str = "text-generation", **kwargs):
        """Initialize the HuggingFace model wrapper.
        
        Args:
            model_name: The model name/path on HuggingFace Hub
            task: The task type (text-generation or text2text-generation)
            **kwargs: Additional parameters for the model
        """
        super().__init__(model_name=model_name, task=task, **kwargs)
        
        # Set additional parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key) and key not in ("model_name", "task"): # Avoid re-assigning already handled by explicit args to super()
                setattr(self, key, value)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "custom_huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the HuggingFace model with the given prompt.
        
        Args:
            prompt: The input text
            stop: Optional stop sequences (not directly supported by HF API)
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            Generated text from the model
        """
        if run_manager:
            run_manager.on_text(prompt, verbose=self.verbose)
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Build inference parameters based on model type
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False
            }
        }
        
        # Make API request
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response based on task type
            if response.headers.get('content-type') == 'application/json':
                # JSON response format
                result = response.json()
                
                if isinstance(result, list):
                    if self.task == "text-generation":
                        # Text generation usually returns a list of generation objects
                        generated_text = result[0].get("generated_text", "")
                    else:
                        # Text2text generation might have a different format
                        generated_text = result[0] if isinstance(result[0], str) else result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated_text = result.get("generated_text", "")
                else:
                    generated_text = str(result)
            else:
                # Plain text response
                generated_text = response.text
                
            # Apply stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text[:generated_text.find(stop_seq)]
            
            if run_manager:
                run_manager.on_text(generated_text, verbose=self.verbose)
                
            return generated_text
        
        except Exception as e:
            print(f"Error calling HuggingFace API: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            # Return a placeholder response in case of error
            return f"Error generating text: {str(e)}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters for the model."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "task": self.task
        }


# Factory function to create models
def create_hf_model(model_type: str) -> CustomHuggingFaceModel:
    """
    Factory function to create a HuggingFace model wrapper.
    
    Args:
        model_type: Type of model to create ('mixtral', 'flan-t5', etc)
        
    Returns:
        CustomHuggingFaceModel instance
    """
    if model_type.lower() == "mixtral" or model_type.lower() == "mixtral-8x7b":
        return CustomHuggingFaceModel(
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            temperature=0.05,
            max_new_tokens=512
        )
    elif model_type.lower() == "flan-t5" or model_type.lower() == "flan-t5-xl":
        return CustomHuggingFaceModel(
            model_name="google/flan-t5-base",
            task="text2text-generation",
            temperature=0.05,
            max_new_tokens=512
        )
    elif model_type.lower() == "llama2" or model_type.lower() == "llama-2-70b":
        return CustomHuggingFaceModel(
            model_name="meta-llama/Llama-2-70b-chat-hf",
            task="text-generation",
            temperature=0.05,
            max_new_tokens=512,
        )
    elif model_type.lower() == "gemma":
        return CustomHuggingFaceModel(
            model_name="deepseek-ai/DeepSeek-Prover-V2-671B",
            task="text-generation",
            temperature=0.05,
            max_new_tokens=512
            , trust_remote_code=True
        )
    else:
        # Default to providing a model for the specified name
        return CustomHuggingFaceModel(
            model_name=model_type,
            task="text-generation",
            temperature=0.05,
            max_new_tokens=512
        )


# Testing function
def test_model(model_type: str, prompt: str) -> str:
    """
    Test function to try a model with a prompt.
    
    Args:
        model_type: Type of model to test
        prompt: Test prompt
        
    Returns:
        Generated text
    """
    model = create_hf_model(model_type)
    return model(prompt)


if __name__ == "__main__":
    # Simple test
    test_prompt = "Write a short poem about artificial intelligence."
    
    print("Testing Mixtral model:")
    result = test_model("mixtral", test_prompt)
    print(result)
    
    print("\nTesting FLAN-T5 model:")
    result = test_model("flan-t5", test_prompt)
    print(result)