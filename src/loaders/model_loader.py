import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from loguru import logger

# API model imports
from decouple import config as config_decouple
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAI

# Model registry
MODEL_REGISTRY = {
    # Dense models - Small (1 GPU)
    "Qwen3-1.7B": {
        "hf_path": "Qwen/Qwen3-1.7B",
        "gpu_memory": 0.75,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,
    },
    "Llama-3.2-3B-Instruct": {
        "hf_path": "meta-llama/Llama-3.2-3B-Instruct",
        "gpu_memory": 0.8,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "gemma-3-4b-it": {
        "hf_path": "google/gemma-3-4b-it",
        "gpu_memory": 0.8,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    
    # Dense models - Medium (1 GPU)
    "Qwen2.5-7B-Instruct": {
        "hf_path": "Qwen/Qwen2.5-7B-Instruct",
        "gpu_memory": 0.85,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,
    },
    "Qwen3-8B": {
        "hf_path": "Qwen/Qwen3-8B",
        "gpu_memory": 0.85,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,
    },
    "Mistral-7B-Instruct-v0.3": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "gpu_memory": 0.85,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "Mistral-7B-Instruct-v0.1": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.1",
        "gpu_memory": 0.85,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    
    # Dense models - Large (2-4 GPUs)
    "Mistral-Small-24B-Instruct-2501": {
        "hf_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "gpu_memory": 0.90,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,  # 24B needs 2 GPUs
    },
    "Llama-3.3-70B-Instruct": {
        "hf_path": "meta-llama/Llama-3.3-70B-Instruct",
        "gpu_memory": 0.90,
        "max_model_len": 8192,
        "tensor_parallel_size": 4,  # 70B needs 4 GPUs
    },
    
    # MoE models (4 GPUs)
    "Mixtral-8x7B-Instruct-v0.1": {
        "hf_path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpu_memory": 0.90,
        "max_model_len": 8192,
        "tensor_parallel_size": 4,  # 47B effective
    },
    "Qwen3-30B-A3B": {
        "hf_path": "Qwen/Qwen3-30B-A3B",
        "gpu_memory": 0.90,
        "max_model_len": 8192,
        "tensor_parallel_size": 2,  # MoE with 3B active
    },
}


class VLLMWrapper:
    """vLLM wrapper for local model inference with batch support and thinking mode"""
    
    def __init__(self, model_name, temperature=0.7, top_p=0.9, max_tokens=1024, enable_thinking=False, **kwargs):
        # Remove vllm: prefix if present
        if model_name.startswith("vllm:"):
            model_name = model_name.replace("vllm:", "")
        
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        
        # Get model configuration
        model_config = self._get_model_config(model_name, enable_thinking=enable_thinking, **kwargs)
        
        # Set environment optimizations
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["hf_path"])
        
        # Log configuration
        logger.info(f"=" * 80)
        logger.info(f"Loading model: {model_config['hf_path']}")
        logger.info(f"  Tensor parallel size: {model_config['tensor_parallel_size']} GPU(s)")
        logger.info(f"  GPU memory utilization: {model_config['gpu_memory']:.2%}")
        logger.info(f"  Max model length: {model_config['max_model_len']}")
        logger.info(f"  Thinking mode: {enable_thinking}")
        logger.info(f"=" * 80)
        
        # Build vLLM configuration
        vllm_kwargs = {
            "model": model_config["hf_path"],
            "gpu_memory_utilization": model_config["gpu_memory"],
            "max_model_len": model_config["max_model_len"],
            "enable_prefix_caching": True,
            "tensor_parallel_size": model_config["tensor_parallel_size"],
        }
        
        # Qwen thinking settings
        is_qwen_thinking = enable_thinking and "qwen" in model_config["hf_path"].lower()
        if is_qwen_thinking:
            vllm_kwargs.update({
                "disable_cascade_attn": True, # Works around a bug with Qwen thinking models
            })

            # Set thinking-specific sampling parameters
            temperature = 0.6
            top_p = 0.95
            max_tokens = 32768

        try:
            self.llm = LLM(**vllm_kwargs)
            logger.info(f"Successfully loaded {model_config['hf_path']}")
        except Exception as e:
            logger.error(f"Failed to load {model_config['hf_path']}: {e}")
            raise

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def _get_model_config(self, model_name, enable_thinking=False, **kwargs):
        """Get model configuration from registry or defaults"""
        
        # Handle :thinking suffix
        is_thinking_mode = model_name.endswith(":thinking") or enable_thinking
        if is_thinking_mode:
            model_name = model_name.replace(":thinking", "")
        
        if "/" in model_name:
            hf_path = model_name
            registry_key = model_name.split("/")[-1]
        else:
            registry_key = model_name
            hf_path = model_name
        
        if registry_key in MODEL_REGISTRY:
            config = MODEL_REGISTRY[registry_key].copy()
        else:
            config = {
                "hf_path": hf_path,
                "gpu_memory": 0.85,
                "max_model_len": kwargs.get("max_model_len", 8192),
            }
        
        # Override with explicitly provided kwargs
        config["gpu_memory"] = kwargs.get("gpu_memory_utilization", config["gpu_memory"])
        config["max_model_len"] = kwargs.get("max_model_len", config["max_model_len"])
        
        # Apply thinking mode override
        if is_thinking_mode:
            config["max_model_len"] =  32768
        
        return config
    
    def _remove_thinking_tokens(self, text):
        """Remove thinking tokens from generated text"""
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            # Handle malformed cases - missing opening tag
            r'^.*?</think>\s*',
            r'^.*?</thinking>\s*',
            # Handle partial/broken tags
            r'<think[^>]*>.*?</think>',
            r'<thinking[^>]*>.*?</thinking>',
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        return re.sub(r'\n\s*\n+', '\n', cleaned.strip())
    
    def generate(self, prompt_text):
        """Generate single response"""
        try:
            messages = [{"role": "user", "content": prompt_text}]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except:
            chat_prompt = prompt_text
        
        try:
            outputs = self.llm.generate([chat_prompt], self.sampling_params)
            result = outputs[0].outputs[0].text.strip()
            
            # Remove thinking tokens if present
            result = self._remove_thinking_tokens(result)
            return result
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_batch(self, prompt_texts):
        """Generate batch responses"""
        formatted_prompts = []
        for prompt_text in prompt_texts:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                formatted_prompts.append(chat_prompt)
            except:
                formatted_prompts.append(prompt_text)
        
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        results = [output.outputs[0].text.strip() for output in outputs]
        
        # Remove thinking tokens from all results
        return [self._remove_thinking_tokens(result) for result in results]


class APIModelWrapper:
    """Wrapper for API-based models (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, model_name, temperature=0.7, max_tokens=512, **kwargs):
        self.model_name = model_name
        self.llm = self._create_api_model(model_name, temperature, max_tokens)
    
    def _create_api_model(self, model_name, temperature, max_tokens):
        """Create appropriate API model based on name"""
        if model_name.startswith("gpt-"):
            return ChatOpenAI(
                openai_api_key=config_decouple("OPEN_AI_KEY"),
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif model_name.startswith("claude-"):
            return ChatAnthropic(
                model=model_name,
                anthropic_api_key=config_decouple("ANTHROPIC_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif model_name.startswith("mistral-"):
            return ChatMistralAI(
                model=model_name,
                mistral_api_key=config_decouple("MISTRAL_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif model_name.startswith("gemini-"):
            return GoogleGenerativeAI(
                model=model_name,
                google_api_key=config_decouple("GOOGLE_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported API model: {model_name}")
    
    def generate(self, prompt_text):
        """Generate single response"""
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt_text}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_batch(self, prompt_texts):
        """Generate batch responses"""
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompt_texts]
        responses = self.llm.batch(messages_batch)
        if not self.model_name.startswith("gemini-"):
            responses = [response.content for response in responses]
        return [resp.strip() for resp in responses]


def load_model(model_name, temperature=0.7, max_tokens=1024, enable_thinking=False, **kwargs):
    """Load a local vLLM model ("vllm:" prefix) or an API model, chosen by name."""
    if model_name.startswith("vllm:"):
        return VLLMWrapper(
            model_name, 
            temperature=temperature, 
            top_p=kwargs.get("top_p", 0.9), 
            max_tokens=max_tokens, 
            enable_thinking=enable_thinking,
            **kwargs
        )
    elif any(model_name.startswith(prefix) for prefix in ["gpt-", "claude-", "mistral-", "gemini-"]):
        return APIModelWrapper(model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")