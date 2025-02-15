import logging
import textwrap
from langchain_core.prompts import PromptTemplate


def initialize_llm(backend="openai", **kwargs):
    """
    Initialize the Language Model (LLM) based on the selected backend.

    Args:
        backend (str): The backend to use for the LLM. Supported options:
            - 'openai': OpenAI's GPT models
            - 'anthropic': Anthropic's Claude models
            - 'google': Google's Vertex AI models
            - 'qianfan': Baidu's ERNIE models
            - 'ollama': Local Ollama models
            - 'huggingface': HuggingFace models via API
            - 'local': Local HuggingFace models
        **kwargs: Additional keyword arguments for LLM initialization including:
            - model_name (str): Name of the model to use
            - temperature (float): Sampling temperature
            - max_output_tokens (int): Maximum output length
            - openai_api_base (str): Base URL for OpenAI API
            - anthropic_api_base (str): Base URL for Anthropic API
            - repo_id (str): HuggingFace model repository ID
            - device (str): Device to run local models on

    Returns:
        object: Initialized LLM object from the specified backend.

    Raises:
        ValueError: If an unsupported backend is specified.
    """
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o-latest"),
            openai_api_base=kwargs.get("openai_api_base", "https://api.openai.com/v1"),
        )
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet"),
            base_url=kwargs.get("anthropic_api_base", "https://api.anthropic.com"),
        )
    elif backend == "google":
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model_name=kwargs.get("model_name", "chat-bison"),
        )
    elif backend == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        llm = ChatDeepSeek(
            model_name=kwargs.get("model_name", "deepseek-chat"),
            api_base=kwargs.get("api_base", "https://api.deepseek.com"),
        )
    elif backend == "qianfan":
        from langchain_community.chat_models import QianfanChatEndpoint

        llm = QianfanChatEndpoint(
            model=kwargs.get("model_name", "ERNIE-3.5-8K"),
            temperature=kwargs.get("temperature", 0.7),
        )
    elif backend == "ollama":
        from langchain_ollama.chat_models import ChatOllama

        llm = ChatOllama(
            base_url=kwargs.get("ollama_base_url", "http://localhost:11434"),
            model=kwargs.get("model_name", "llama2"),
        )
    elif backend == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=kwargs.get("repo_id", "gpt2"),
            model_kwargs={
                "temperature": kwargs.get("temperature", 0.7),
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
            },
        )
    elif backend == "local":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_name = kwargs.get("model_name", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            device=kwargs.get("device", -1),
        )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
    logging.info(f"Initialized LLM with backend: {backend}")
    return llm


def initialize_embeddings(backend="openai", **kwargs):
    """
    Initialize the embeddings model based on the selected backend.

    Args:
        backend (str): The backend to use for embeddings. Supported options:
            - 'openai': OpenAI's embedding models
            - 'qianfan': Baidu's ERNIE embedding models
            - 'huggingface': HuggingFace embedding models
            - 'local': Local HuggingFace embedding models
        **kwargs: Additional keyword arguments for embeddings initialization including:
            - model_name (str): Name of the embedding model
            - openai_api_base (str): Base URL for OpenAI API
            - embedding_model_name (str): Name of HuggingFace embedding model

    Returns:
        object: Initialized embeddings object from the specified backend.

    Raises:
        ValueError: If an unsupported backend is specified.
    """
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=kwargs.get("model_name"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "qianfan":
        from langchain_community.embeddings import QianfanEmbeddingsEndpoint

        embeddings = QianfanEmbeddingsEndpoint()
    elif backend == "huggingface" or backend == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = kwargs.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    logging.info(f"Initialized embeddings with backend: {backend}")
    return embeddings


def get_prompt_template(prompt_template: str):
    """
    Create a prompt template for generating responses.

    Args:
        prompt_template (str): The template string containing placeholders for:
            - {input}: The user's input/question
            - {context}: Retrieved context information
            - {sources}: Source references

    Returns:
        PromptTemplate: A configured prompt template object with the specified variables.
    """
    return PromptTemplate(
        template=textwrap.dedent(prompt_template),
        input_variables=["input", "context", "sources"],
    )
