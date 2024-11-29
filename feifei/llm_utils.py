import logging
import textwrap
from langchain_core.prompts import PromptTemplate


def initialize_llm(backend="openai", **kwargs):
    """
    Initialize the Language Model (LLM) based on the selected backend.

    Args:
        backend (str): The backend to use for the LLM.
        **kwargs: Additional keyword arguments for the LLM initialization.

    Returns:
        object: Initialized LLM object.
    """
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-3.5-turbo"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-v1"),
            max_tokens_to_sample=kwargs.get("max_tokens_to_sample", 1024),
        )
    elif backend == "google":
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model_name=kwargs.get("model_name", "chat-bison"),
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_output_tokens", 1024),
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
        backend (str): The backend to use for the embeddings.
        **kwargs: Additional keyword arguments for the embeddings initialization.

    Returns:
        object: Initialized embeddings object.
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
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    logging.info(f"Initialized embeddings with backend: {backend}")
    return embeddings


def get_prompt_template(prompt_template: str):
    """
    Retrieve the prompt template.

    Args:
        prompt_template (str): The prompt template string.

    Returns:
        PromptTemplate: The prompt template object.
    """
    return PromptTemplate(
        template=textwrap.dedent(prompt_template),
        input_variables=["input", "context", "sources"],
    )
