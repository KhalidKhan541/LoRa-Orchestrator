import os
from langchain_core.language_models import BaseChatModel


def build_llm(provider: str = None, model: str = None, temperature: float = 0.2) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20241022", temperature=temperature)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.1-70b-versatile", temperature=temperature)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1", temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")