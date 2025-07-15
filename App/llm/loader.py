from App.config.settings import settings
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_llm():
    """
    Load a remote Hugging Face model and wrap it into a LangChain-compatible pipeline.
    """
    model_id = settings.HF_MODEL
    hf_token = settings.HUGGINGFACEHUB_API_TOKEN

    if not model_id or not hf_token:
        raise ValueError("HF_MODEL or HUGGINGFACEHUB_API_TOKEN not set in .env file.")

    print(f"Loading Hugging Face model '{model_id}' via API...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        temperature=settings.TEMPERATURE,
        repetition_penalty=1.1,
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=generation_pipeline)