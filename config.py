import os
from dotenv import load_dotenv

load_dotenv()


def _split_endpoint(endpoint: str | None) -> tuple[str | None, str | None]:
    """Return (resource_base_url, api_version) extracted from a raw endpoint."""
    if not endpoint:
        return None, None

    base = endpoint.strip()
    if not base:
        return None, None

    api_version = None
    if "api-version=" in base:
        api_version = base.split("api-version=")[-1].split("&")[0].strip()
    base = base.split("?")[0]
    if "/openai/" in base:
        base = base.split("/openai/")[0]

    normalized = base.rstrip("/") + "/"
    return normalized, api_version


_raw_text_endpoint = os.getenv("AZURE_TEXT_ENDPOINT")
_text_endpoint, _text_version = _split_endpoint(_raw_text_endpoint)

_raw_image_endpoint = os.getenv("AZURE_IMAGE_ENDPOINT")
_image_endpoint, _image_version = _split_endpoint(_raw_image_endpoint)


class Config:
    # Azure OpenAI text
    AZURE_TEXT_API_KEY = os.getenv("AZURE_TEXT_API_KEY")
    AZURE_TEXT_ENDPOINT = _text_endpoint
    AZURE_TEXT_API_VERSION = os.getenv("AZURE_TEXT_API_VERSION", _text_version or "2024-02-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

    # Azure AI Speech
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

    # Azure Image
    AZURE_IMAGE_API_KEY = os.getenv("AZURE_IMAGE_KEY")
    AZURE_IMAGE_ENDPOINT = _image_endpoint
    AZURE_IMAGE_API_VERSION = os.getenv("AZURE_IMAGE_API_VERSION", _image_version or "2023-12-01-preview")
    AZURE_IMAGE_DEVELOPMENT_NAME = "FLUX.1-Kontext-pro"

    ARTICLE_DIR = "./article"
    OUTPUT_DIR = "./output"
    JP_FONT_PATH = os.getenv("JP_FONT_PATH")
    MOVIE_DIR = "./movie"
