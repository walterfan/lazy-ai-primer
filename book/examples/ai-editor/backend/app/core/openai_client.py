"""OpenAI client kwargs for self-hosted / self-signed TLS."""
import httpx

from app.core.config import Settings


def get_openai_client_kwargs(settings: Settings) -> dict:
    """
    Build kwargs for openai.OpenAI when using self-hosted or
    OpenAI-compatible APIs with optional self-signed TLS.

    - OPENAI_BASE_URL: endpoint (e.g. https://your-llm.example.com/v1).
    - OPENAI_CA_BUNDLE: path to CA cert file (preferred for self-signed).
    - OPENAI_SSL_VERIFY: set False only if no CA bundle (insecure).
    """
    kwargs: dict = {}
    if getattr(settings, "OPENAI_BASE_URL", "") and settings.OPENAI_BASE_URL.strip():
        kwargs["base_url"] = settings.OPENAI_BASE_URL.strip()

    verify: bool | str = getattr(settings, "OPENAI_SSL_VERIFY", True)
    ca_bundle = getattr(settings, "OPENAI_CA_BUNDLE", "") or ""
    if ca_bundle and ca_bundle.strip():
        verify = ca_bundle.strip()

    if verify is not True:
        kwargs["http_client"] = httpx.Client(verify=verify)

    return kwargs
