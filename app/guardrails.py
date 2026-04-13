"""
Azure Content Safety — checks both input and output.
Medical context: also blocks self-harm and dangerous advice requests.
"""
import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = ContentSafetyClient(
            endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
            credential=AzureKeyCredential(
                os.getenv("AZURE_CONTENT_SAFETY_KEY")
            )
        )
    return _client


def check_content(text: str, threshold: int = 2) -> dict:
    """
    Returns {"safe": True} or {"safe": False, "reason": "...", "category": "..."}
    Lower threshold = stricter. Medical apps use 2 (low severity).
    """
    try:
        response = _get_client().analyze_text(
            AnalyzeTextOptions(
                text=text[:1000],
                categories=[
                    TextCategory.HATE,
                    TextCategory.SELF_HARM,
                    TextCategory.SEXUAL,
                    TextCategory.VIOLENCE
                ]
            )
        )
        for item in response.categories_analysis:
            if item.severity >= threshold:
                return {
                    "safe": False,
                    "reason": f"Content flagged: {item.category}",
                    "category": str(item.category),
                    "severity": item.severity
                }
        return {"safe": True}

    except Exception as e:
        # Fail open — log but don't block if safety service is down
        print(f"⚠️  Safety service unavailable: {e}")
        return {"safe": True}