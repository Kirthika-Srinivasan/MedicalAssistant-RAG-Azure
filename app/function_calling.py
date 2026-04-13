"""
Function calling router — the model decides which tool to invoke:
1. search_knowledge_base  → query the NIH MedQuAD index
2. recommend_specialist   → suggest what kind of doctor to see
3. emergency_redirect     → flag urgent/emergency queries

This shows production-grade tool routing, not just basic RAG.
"""
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the NIH medical knowledge base for information "
                "about diseases, symptoms, treatments, medications, "
                "and general health questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Refined search query"
                    },
                    "topic_type": {
                        "type": "string",
                        "enum": ["disease", "symptom", "treatment",
                                 "medication", "prevention", "general"],
                        "description": "Type of medical information needed"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_specialist",
            "description": (
                "When the user's condition needs professional assessment, "
                "recommend the appropriate type of medical specialist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_area": {
                        "type": "string",
                        "description": "Body system or condition area"
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["routine", "soon", "urgent"],
                        "description": "How urgently they should seek care"
                    }
                },
                "required": ["condition_area", "urgency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_redirect",
            "description": (
                "Use ONLY when user describes symptoms suggesting "
                "a medical emergency: chest pain, stroke signs, "
                "severe breathing difficulty, overdose, or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_summary": {
                        "type": "string",
                        "description": "Brief description of emergency symptoms"
                    }
                },
                "required": ["symptom_summary"]
            }
        }
    }
]


def route_query(user_query: str) -> dict:
    """
    Uses function calling to decide the right action for a medical query.
    Returns a dict with "action" key plus relevant parameters.
    """
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical query router. Analyse the user's "
                    "question and select the most appropriate tool. "
                    "Always prefer search_knowledge_base for general questions. "
                    "Only use emergency_redirect for genuine emergencies."
                )
            },
            {"role": "user", "content": user_query}
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool = message.tool_calls[0]
        args = json.loads(tool.function.arguments)
        return {"action": tool.function.name, **args}

    # Default fallback
    return {"action": "search_knowledge_base", "query": user_query}