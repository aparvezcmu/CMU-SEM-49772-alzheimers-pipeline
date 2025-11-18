from typing import List, Dict, Optional
import json
import os

import httpx

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")


async def _post_openai(payload: Dict) -> Dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def get_openai_streaming_response(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
):
    """
    Get streaming response from OpenAI API (async version) using direct HTTP calls.

    This avoids relying on the `openai` Python SDK so we don't hit
    compatibility issues with the installed `httpx` version.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": True,
    }

    try:
        print(f"Starting OpenAI stream for model: {model}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
                response.raise_for_status()

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # OpenAI streams lines prefixed with "data: "
                    if not line.startswith("data: "):
                        continue

                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON from line: {line}")
                        continue

                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if content:
                            chunk_count += 1
                            print(f"Chunk {chunk_count}: '{content}'")
                            yield content

        print(f"Stream completed. Total chunks: {chunk_count}")

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        yield f"Error: {str(e)}"


async def summarize_plot_with_image(base64_image: str, prompt: str) -> str:
    """Call GPT-4o to summarize a PNG provided as base64 data."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    try:
        data = await _post_openai(payload)
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"Image summary failed: {exc}")
        raise

def format_messages(user_input: str, chat_history: List[Dict] = None) -> List[Dict[str, str]]:
    """
    Format messages for OpenAI API
    
    Args:
        user_input: Current user message
        chat_history: Previous conversation history
    
    Returns:
        List of formatted messages
    """
    messages = []
    
    # Add system message
    messages.append({
        "role": "system",
        "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
    })
    
    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_input
    })
    
    return messages