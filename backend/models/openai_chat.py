import openai
from typing import List, Dict, Generator, Optional
import json
import os

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def safe_extract_content(chunk) -> Optional[str]:
    """Safely extract content from OpenAI streaming response chunk"""
    try:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                return delta.content
        return None
    except Exception as e:
        print(f"Error extracting content: {e}")
        return None

def query_openai_api(instruction: str, inp: str, model: str = "gpt-4o"):
    """
    Query OpenAI API
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

async def get_openai_streaming_response(messages: List[Dict[str, str]], model: str = "gpt-4o"):
    """
    Get streaming response from OpenAI API (async version)
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model to use
    
    Yields:
        str: Content chunks from the streaming response
    """
    try:
        print(f"Starting OpenAI stream for model: {model}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        chunk_count = 0
        for chunk in response:
            content = safe_extract_content(chunk)
            if content is not None:
                chunk_count += 1
                print(f"Chunk {chunk_count}: '{content}'")  # Debug log
                yield content
        
        print(f"Stream completed. Total chunks: {chunk_count}")
                
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        yield f"Error: {str(e)}"

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