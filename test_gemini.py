import asyncio
import os
from google import genai
from google.genai import types

async def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        print("Calling gemini-3.7-flash...")
        response = await client.aio.models.generate_content(
            model='gemini-3.7-flash',
            contents='Hello',
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
