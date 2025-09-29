from google import genai

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

async def get_gemini_response(text: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=text
    )
    return response.text
