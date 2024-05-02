# main.py
import os
# import json
import uuid
import aiofiles
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException

# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

from db_utils import DatabaseMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain import hub
import googleapiclient.discovery
from typing import Optional, Type
import openai
import base64

# from fastapi.responses import JSONResponse
# from fastapi.exception_handlers import request_validation_exception_handler
# from pydantic import ValidationError

# Securely manage API keys and database connection strings
# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = "sk-proj-REDACTED"
yt_api_key = os.getenv("YOUTUBE_API_KEY")



# Define the input schema for YouTubeSearchTool
class YouTubeSearchInput(BaseModel):
    key_word_str: str = Field(description="The search query for finding relevant YouTube videos.")

# Define the YouTube API tool
class YouTubeSearchTool(BaseTool):
    name = "YouTubeSearch"
    description = "Searches YouTube for videos related to a given query."
    args_schema = Type[BaseModel] = YouTubeSearchInput(BaseModel)
    
    def __init__(self, api_key: str):
        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    def run(self, query: str):
        request = self.youtube.search().list(q=query, part="snippet", type="video")
        response = request.execute()
        # Return a formatted response or just the URLs as needed
        video_urls = [item['id']['videoId'] for item in response.get('items', [])]
        return {"videos": video_urls}



# Define the input schema for VisionTool
class VisionInput(BaseModel):
    image_path: str = Field(description="File path to the image for OCR processing.")

# VisionTool using OpenAI's GPT-4 Turbo Vision for OCR
class VisionTool(BaseTool):
    name = "VisionOCR"
    description = "Performs OCR using OpenAI's GPT-4 Turbo Vision to extract text from images."
    args_schema: Type[BaseModel] = VisionInput

    def __init__(self):
        global openai_api_key
        openai.api_key = openai_api_key

    def encode_image_to_base64(self, image_path: str) -> str:
        """Read an image file and encode it to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _run(self, image_path: str):
        image_base64 = self.encode_image_to_base64(image_path)
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-vision",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can read and describe images."},
                {"role": "user", "content": image_base64, "file_type": "image/jpeg"}  # Ensure file_type matches your image format
            ]
        )
        # Extract the text portion from the response, assuming the model outputs text directly after the image description
        text_output = response['choices'][0]['message']['content']
        return text_output



# Initialize the LLM and Database Memory
llm = OpenAI(model="gpt-3.5-turbo-0125", api_key=openai_api_key)
db_memory = DatabaseMemory()

# Initialize tools
youtube_tool = YouTubeSearchTool(yt_api_key)
vision_tool = VisionTool()
tools = [
    youtube_tool, 
    vision_tool
    ]

Create a tool-calling agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI tutor. You will answer questions.If required you will use tools to answer questions to make sure the student has the answer.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    )

agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)




app = FastAPI(title="AI Tutor Chatbot")

@app.post("/query/")
async def query(query: str = Form(...), image: Optional[UploadFile] = File(None)):
    session_id = str(uuid.uuid4())  # Dynamically generate a unique session ID
    image_path = None

    # Save and process the image if provided
    if image:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        image_path = f"temp/{image.filename}"
        async with aiofiles.open(image_path, 'wb') as out_file:
            await out_file.write(await image.read())

    # Execute agent with input and manage state
    previous_state = db_memory.get(session_id) or {}
    context = {"text": query, "image_path": image_path}
    response = agent_executor.invoke(context, session_state=previous_state)
    
    # Update the state in the database
    db_memory.set(session_id, response.session_state)

    return {
        "query": query,
        "response": response.output
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
