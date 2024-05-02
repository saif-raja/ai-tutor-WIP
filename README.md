# AI Tutor Chatbot Project

## Overview
This project aims to develop an AI Tutor Chatbot capable of handling both text and image-based queries from users. The chatbot will fetch relevant YouTube video links to assist users in learning and understanding various topics.

## Features
- Respond to text and image-based queries.
- Fetch and display relevant YouTube video links.
- Maintain a conversation state to allow continuous interaction.
- Utilize OpenAI models for processing queries.
- Built with FastAPI for efficient asynchronous API performance.

## Technologies Used
- FastAPI for the API server.
- OpenAI's GPT models for natural language understanding.
- YouTube Data API for fetching video content.
- Docker for containerization and easy deployment.

## Local Setup
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Set up environment variables for API keys (OpenAI and YouTube).
4. Run the server using Uvicorn: `uvicorn app.main:app --reload`.

## Testing
The application includes a suite of automated tests to ensure functionality. Run tests using the pytest framework.

## Deployment
The application is containerized using Docker, allowing for deployment on any system that supports Docker.

## Documentation
API documentation is available via Swagger UI, accessible from the root URL of the API server when running locally.
