# AI Study Assistant

An AI-powered study assistant that helps students understand their lecture notes by answering questions and generating multiple-choice quizzes from PDF files.

## Features

- Upload lecture notes in PDF format
- Extract text from PDF files
- Split text into smaller chunks
- Generate embeddings using Sentence Transformers
- Store embeddings using FAISS
- Search relevant information from uploaded notes
- Ask questions and receive AI-generated answers
- Generate 3 multiple-choice questions with four options, correct answers, and explanations

## Technologies Used

- Python
- Gradio
- Google Gemini API
- Sentence Transformers
- FAISS
- PyPDF
- Python-dotenv

## Project Structure

- `app.py` – Main application
- `requirements.txt` – Required Python libraries
- `.gitignore` – Files excluded from GitHub

## Installation and Setup

1. Clone the repository:

   `git clone https://github.com/arihajamil33-ui/AI_Study_Assistant.git`

2. Open the project folder:

   `cd AI_Study_Assistant`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file in the project folder and add your Gemini API key:

   `GEMINI_API_KEY=your_api_key_here`

5. Run the application:

   `python app.py`

6. Open the Gradio local URL shown in the terminal.

## How to Use

1. Upload your lecture notes in PDF format.
2. Ask a question related to your notes.
3. Receive an AI-generated answer.
4. Click Generate 3 MCQs to create a practice quiz.

## Purpose

This project demonstrates the use of Retrieval-Augmented Generation (RAG), vector search, and generative AI to create an interactive study assistant.

## Note

The Gemini API key should be kept private and must never be uploaded to GitHub.
