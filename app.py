import os
import time

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai


# Load local .env values (Render uses Environment Variables instead).
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Keep the text-generation model already used by the working local project.
MODEL_NAME = "gemini-3.5-flash"

# Hosted text-embedding model: no local PyTorch/SentenceTransformer model needed.
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32


def ensure_gemini_client():
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your local .env file "
            "or to the Render Environment settings."
        )


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text


def split_text_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def _embed_texts(texts, task_type):
    """Create Gemini embeddings in small batches to keep memory use low."""
    ensure_gemini_client()

    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    all_vectors = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    contents=batch,
                    config={
                        "task_type": task_type,
                        "output_dimensionality": EMBEDDING_DIMENSION,
                    },
                )

                if not response.embeddings:
                    raise RuntimeError("Gemini returned no embeddings.")

                batch_vectors = [embedding.values for embedding in response.embeddings]
                all_vectors.extend(batch_vectors)
                break

            except Exception as error:
                error_text = str(error).upper()
                retryable = any(
                    marker in error_text
                    for marker in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
                )

                if retryable and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue

                raise RuntimeError(f"Gemini embedding error: {error}") from error

    vectors = np.asarray(all_vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError("Gemini returned an unexpected number of embeddings.")

    return vectors


def create_embeddings(chunks):
    return _embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")


def create_vector_store(embeddings):
    """Store normalized vectors in NumPy; this replaces the FAISS index."""
    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No embeddings were created for this PDF.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def search_similar_chunks(question, chunks, vector_store, top_k=2):
    if not chunks or vector_store is None:
        return []

    question_embedding = _embed_texts(
        [question], task_type="RETRIEVAL_QUERY"
    )
    query_vector = question_embedding[0]
    query_norm = np.linalg.norm(query_vector)

    if query_norm != 0:
        query_vector = query_vector / query_norm

    similarities = vector_store @ query_vector
    result_count = min(top_k, len(chunks))
    best_indices = np.argsort(similarities)[-result_count:][::-1]

    return [chunks[int(index)] for index in best_indices]


def process_pdf(pdf_file):
    global pdf_chunks, pdf_vector_store

    if pdf_file is None:
        return "Please upload a PDF file."

    try:
        extracted_text = extract_text_from_pdf(pdf_file)

        if not extracted_text.strip():
            return "No readable text found in the PDF."

        chunks = split_text_into_chunks(extracted_text)
        if not chunks:
            return "No readable text found in the PDF."

        embeddings = create_embeddings(chunks)
        vector_store = create_vector_store(embeddings)

        # Save the new PDF data only after processing succeeds.
        pdf_chunks = chunks
        pdf_vector_store = vector_store

        return "PDF processed successfully!\n\n" + "\n\n".join(
            [f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)]
        )

    except Exception as error:
        return f"PDF Processing Error: {error}"


def search_pdf(question):
    if "pdf_chunks" not in globals() or "pdf_vector_store" not in globals():
        return ""

    results = search_similar_chunks(
        question,
        pdf_chunks,
        pdf_vector_store,
    )
    return "\n\n".join(results)


def generate_answer(question, context):
    ensure_gemini_client()

    prompt = f"""
You are an AI Study Assistant.

Answer using ONLY the provided context.
If the answer is not available in the context, say that it is not available
in the uploaded notes.

Context:
{context}

Student Question:
{question}

Answer:
"""

    for attempt in range(3):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )
            return response.output_text

        except Exception as error:
            if "503" in str(error) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def generate_quiz(context):
    ensure_gemini_client()

    prompt = f"""
You are an AI Quiz Generator.

Create exactly 3 multiple-choice questions using ONLY the provided study notes.

Each question must have:
1. Four options (A, B, C, D)
2. Only one correct answer
3. The correct answer clearly mentioned
4. A short explanation of the correct answer

Do not use information outside the study notes.

Study Notes:
{context}

Generate the quiz in a clear and easy-to-read format.
"""

    for attempt in range(3):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )
            return response.output_text

        except Exception as error:
            if "503" in str(error) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def answer_question(question):
    if not question or not question.strip():
        return "Please enter a question."

    if "pdf_chunks" not in globals():
        return "Please upload a PDF first."

    try:
        context = search_pdf(question)
        if not context:
            return "No relevant information found."

        return generate_answer(question, context)

    except Exception as error:
        return f"Gemini Error: {error}"


def create_quiz():
    if "pdf_chunks" not in globals():
        return "Please upload a PDF first."

    try:
        context = "\n\n".join(pdf_chunks[:5])
        return generate_quiz(context)

    except Exception as error:
        return f"Quiz Generation Error: {error}"


# Gradio user interface
demo = gr.Blocks()

with demo:
    gr.Markdown("# AI Study Assistant")
    gr.Markdown("Upload your lecture notes and ask questions.")

    pdf_input = gr.File(
        label="Upload Your PDF Notes",
        file_types=[".pdf"],
        type="filepath",
    )

    pdf_output = gr.Textbox(
        label="Extracted PDF Text",
        lines=10,
    )

    pdf_input.upload(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=pdf_output,
    )

    question_input = gr.Textbox(
        label="Ask a Question",
        placeholder="What is Machine Learning?",
    )

    answer_output = gr.Textbox(
        label="AI Generated Answer",
        lines=8,
    )

    ask_button = gr.Button("ASK AI")
    ask_button.click(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output,
    )

    gr.Markdown("## Generate Practice Quiz")
    quiz_button = gr.Button("Generate 3 MCQs")
    quiz_output = gr.Textbox(
        label="Your Practice Quiz",
        lines=15,
    )
    quiz_button.click(
        fn=create_quiz,
        inputs=None,
        outputs=quiz_output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
