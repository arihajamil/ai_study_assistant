import os 
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai 
import time 

load_dotenv()     #loaf environment variables
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  #load sentence transformer model


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  #get gemini api key
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash"


TITLE ="AI Study Assistant"

DESCRIPTION = """
upload your lecture notes in PDF format. 
Ask questions and generate quizzes 
from your notes"""


# def welcome():
#     return "Welcome to AI Study Assistant! Upload your lecture notes in PDF format, ask questions, and generate quizzes from your notes."

def extract_text_from_pdf(pdf_file):            #function to extract text from PDF
    reader = PdfReader(pdf_file)
    extracted_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            extracted_text +=text + "\n"

    return extracted_text

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def create_vector_store(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index

def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file."




    extracted_text = extract_text_from_pdf(pdf_file)

    if not extracted_text:
        return "No readable text found in the PDF."

    chunks = split_text_into_chunks(extracted_text)
    global pdf_chunks
    pdf_chunks = chunks

    embeddings = create_embeddings(chunks)

    global pdf_vector_store
    pdf_vector_store = create_vector_store(embeddings)

    return "\n\n".join(
        [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)]
)

def search_pdf(question):
    results = search_similar_chunks(
        question,
        pdf_chunks,
        pdf_vector_store
    )

    return "\n\n".join(results)

def generate_answer(question, context):

    prompt = f"""
    You are an AI Study Assistant.

    Answer using ONLY the provided context.

    If the answer is not available in the context,
    say that it is not available in the uploaded notes.

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
                input=prompt
            )

            return response.output_text

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise e
            



def generate_quiz(context):

    prompt = f"""
    You are an AI Quiz Generator.

    Create exactly 3 multiple-choice questions
    using ONLY the provided study notes.

    Each question must have:
    1. Four options (A, B, C, D)
    2. Only one correct answer
    3. Correct answer clearly mentioned
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
                input=prompt
            )

            return response.output_text

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise e



def answer_question(question):

    if not question:
        return "Please enter a question."

    if "pdf_chunks" not in globals():
        return "Please upload a PDF first."

    context = search_pdf(question)

    if not context:
        return "No relevant information found."

    try:
        answer = generate_answer(question, context)
        return answer

    except Exception as e:
        return f"Gemini Error: {str(e)}"



def create_quiz():

    if "pdf_chunks" not in globals():
        return "Please upload a PDF first."

    context = "\n\n".join(pdf_chunks[:5])

    try:
        quiz = generate_quiz(context)
        return quiz

    except Exception as e:
        return f"Quiz Generation Error: {str(e)}"


demo = gr.Blocks()

with demo:
    gr.Markdown("# AI Study Assistant")
    gr.Markdown("Upload your lecture notes and ask questions.")

    pdf_input = gr.File(
        label="Upload Your PDF Notes",
        file_types=[".pdf"],
        type="filepath"
    )

    pdf_output = gr.Textbox(
        label="Extracted PDF Text",
        lines=10
    )

    pdf_input.upload(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=pdf_output
    )

    question_input = gr.Textbox(
        label="Ask a Question",
        placeholder="What is Machine Learning?"
    )

    answer_output = gr.Textbox(
        label="AI Generated Answer",
        lines=8
    )

    ask_button = gr.Button("ASK AI")

    ask_button.click(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output
    )


    gr.Markdown("## Generate Practice Quiz")

    quiz_button = gr.Button("Generate 3 MCQs")

    quiz_output = gr.Textbox(
        label="Your Practice Quiz",
        lines=15
    )

    quiz_button.click(
        fn=create_quiz,
        inputs=None,
        outputs=quiz_output
    )


sample_text = "Artificial Intelligence (AI) helps computers learn and solve problems. " * 20
chunks = split_text_into_chunks(sample_text)
print("Total Chunks:",len(chunks))

for i, chunk in enumerate(chunks,1):
    print(f"\nChunk {i}:")
    print(chunk)

sample_chunks = ["Artificail Intelligence (AI) helps computers learn. ",
                 "Machine Learning is a part of Artificial Intelligence."]


sample_embeddings = create_embeddings(sample_chunks)
print("Embeddings shape:", sample_embeddings.shape)

vector_store = create_vector_store(sample_embeddings)

print("Vector Store Created Successfully!")
print("Total Vectors:", vector_store.ntotal)


def search_similar_chunks(question, chunks, index, top_k=2):
    question_embedding = embedding_model.encode([question])

    distances, indices = index.search(
        question_embedding.astype("float32"), top_k
    )
    results = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    return results 


  
search_results = search_similar_chunks(
    "What is Machine Learning?",
    sample_chunks,
    vector_store
)

print("\nSearch Results:")

for result in search_results:
    print(result)

if __name__ == "__main__":      #Launch application
    demo.launch()
    

