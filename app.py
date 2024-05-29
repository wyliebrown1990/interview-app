import os
import re
from dotenv import load_dotenv
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import numpy as np
import faiss
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from werkzeug.utils import secure_filename
#opentelemetry modules
from opentelemetry import trace
from opentelemetry.trace.status import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
#audio transcription
from pydub import AudioSegment
import whisper

# Load the Whisper model - there are english only models we can test out in the future.
model = whisper.load_model("tiny.en")

#start otel config

# Resource configuration for tracing
resource = Resource(attributes={
    "service.name": "Wylies-MacBook-Air",
    "os-version": 14.1,
    "cluster": "A",
    "datacentre": "us-east-1a"
})

# Configure the OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'otel-collector:4317'),  # Default to 'otel-collector:4317' if not set
    insecure=True  # Use TLS in production environments
)


# Set up OpenTelemetry Tracer Provider with OTLP exporter
provider = TracerProvider(resource=resource)
otlp_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(otlp_processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my.tracer.name")

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
database_url = os.getenv('DATABASE_URL')

#Output any issue with database env variable
if not database_url:
   raise ValueError("DATABASE_URL is not set in the environment variables")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/code/uploads'
app.config['MAX_CONTENT_PATH'] = 100000

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize SQLAlchemy
engine = create_engine(database_url)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define the database model
class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    job_title = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    data = Column(Text, nullable=False)
    embeddings = Column(LargeBinary, nullable=True)
    processed_files = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

# Define the database model for InterviewAnswer
class InterviewAnswer(Base):
    __tablename__ = 'interview_answers'
    id = Column(Integer, primary_key=True)
    job_title = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    critique = Column(Text, nullable=False)
    score = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


Base.metadata.create_all(engine)

# Initialize the OpenAI chat model and embeddings model with temperature adjustment
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.5)
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# In-memory store for chat histories. This allows the chat model to reference previous disucssions.
chat_histories = {}

# Helper functions with tracing
def get_session_history(session_id: str) -> ChatMessageHistory:
    with tracer.start_as_current_span("get_session_history") as span:
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        return chat_histories[session_id]

def load_training_data(job_title, company_name):
    with tracer.start_as_current_span("load_training_data") as span:
        return session.query(TrainingData).filter_by(job_title=job_title, company_name=company_name).first()

def create_chunks_and_embeddings_from_file(file_path):
    with tracer.start_as_current_span("create_chunks_and_embeddings_from_file") as span:
        logging.debug(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            data = f.read()
        chunk_size = 1000
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        embeddings = embedder.embed_documents(chunks)
        embedding_array = np.array(embeddings).astype('float32')
        logging.debug(f"Created {len(chunks)} chunks and embeddings")
        return chunks, embedding_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    with tracer.start_as_current_span("start_interview") as span:
        job_title = request.form['job_title'].strip().lower()
        company_name = request.form['company_name'].strip().lower()
        industry = request.form['industry'].strip().lower()

        training_data = load_training_data(job_title, company_name)
        if training_data:
            return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message="Training data already exists. Do you want to add more training data?", skip_url=url_for('start_interview_without_adding', job_title=job_title, company_name=company_name, industry=industry))
        else:
            return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message="No training data found. To improve results please upload a file with training data:")

@app.route('/start_interview_without_adding', methods=['GET'])
def start_interview_without_adding():
    with tracer.start_as_current_span("start_interview_without_adding") as span:
        try:
            job_title = request.args.get('job_title').strip().lower()
            company_name = request.args.get('company_name').strip().lower()
            industry = request.args.get('industry').strip().lower()

            logging.debug(f"Starting interview without adding for job_title={job_title}, company_name={company_name}, industry={industry}")

            training_data = load_training_data(job_title, company_name)
            if training_data:
                initial_question = get_initial_question(training_data, industry, job_title, company_name)
                session_id = os.urandom(24).hex()
                session_history = get_session_history(session_id)
                session_history.add_message(AIMessage(content=initial_question))
                logging.debug(f"Initial question generated: {initial_question}")
                return render_template('chat.html', question=initial_question, session_id=session_id, job_title=job_title, company_name=company_name, industry=industry)
            else:
                logging.error(f"No training data found for job_title={job_title}, company_name={company_name}")
                return render_template('index.html', message="No training data found. To improve results please provide a file path to training data:", job_title=job_title, company_name=company_name, industry=industry)
        except Exception as e:
            logging.error(f"Error in start_interview_without_adding: {e}")
            span.set_status(StatusCode.ERROR, str(e))
            return render_template('index.html', message=f"An error occurred: {e}")

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    with tracer.start_as_current_span("upload_training_data") as span:
        job_title = request.form['job_title'].strip().lower()
        company_name = request.form['company_name'].strip().lower()
        industry = request.form['industry'].strip().lower()
        files = request.files.getlist('files')

        log_messages = []
        log_messages.append(f"Received upload request for job_title={job_title}, company_name={company_name}, industry={industry}")

        for file in files:
            if file and file.filename.endswith('.txt'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                training_data = load_training_data(job_title, company_name)
                existing_files = training_data.processed_files.split(',') if training_data and training_data.processed_files else []

                if filename not in existing_files:
                    chunks, embedding_array = create_chunks_and_embeddings_from_file(file_path)
                    if training_data:
                        log_messages.append(f"Updating existing training data for {job_title} at {company_name}")
                        training_data.data += '\n' + '\n'.join(chunks)
                        existing_embeddings = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
                        if embedding_array.size > 0:
                            if len(embedding_array.shape) == 1:
                                embedding_array = embedding_array.reshape(1, -1)
                            training_data.embeddings = np.concatenate((existing_embeddings, embedding_array), axis=0).tobytes()
                        training_data.processed_files += ',' + filename
                    else:
                        log_messages.append(f"Creating new training data for {job_title} at {company_name}")
                        new_training_data = TrainingData(
                            job_title=job_title,
                            company_name=company_name,
                            data='\n'.join(chunks),
                            embeddings=embedding_array.tobytes(),
                            processed_files=filename
                        )
                        session.add(new_training_data)
                        training_data = new_training_data

        try:
            log_messages.append("Committing changes to the database...")
            session.commit()
            log_messages.append("Changes committed to the database.")
        except Exception as e:
            logging.error(f"Error committing changes to the database: {e}")
            session.rollback()
            span.set_status(StatusCode.ERROR, str(e))
            return jsonify({'error': f"Error saving training data: {e}", 'logs': log_messages}), 500

        return jsonify({'message': 'Training data uploaded successfully', 'logs': log_messages})

#Function to query the FAISS index:
def query_faiss_index(index, embedding_array, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)  # Search the top k nearest neighbors
    return [embedding_array[i] for i in I[0]]

#Initial question uses the prompt and Faiss index query data: 
def get_initial_question(training_data, industry, job_title, company_name):
    with tracer.start_as_current_span("get_initial_question") as span:
        chunks = training_data.data.split('\n')
        embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
        logging.debug(f"Loaded {len(chunks)} chunks and {embedding_array.shape[0]} embeddings for FAISS index")
        dimension = embedding_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_array)
        logging.debug("FAISS index created and embeddings added")
        
        # Create a dynamic query using the job title, company name, and industry
        query_text = f"How would a {job_title} demonstrate knowledge of {company_name} in the {industry} industry?"
        example_query_embedding = embedder.embed_query(query_text)
        
        relevant_embeddings = query_faiss_index(index, embedding_array, example_query_embedding)
        relevant_chunks = [chunks[np.where(embedding_array == emb)[0][0]] for emb in relevant_embeddings]
        relevant_context = " ".join(relevant_chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are helping me land a new job by conducting a mock interview with me. You should ask me a new question each time that is related to {job_title} job role at {company_name} company in the {industry} industry. Your questions should test my knowledge of the {job_title} job role and {company_name} company. You should challenge me to give concise and relevant answers. Here is some context about {company_name}: {relevant_context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | model

        initial_prompt = "Ask a challenging interview question."
        response = chain.invoke({"messages": [HumanMessage(content=initial_prompt)]})
        logging.debug(f"Initial interview question: {response.content}")

        return response.content

def get_next_question(session_id, user_response, job_title, company_name, industry):
    with tracer.start_as_current_span("get_next_question") as span:
        logging.debug(f"Getting next question for job_title={job_title}, company_name={company_name}, industry={industry}")

        session_history = get_session_history(session_id)
        session_history.add_message(HumanMessage(content=user_response))

        training_data = load_training_data(job_title, company_name)
        if not training_data:
            logging.error(f"No training data found for job_title={job_title}, company_name={company_name}")
            raise ValueError("No training data found for the specified job title and company name.")
        
        # Load FAISS index
        chunks = training_data.data.split('\n')
        embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
        dimension = embedding_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_array)

        # Query FAISS index for fact-checking the user response
        response_embedding = embedder.embed_query(user_response)
        relevant_embeddings_for_fact_checking = query_faiss_index(index, embedding_array, response_embedding)
        relevant_chunks_for_fact_checking = [chunks[np.where(embedding_array == emb)[0][0]] for emb in relevant_embeddings_for_fact_checking]
        relevant_context_for_fact_checking = " ".join(relevant_chunks_for_fact_checking)

        # Fact-check the user's response
        fact_check_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Give me critical feedback on how well I answered your last question. Specifically call out the following: Was my answer concise? Did I provide a STAR (Situation, Task, Action, and Result) formatted answer? Did I use too many filler words? Did I provide an answer that was specific to being a {job_title} at {company_name}? Finally, you must always score my answer from between 0 being the worst answer and 10 being the best. Always return a score out of 10. Once you complete giving feedback, move on to ask me a new question you haven’t asked before in this interview. If you need additional context about being a {job_title} at {company_name}, use this: {relevant_context_for_fact_checking}"),
            ("user", user_response),
            MessagesPlaceholder(variable_name="messages"),
        ])

        fact_check_chain = fact_check_prompt | model
        fact_check_response = fact_check_chain.invoke({"messages": session_history.messages})
        fact_check_feedback = fact_check_response.content
        logging.debug(f"Fact check feedback: {fact_check_feedback}")

        # Extract score from feedback
        score = extract_score(fact_check_feedback)
        logging.debug(f"Extracted score: {score}")

        # Query FAISS index for generating the next question
        relevant_embeddings_for_next_question = query_faiss_index(index, embedding_array, response_embedding)
        relevant_chunks_for_next_question = [chunks[np.where(embedding_array == emb)[0][0]] for emb in relevant_embeddings_for_next_question]
        relevant_context_for_next_question = " ".join(relevant_chunks_for_next_question)

        # Generate the next interview question
        next_question_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are helping me land a new job by conducting a mock interview with me. You should ask me a new question each time that is related to {job_title} job role at {company_name} company. You should reference this Context: {relevant_context_for_next_question}"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        next_question_chain = next_question_prompt | model
        next_question_response = next_question_chain.invoke({"messages": session_history.messages})
        next_question = next_question_response.content
        session_history.add_message(AIMessage(content=next_question))

        logging.debug(f"Next interview question: {next_question}")

        # Save to database
        new_answer = InterviewAnswer(
            job_title=job_title,
            company_name=company_name,
            industry=industry,
            question=session_history.messages[0].content,  # Use the first message in the session history as the question
            answer=user_response,
            critique=fact_check_feedback,
            score=score
        )
        logging.debug(f"Saving to database: {new_answer}")
        session.add(new_answer)
        session.commit()

        return {
            "next_question": next_question,
            "fact_check_feedback": fact_check_feedback,
            "score": score
        }

def extract_score(feedback):
    match = re.search(r"\b(\d{1,2})\b", feedback)
    if match:
        return match.group(1)
    else:
        return "Score not found"

def extract_critique(feedback):
    # Implement logic to extract the critique from the feedback text
    return feedback  # Adjust this based on your needs


@app.route('/continue_interview', methods=['POST'])
def continue_interview():
    with tracer.start_as_current_span("continue_interview") as span:
        data = request.get_json()
        user_response = data.get('user_response')
        session_id = data.get('session_id')
        job_title = data.get('job_title')
        company_name = data.get('company_name')
        industry = data.get('industry')

        logging.debug(f"Continue interview called with job_title={job_title}, company_name={company_name}, industry={industry}")

        next_question_and_feedback = get_next_question(session_id, user_response, job_title, company_name, industry)

        return jsonify(next_question_and_feedback)

   
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    with tracer.start_as_current_span("submit_answer") as span:
        try:
            data = request.get_json()
            job_title = data.get('job_title')
            company_name = data.get('company_name')
            industry = data.get('industry')
            question = data.get('question')
            answer = data.get('answer')

            new_response = InterviewResponse(
                job_title=job_title,
                company_name=company_name,
                industry=industry,
                question=question,
                answer=answer
            )
            session.add(new_response)
            session.commit()

            return jsonify({'message': 'Answer saved successfully'})
        except Exception as e:
            logging.error(f"Error in submit_answer: {e}")
            span.set_status(StatusCode.ERROR, str(e))
            return jsonify({'message': f"An error occurred: {e}"}), 500

import whisper

# Load the Whisper model
whisper_model = whisper.load_model("tiny.en")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    with tracer.start_as_current_span("upload_audio") as span:
        try:
            logging.info(f"Request headers: {request.headers}")
            logging.info(f"Form data: {request.form}")

            if 'audio_data' not in request.files:
                raise ValueError("No audio_data file in request")

            file = request.files['audio_data']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.webm')
            logging.info(f"Received audio file: {file.filename}")
            file.save(file_path)
            logging.info(f"Audio file saved at {file_path}")
            
            # Convert audio file to WAV format using pydub
            audio = AudioSegment.from_file(file_path)
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
            audio.export(wav_path, format='wav')
            logging.info(f"Audio file converted to WAV format at {wav_path}")

            # Transcribe audio using Whisper
            logging.info("Starting transcription with Whisper...")
            transcription = whisper_model.transcribe(wav_path)
            text = transcription['text']
            logging.info(f"Transcription result: {text}")

            # Save the transcription to the database
            job_title = request.form.get('job_title')
            company_name = request.form.get('company_name')
            industry = request.form.get('industry')
            interview_question = request.form.get('interview_question')
            
            new_answer = InterviewAnswer(
                job_title=job_title,
                company_name=company_name,
                industry=industry,
                question=interview_question,
                answer=text
            )
            session.add(new_answer)
            session.commit()

            return jsonify({'transcription': text})
        except Exception as e:
            logging.error(f"Error in upload_audio: {e}")
            span.set_status(StatusCode.ERROR, str(e))
            return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
