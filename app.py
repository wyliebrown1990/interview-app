import os
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

Base.metadata.create_all(engine)

# Initialize the OpenAI chat model and embeddings model
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
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
                initial_question = get_initial_question(training_data, industry)
                session_id = os.urandom(24).hex()
                session_history = get_session_history(session_id)
                session_history.add_message(AIMessage(content=initial_question))
                logging.debug(f"Initial question generated: {initial_question}")
                return render_template('chat.html', question=initial_question, session_id=session_id)
            else:
                logging.error(f"No training data found for job_title={job_title}, company_name={company_name}")
                return render_template('index.html', message="No training data found. To improve results please provide a file path to training data:", job_title=job_title, company_name=company_name)
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

        logging.debug(f"Received upload request for job_title={job_title}, company_name={company_name}, industry={industry}")

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
                        logging.debug(f"Updating existing training data for {job_title} at {company_name}")
                        training_data.data += '\n' + '\n'.join(chunks)
                        existing_embeddings = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
                        if embedding_array.size > 0:
                            if len(embedding_array.shape) == 1:
                                embedding_array = embedding_array.reshape(1, -1)
                            training_data.embeddings = np.concatenate((existing_embeddings, embedding_array), axis=0).tobytes()
                        training_data.processed_files += ',' + filename
                    else:
                        logging.debug(f"Creating new training data for {job_title} at {company_name}")
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
            logging.debug("Committing changes to the database...")
            session.commit()
            logging.debug("Changes committed to the database.")
        except Exception as e:
            logging.error(f"Error committing changes to the database: {e}")
            session.rollback()
            span.set_status(StatusCode.ERROR, str(e))
            return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message=f"Error saving training data: {e}")

        return redirect(url_for('start_interview_without_adding', job_title=job_title, company_name=company_name, industry=industry))

def get_initial_question(training_data, industry):
    with tracer.start_as_current_span("get_initial_question") as span:
        chunks = training_data.data.split('\n')
        embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
        logging.debug(f"Loaded {len(chunks)} chunks and {embedding_array.shape[0]} embeddings for FAISS index")
        dimension = embedding_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_array)
        logging.debug("FAISS index created and embeddings added")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"I want you to conduct a real life job interview with me where you ask me real interview questions I would get as a {training_data.job_title} at {training_data.company_name} in the {industry} industry. Your questions should test my knowledge of the job role and company and challenge me to give concise and relevant answers."),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | model

        initial_prompt = "Ask a challenging interview question."
        response = chain.invoke({"messages": [HumanMessage(content=initial_prompt)]})
        logging.debug(f"Initial interview question: {response.content}")

        return response.content

def get_next_question(session_id, user_response):
    with tracer.start_as_current_span("get_next_question") as span:
        session_history = get_session_history(session_id)
        session_history.add_message(HumanMessage(content=user_response))

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the world's best interview coach. People looking to advance their careers and perfect their interview answers come to you for critical feedback on their ability to answer interview questions as best as possible."),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Provide feedback on the user's answer. Be very critical of their ability to provide concise and accurate answers. Give them a rating between 0 - 10 on how good their answer was based on how you would expect the world's best job interviewers to perform. After your feedback if you felt the answer given was incomplete then ask another question that goes deeper and more specific on your last question. If the answer is satisfactory to the last question then please ask a new question that is different than any questions you've asked before."),
        ])

        chain = prompt | model

        response = chain.invoke({"messages": session_history.messages})
        next_question = response.content
        session_history.add_message(AIMessage(content=next_question))

        logging.debug(f"Next interview question: {next_question}")

        return next_question

@app.route('/continue_interview', methods=['POST'])
def continue_interview():
    with tracer.start_as_current_span("continue_interview") as span:
        data = request.get_json()
        user_response = data.get('user_response')
        session_id = data.get('session_id')

        next_question = get_next_question(session_id, user_response)

        return jsonify({'next_question': next_question})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)