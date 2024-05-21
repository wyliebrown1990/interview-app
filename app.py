import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import select
import faiss
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
database_url = os.getenv('DATABASE_URL')

if not database_url:
    raise ValueError("DATABASE_URL is not set in the environment variables")

# Initialize the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize the database
db = SQLAlchemy(app)

# Define the database model
class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String, nullable=False)
    company_name = db.Column(db.String, nullable=False)
    data = db.Column(db.Text, nullable=False)
    embeddings = db.Column(db.LargeBinary)
    processed_files = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

# Initialize the OpenAI chat model and embeddings model
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# In-memory store for chat histories
chat_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def load_training_data(job_title, company_name):
    return TrainingData.query.filter_by(job_title=job_title, company_name=company_name).first()

def create_chunks_and_embeddings_from_new_files(data_directory_path, processed_files):
    data = ""
    new_files = []
    for filename in os.listdir(data_directory_path):
        if filename.endswith(".txt") and filename not in processed_files:
            new_files.append(filename)
            with open(os.path.join(data_directory_path, filename), "r") as f:
                data += f.read() + "\n"

    if not new_files:
        return None, None, None

    chunk_size = 1000
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    embeddings = embedder.embed_documents(chunks)
    embedding_array = np.array(embeddings).astype('float32')

    return chunks, embedding_array, new_files

def get_initial_question(training_data, industry):
    chunks = training_data.data.split('\n')
    embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an interview coach for a {training_data.job_title} at {training_data.company_name} in the {industry} industry."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = RunnablePassthrough.assign(messages=lambda x: x["messages"]) | prompt | model

    initial_prompt = "Ask a challenging interview question."
    response = chain.invoke({"messages": [HumanMessage(content=initial_prompt)]})

    return response.content

def get_next_question(session_id, user_response):
    session_history = get_session_history(session_id)
    session_history.add_message(HumanMessage(content=user_response))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an interview coach."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Provide feedback on the user's answer and ask the next question."),
    ])

    chain = RunnablePassthrough.assign(messages=lambda x: session_history.messages) | prompt | model

    response = chain.invoke({"messages": session_history.messages})
    next_question = response.content
    session_history.add_message(AIMessage(content=next_question))

    return next_question

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    job_title = request.form['job_title'].strip().lower()
    company_name = request.form['company_name'].strip().lower()
    industry = request.form['industry'].strip().lower()

    training_data = load_training_data(job_title, company_name)
    if training_data:
        initial_question = get_initial_question(training_data, industry)
        session_id = os.urandom(24).hex()
        session_history = get_session_history(session_id)
        session_history.add_message(AIMessage(content=initial_question))
        return render_template('chat.html', question=initial_question, session_id=session_id)
    else:
        return render_template('index.html', message="No training data found. Provide a file path to training data:", job_title=job_title, company_name=company_name)

@app.route('/continue_interview', methods=['POST'])
def continue_interview():
    data = request.get_json()
    user_response = data.get('user_response')
    session_id = data.get('session_id')

    next_question = get_next_question(session_id, user_response)

    return jsonify({'next_question': next_question})

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    file_path = request.form['file_path']
    job_title = request.form['job_title'].strip().lower()
    company_name = request.form['company_name'].strip().lower()

    training_data = load_training_data(job_title, company_name)
    chunks, embedding_array, new_files = create_chunks_and_embeddings_from_new_files(file_path, [])

    if training_data:
        training_data.data += '\n' + '\n'.join(chunks)
        training_data.embeddings = np.concatenate((np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536), embedding_array), axis=0).tobytes()
        training_data.processed_files += ',' + ','.join(new_files)
    else:
        new_training_data = TrainingData(
            job_title=job_title,
            company_name=company_name,
            data='\n'.join(chunks),
            embeddings=embedding_array.tobytes(),
            processed_files=','.join(new_files)
        )
        db.session.add(new_training_data)

    db.session.commit()
    return render_template('index.html', message="Training data uploaded successfully.")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)
