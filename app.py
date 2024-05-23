import os
from dotenv import load_dotenv
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, LargeBinary, TIMESTAMP
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
#from langchain_core.runnables import RunnablePassthrough

"""Here begins debug, environment, database and openai initializations
Don't change"""
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
# Here you can change the model to improve results or lower per token cost
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# In-memory store for chat histories. This allows the chat model to reference previous disucssions. 
chat_histories = {}

# Initialize the OpenAI chat model and embeddings model
# Here you can change the model to improve results or lower per token cost
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# In-memory store for chat histories. This allows the chat model to reference previous disucssions. 
chat_histories = {}

"""
    Retrieve the chat history for a given session ID. If the session ID does not 
    already exist in the chat_histories dictionary, initialize a new ChatMessageHistory 
    object for it.

    Args:
    session_id (str): The unique identifier for the chat session.

    Returns:
    ChatMessageHistory: The chat history associated with the given session ID.
    """
def get_session_history(session_id: str) -> ChatMessageHistory:
    # Check if the session_id exists in chat_histories. If not, create a new entry.
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

"""
    Load the training data from the database for a specific job title and company name.

    Args:
    job_title (str): The job title to filter the training data.
    company_name (str): The company name to filter the training data.

    Returns:
    TrainingData: The training data record from the database that matches the given job title and company name.
    """
def load_training_data(job_title, company_name):
    return session.query(TrainingData).filter_by(job_title=job_title, company_name=company_name).first()

"""
    Read a text file, split its content into chunks, and generate embeddings for each chunk.

    Args:
    file_path (str): The path to the text file to be processed.

    Returns:
    tuple: A tuple containing the list of chunks and a NumPy array of their embeddings.
    """
def create_chunks_and_embeddings_from_file(file_path):
    logging.debug(f"Processing file: {file_path}")
    with open(file_path, "r") as f:
        data = f.read()

# Define the size of each chunk.
    chunk_size = 1000
# Split the data into chunks of the specified size.
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
# Generate embeddings for each chunk using the embedder object.
    embeddings = embedder.embed_documents(chunks)
# Convert the list of embeddings to a NumPy array of type float32.
    embedding_array = np.array(embeddings).astype('float32')
# Log the number of chunks and embeddings created for debugging purposes.
    logging.debug(f"Created {len(chunks)} chunks and embeddings")
# Return the chunks and their corresponding embeddings.
    return chunks, embedding_array

"""
    Render the index page of the application.

    This function is mapped to the root URL ('/') of the application. 
    When a user accesses the root URL, it renders the 'index.html' template.
    
    Returns:
    The rendered 'index.html' template.
    """
@app.route('/')
def index():
    return render_template('index.html')

"""
    Handle the form submission to start an interview.

    This function processes the form data submitted by the user to start an interview. 
    It retrieves the job title, company name, and industry from the form, 
    checks if training data exists for the given job title and company name,
    and then renders the appropriate template based on whether the training data exists.

    Returns:
    The rendered 'add_training_data.html' template with a message indicating whether training data exists.
    """
@app.route('/start_interview', methods=['POST'])
def start_interview():
# Retrieve and sanitize form data: job title, company name, and industry.
    job_title = request.form['job_title'].strip().lower()
    company_name = request.form['company_name'].strip().lower()
    industry = request.form['industry'].strip().lower()

# Load training data from the database for the given job title and company name.
    training_data = load_training_data(job_title, company_name)
    if training_data:
        # If training data exists, render 'add_training_data.html' with a message and a skip URL.
        return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message="Training data already exists. Do you want to add more training data?", skip_url=url_for('start_interview_without_adding', job_title=job_title, company_name=company_name, industry=industry))
    else:
        # If no training data is found, render 'add_training_data.html' with a message prompting for a file path.
        return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message="No training data found. To improve results please provide a file path to training data:")

"""
    Start the interview without adding new training data.

    This function handles the GET request to start an interview without adding new training data. 
    It retrieves the job title, company name, and industry from the query parameters, 
    checks if training data exists, and if so, initializes the interview with the first question.
    
    Returns:
    The rendered 'chat.html' template with the initial interview question if training data exists,
    otherwise renders the 'index.html' template with a message prompting for training data.
    """
@app.route('/start_interview_without_adding', methods=['GET'])
def start_interview_without_adding():
    # Retrieve and sanitize query parameters: job title, company name, and industry.
    job_title = request.args.get('job_title').strip().lower()
    company_name = request.args.get('company_name').strip().lower()
    industry = request.args.get('industry').strip().lower()

# Load training data from the database for the given job title and company name.
    training_data = load_training_data(job_title, company_name)
    if training_data:
        # If training data exists, get the initial interview question.
        initial_question = get_initial_question(training_data, industry)
        # Generate a unique session ID for the interview session.
        session_id = os.urandom(24).hex()
        # Retrieve the session history for the generated session ID.
        session_history = get_session_history(session_id)
        # Add the initial question to the session history as an AI message.
        session_history.add_message(AIMessage(content=initial_question))
        # Render the 'chat.html' template with the initial question and session ID.
        return render_template('chat.html', question=initial_question, session_id=session_id)
    else:
        # If no training data is found, render the 'index.html' template with a message prompting for training data.
        return render_template('index.html', message="No training data found. To improve results please provide a file path to training data:", job_title=job_title, company_name=company_name)

"""
    Upload training data from provided file path, process and store it in the database.

    This function handles the POST request to upload training data. It processes the files in the provided
    file path, creates embeddings for the text data, and updates the database with the new training data.

    Returns:
    A redirect to the interview start page or the add training data page with an error message if the file path is invalid.
"""
"""
Expected Output:
job_title: 'sales engineer'
company_name: 'observe inc.'
data: Long text data representing all concatenated chunks from processed files.
embeddings: Binary data representing the concatenated embeddings.
processed_files: Comma-separated list of file names that have been processed.
"""
@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    file_path = request.form['file_path']
    job_title = request.form['job_title'].strip().lower()
    company_name = request.form['company_name'].strip().lower()
    industry = request.form['industry'].strip().lower()

    logging.debug(f"Received upload request for job_title={job_title}, company_name={company_name}, industry={industry}")

    if not os.path.exists(file_path):
        logging.error(f"Provided file path does not exist: {file_path}")
        return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message=f"Provided file path does not exist: {file_path}")

    training_data = load_training_data(job_title, company_name)
    existing_files = training_data.processed_files.split(',') if training_data and training_data.processed_files else []

    new_files = []
    for filename in os.listdir(file_path):
        full_path = os.path.join(file_path, filename)
        logging.debug(f"Checking file: {full_path}")
        if filename.endswith(".txt") and filename not in existing_files:
            chunks, embedding_array = create_chunks_and_embeddings_from_file(full_path)
            new_files.append(filename)

            if training_data:
                logging.debug(f"Updating existing training data for {job_title} at {company_name}")
                training_data.data += '\n' + '\n'.join(chunks)
                existing_embeddings = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
                logging.debug(f"Existing embeddings shape: {existing_embeddings.shape}")
                logging.debug(f"New embedding array shape: {embedding_array.shape}")
                if embedding_array.size > 0:
                    if len(embedding_array.shape) == 1:
                        embedding_array = embedding_array.reshape(1, -1)
                    training_data.embeddings = np.concatenate((existing_embeddings, embedding_array), axis=0).tobytes()
                    logging.debug(f"Updated embeddings shape: {np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536).shape}")
                training_data.processed_files += ',' + filename
                logging.debug(f"Updated processed files: {training_data.processed_files}")
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
                logging.debug(f"New training data created with ID: {new_training_data.id}")

    try:
        logging.debug("Committing changes to the database...")
        session.commit()
        logging.debug("Changes committed to the database.")
    except Exception as e:
        logging.error(f"Error committing changes to the database: {e}")
        session.rollback()
        return render_template('add_training_data.html', job_title=job_title, company_name=company_name, industry=industry, message=f"Error saving training data: {e}")

    updated_training_data = load_training_data(job_title, company_name)
    logging.debug(f"Post-commit Training Data ID: {updated_training_data.id}")
    logging.debug(f"Post-commit Training Data Processed Files: {updated_training_data.processed_files}")

    return redirect(url_for('start_interview_without_adding', job_title=job_title, company_name=company_name, industry=industry))

"""
    Generates an initial interview question using the provided training data and industry context.

    Args:
    training_data (TrainingData): The training data containing job title, company name, data, and embeddings.
    industry (str): The industry context for the interview.

    Returns:
    str: The initial interview question generated by the model.
    """
def get_initial_question(training_data, industry):
    # Split the training data into chunks and load the embeddings.
    chunks = training_data.data.split('\n')
    embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
    logging.debug(f"Loaded {len(chunks)} chunks and {embedding_array.shape[0]} embeddings for FAISS index")
    # Create a FAISS index for the embeddings.
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)
    logging.debug("FAISS index created and embeddings added")

# Define the prompt template for the initial interview question. This is where you can really optimize output.
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"I want you to conduct a real life job interview with me where you ask me real interview questions I would get as a {training_data.job_title} at {training_data.company_name} in the {industry} industry. Youre questions should test my knowledge of the job role and company and challenge me to give concise and relevant answers."),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Create a chain to process the initial prompt.
    chain = prompt | model

    initial_prompt = "Ask a challenging interview question."
    response = chain.invoke({"messages": [HumanMessage(content=initial_prompt)]})
    logging.debug(f"Initial interview question: {response.content}")

    return response.content

"""
    Generates the next interview question based on the user's response.

    Args:
    session_id (str): The session ID for the interview.
    user_response (str): The user's response to the previous question.

    Returns:
    str: The next interview question generated by the model.
"""
def get_next_question(session_id, user_response):
# Retrieve the session history and add the user's response to it.
    session_history = get_session_history(session_id)
    session_history.add_message(HumanMessage(content=user_response))

# Define the prompt template for the next interview question. This is where you can really optimize output. 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the world's best interview coach. People looking to advance their careers and perfect their interview answers come to you for critical feedback on their ability to answer interview questions as best as possible."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Provide feedback on the user's answer. Be very critical of their ability to provide concise and accurate answers. Give them a rating between 0 - 10 on how good their answer was based on how you would expect the world's best job interviewers to perform. After your feedback if you felt the answer given was incomplete then ask another question that goes deeper and more specific on your last question. If the answer is satisfactory to the last question then please ask a new question that is different than any questions you've asked before."),
    ])

# Create a chain to process the user's response and generate the next question.
    chain = prompt | model

# Generate the next interview question.
    response = chain.invoke({"messages": session_history.messages})
    next_question = response.content
    session_history.add_message(AIMessage(content=next_question))

    logging.debug(f"Next interview question: {next_question}")

    return next_question

"""
    Continues the interview by generating the next question based on the user's response.

    This endpoint handles the POST request to continue the interview. It processes the user's response
    and generates the next interview question.

    Returns:
    A JSON response containing the next interview question.
"""
@app.route('/continue_interview', methods=['POST'])
def continue_interview():
# Extract the user's response and session ID from the request data.
    data = request.get_json()
    user_response = data.get('user_response')
    session_id = data.get('session_id')

# Generate the next interview question.
    next_question = get_next_question(session_id, user_response)

    return jsonify({'next_question': next_question})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
