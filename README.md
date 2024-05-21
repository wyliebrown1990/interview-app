# Fine Tune an AI Bot to do Interview Prep

## This project uses the following technologies:
1. Langchain for retrieval and chaining
2. OpenAI Chatmodel
3. FAISS vectorstore 
4. Postgres db for persistent training data
5. Flask app server
6. Poll model to update front end with chat bot responses

## The documentation used:
* https://python.langchain.com/v0.2/docs/tutorials/chatbot/

### How to set up this project locally:
-  Clone github repo locally
-  Create your postgres database 
```sql
CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    data TEXT NOT NULL,
    embeddings BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_files TEXT
);
```

-  Create a .env environment for you OPENAI_API_KEY and DATABASE_URL
-  Install dependencies:

   ```sh
pip install -r requirements.txt

-  Prepare a file in your directory with .txt documents holding the information you want your model trained on. I will be launching an update later on this project with youtube transcription bot, web scraper and a user friendly way to build out this folder. 
-  run the python file and when prompted provided the file path to your training data. You can always add more data later on and the code will only process new files.txt it hasn't stored in the database previously. 
-  You will now see 2 new files in your directory: embedding.npy and chunks.txt these were created from the files in the file path you provided. 
-  Start chatting, feel free to ask bot to reference previous converation, continue getting interview questions until you are satisfied. # interview-bot

