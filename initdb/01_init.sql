-- Create the training_data table
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    data TEXT NOT NULL,
    embeddings BYTEA,
    processed_files TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Additional indexes or constraints can be added here if needed
-- For example, you might want to create an index on job_title or company_name for faster queries
