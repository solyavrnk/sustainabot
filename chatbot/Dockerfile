FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY chatbot/ .

# Create a non-root user and set up permissions
RUN useradd -m dev && \
    mkdir -p /app/data && \
    touch /app/data/chat_history.json && \
    chown -R dev:dev /app && \
    chmod +x start.sh

USER dev

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run services with output redirection
CMD ["./start.sh"]