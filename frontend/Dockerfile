#FROM python:3.11-slim
#WORKDIR /app
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#COPY app.py .
#EXPOSE 8501
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Step 1: Use the official Python slim image AS a base builder
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies into a virtual environment
# This isolates dependencies from the system's Python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# Step 2: Build the final, smaller image from the same base
FROM python:3.11-slim AS final

WORKDIR /app

# Create a non-root user and group for security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY app.py .

# Make the virtual environment's Python the default
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the application
# --server.address=0.0.0.0 is crucial to allow access from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]