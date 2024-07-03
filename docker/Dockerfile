# Use the official Ubuntu base image
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && apt-get clean

# Install Jupyter Notebook and other Python packages
RUN pip3 install --no-cache-dir jupyter numpy pandas matplotlib scipy scikit-learn

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Set environment variables
ENV JUPYTER_TOKEN=mysecretpassword

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=${JUPYTER_TOKEN}"]
