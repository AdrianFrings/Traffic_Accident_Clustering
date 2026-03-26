# Use python 3.9 slim as requested
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for geospatial libraries (geopandas/shapely)
# libgdal-dev etc might be needed depending on wheel availability, 
# but usually for modern pip/wheels, basic libs are enough. 
# However, let's include some common ones just in case.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ app/
COPY src/ src/
# Note: We assume data is mounted or processed in the build? 
# The user wants "Offline-First". 
# Usually we run the processor to generate data, OR mount the data volume.
# docker-compose.yml instruction says "mount the ./data volume".
# So we don't COPY data.

# Expose port
EXPOSE 8501

# Command to run the app
# Healthcheck ensures app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
