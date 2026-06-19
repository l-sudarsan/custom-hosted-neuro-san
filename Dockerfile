# Build for the Foundry hosted-agent runtime (Linux AMD64):
#   docker build --platform linux/amd64 -t <acr>.azurecr.io/custom-hosted-neuro-san:latest .
FROM python:3.12-slim

WORKDIR /app
COPY . user_agent/
WORKDIR /app/user_agent

RUN pip install --no-cache-dir -r requirements.txt

# Make the project importable (neuro-san resolves foundry_llm by dotted path).
ENV PYTHONPATH=/app/user_agent
ENV PYTHONUNBUFFERED=1

# Foundry hosted agents serve on port 8088.
EXPOSE 8088
CMD ["python", "main.py"]
