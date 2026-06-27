# Docker Deployment Guide - Agentic Self-RAG

## Overview

Docker containers are **already implemented and configured** in this project. This guide explains the existing Docker setup and how to use it.

## Current Docker Configuration

### Files Present
- ✅ `Dockerfile` - Multi-stage Docker image definition
- ✅ `docker-compose.yml` - Container orchestration configuration
- ✅ `.dockerignore` - Build optimization (if present)

### What's Included in Docker

The Docker image includes:
- Python 3.10 slim base
- All dependencies from `requirements.txt`
- spaCy language model (`en_core_web_sm`)
- Complete source code
- Pre-created results directory
- Configured environment variables

## Quick Start

### 1. Build the Docker Image

```bash
# Build the image with docker-compose
docker-compose build

# Or build directly with Docker
docker build -t agentic-self-rag:latest .
```

### 2. Run Containers

#### Option A: Run Main Evaluation (500 samples)
```bash
docker-compose up agentic-self-rag
```

#### Option B: Run Ablation Study
```bash
docker-compose up ablation-study
```

#### Option C: Run Smoke Test (5 samples for quick validation)
```bash
docker-compose up smoke-test
```

#### Option D: Run Individual Container
```bash
docker run -v $(pwd)/results:/app/results \
           -v $(pwd)/data:/app/data \
           -e GROQ_API_KEY=your_key_here \
           agentic-self-rag:latest
```

### 3. View Results

After execution completes, results are available in:
```
./results/
├── main_evaluation/
│   ├── live_results_main.csv
│   ├── results_500.csv
│   ├── metadata.json
│   ├── summary.json
│   └── statistical_comparisons.json
├── paper_tables/
│   ├── table_*.tex
│   └── table_*.md
├── table_visualizations/
│   └── table_*.png
└── logs/
```

## Configuration

### Environment Variables

Set in `docker-compose.yml` or `.env` file:

```yaml
environment:
  - GROQ_API_KEY=${GROQ_API_KEY}        # Required: API key for LLM
  - NUM_SAMPLES=${NUM_SAMPLES:-500}     # Optional: Number of samples (default: 500)
  - PYTHONUNBUFFERED=1                  # Ensure real-time logs
  - PYTHONHASHSEED=42                   # Reproducibility seed
```

### Volume Mounts

Data persistence through volumes:

```yaml
volumes:
  - ./results:/app/results              # Output results
  - ./data:/app/data                    # Input data
```

## Advanced Usage

### 1. Custom Configuration

Create `.env` file:
```env
GROQ_API_KEY=your_api_key_here
NUM_SAMPLES=500
```

Run with custom settings:
```bash
docker-compose up --env-file .env
```

### 2. Interactive Shell

```bash
docker-compose run agentic-self-rag bash
```

### 3. Scale Multiple Instances

For parallel evaluation:
```bash
docker-compose up --scale agentic-self-rag=3
```

### 4. View Logs

```bash
# Real-time logs
docker-compose logs -f agentic-self-rag

# Logs from specific container
docker logs agentic-self-rag

# Last 100 lines
docker logs --tail 100 agentic-self-rag
```

### 5. Clean Up

```bash
# Stop containers
docker-compose down

# Remove images
docker rmi agentic-self-rag:latest

# Remove volumes
docker-compose down -v

# Full cleanup
docker system prune -a
```

## Services Available

### 1. agentic-self-rag (Main)
- **Purpose**: Full evaluation on 500 samples
- **Duration**: 15-30 minutes depending on hardware
- **Output**: Complete results with all metrics
- **Command**: `docker-compose up agentic-self-rag`

### 2. ablation-study
- **Purpose**: Ablation study on system components
- **Duration**: 20-40 minutes
- **Output**: Component importance analysis
- **Command**: `docker-compose up ablation-study`

### 3. smoke-test
- **Purpose**: Quick validation with 5 samples
- **Duration**: 1-2 minutes
- **Output**: Quick validation results
- **Command**: `docker-compose up smoke-test`

## Requirements for Docker

### System Requirements
- Docker Engine 20.10+
- Docker Compose 1.29+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space
- GPU support (optional, for faster inference)

### API Requirements
- Valid GROQ_API_KEY environment variable
- Internet connection for LLM API calls

## Docker Compose Commands

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# Start specific service
docker-compose up agentic-self-rag

# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# View service status
docker-compose ps

# View logs
docker-compose logs -f

# Run one-off command
docker-compose run agentic-self-rag python --version
```

## Troubleshooting

### Issue: Container exits immediately
**Solution**: Check environment variables and API keys
```bash
docker-compose logs agentic-self-rag
```

### Issue: Disk space error
**Solution**: Clean up Docker resources
```bash
docker system prune -a
docker volume prune
```

### Issue: API connection error
**Solution**: Verify GROQ_API_KEY is set correctly
```bash
echo $GROQ_API_KEY
docker-compose config  # Shows resolved environment
```

### Issue: Mount permission denied
**Solution**: Fix volume ownership
```bash
docker-compose down
sudo chown -R $USER:$USER ./results ./data
docker-compose up
```

## Performance Optimization

### 1. Use GPU Acceleration

Modify `docker-compose.yml`:
```yaml
services:
  agentic-self-rag:
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### 2. Increase Resource Limits

```yaml
services:
  agentic-self-rag:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
```

### 3. Optimize Build Time

Use layer caching:
```bash
docker-compose build --no-cache  # Force rebuild
docker-compose build             # Use cache if available
```

## Production Deployment

### 1. Create Dockerfile for Production

The current Dockerfile is suitable for production. Ensure:
- ✅ Uses slim base image
- ✅ Minimal layer count
- ✅ Proper error handling
- ✅ Health checks included

### 2. Add Health Checks

```yaml
services:
  agentic-self-rag:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Logging Configuration

```yaml
services:
  agentic-self-rag:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Security Considerations

### 1. Secrets Management
```bash
# Use Docker secrets (production)
docker secret create groq_api_key -

# Or use environment file with restricted permissions
chmod 600 .env
docker-compose --env-file .env up
```

### 2. Image Security

```bash
# Scan image for vulnerabilities
docker scan agentic-self-rag:latest

# Sign image
docker trust signer add my-key agentic-self-rag:latest
```

### 3. Network Isolation

```yaml
services:
  agentic-self-rag:
    networks:
      - agentic-network

networks:
  agentic-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_ip_masquerade: 'true'
```

## Monitoring and Logging

### 1. Container Stats

```bash
docker stats agentic-self-rag
```

### 2. Detailed Logs

```bash
# All logs
docker logs agentic-self-rag

# Follow logs in real-time
docker logs -f agentic-self-rag

# Timestamps included
docker logs -t agentic-self-rag

# Last N lines
docker logs --tail 50 agentic-self-rag
```

### 3. Log Aggregation

For production, use:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- CloudWatch (AWS)
- Stackdriver (GCP)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Test

on: [push]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: docker/build-push-action@v2
        with:
          context: .
          push: false
          file: ./Dockerfile
```

## FAQ

**Q: Do I need Docker to run this project?**
A: No, Docker is optional. You can run directly with Python 3.10+ installed.

**Q: Can I modify the Dockerfile?**
A: Yes! Customize as needed for your environment. Common modifications:
- Change base image (e.g., `python:3.11`)
- Add additional system packages
- Include additional Python packages
- Modify working directory

**Q: How do I debug inside the container?**
A: Use interactive mode:
```bash
docker-compose run -it agentic-self-rag bash
```

**Q: Can I use this with Kubernetes?**
A: Yes! Convert docker-compose to Kubernetes manifests using:
```bash
kompose convert -f docker-compose.yml -o k8s/
```

**Q: What's the image size?**
A: Typically 1.2-1.5 GB (depends on dependencies)

## Summary

✅ Docker is **already fully implemented**
✅ Three pre-configured services (main, ablation, smoke-test)
✅ Production-ready configuration
✅ Easy to deploy and scale
✅ Complete documentation provided

**To get started immediately:**
```bash
export GROQ_API_KEY=your_key_here
docker-compose up smoke-test  # Quick validation
docker-compose up agentic-self-rag  # Full evaluation
```

## Additional Resources

- Docker Documentation: https://docs.docker.com/
- Docker Compose Reference: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Security: https://docs.docker.com/engine/security/

---

**Last Updated**: 2025-11-26
**Status**: ✅ Production Ready
