# Elasticsearch RAG Demo with Security

## Overview
This configuration enables X-Pack Security for Elasticsearch and Kibana.

## Setup Details
- **Configuration**:
  - Credentials are stored in `.env` file.
  - **Elasticsearch**: 
    - Security enabled (`xpack.security.enabled=true`)
    - SSL/TLS disabled for HTTP API for simplicity (`xpack.security.http.ssl.enabled=false`)
  - **Kibana**:
    - Configured to connect to Elasticsearch using credentials from `.env`.

## Usage

1. **Configure Credentials**
   Edit the `.env` file if you want to change the default password:
   ```env
   ELASTIC_USERNAME=elastic
   ELASTIC_PASSWORD=changeme
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```

3. **Access Kibana**
   - URL: [http://localhost:5601](http://localhost:5601)
   - Username: `elastic` (or as defined in `.env`)
   - Password: `changeme` (or as defined in `.env`)

4. **Access Elasticsearch API**
   - URL: [http://localhost:9200](http://localhost:9200)
   - You must provide Basic Auth credentials.
   - Example with curl:
     ```bash
     # Load env vars if needed or just type them
     curl -u elastic:changeme http://localhost:9200
     ```

## Managing Passwords
To change the password for the `elastic` user:

1. Exec into the container:
   ```bash
   docker exec -it es_node /bin/bash
   ```
2. Run the password reset tool:
   ```bash
   ./bin/elasticsearch-reset-password -u elastic -i
   ```
   (Note: If you change the password inside the container, remember to update your `.env` file and restart Kibana so it can reconnect)
