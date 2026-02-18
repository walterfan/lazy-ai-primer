# Elasticsearch RAG Demo with Security

## Overview
This configuration enables X-Pack Security for Elasticsearch and Kibana.

## Setup Details
- **Configuration**:
  - Credentials are stored in `.env` file.
  - **Elasticsearch**: 
    - Security enabled (`xpack.security.enabled=true`)
    - SSL/TLS disabled for HTTP API for simplicity (`xpack.security.http.ssl.enabled=false`)
    - Superuser: `elastic` with password from `ELASTIC_PASSWORD`
  - **Kibana**:
    - Uses `kibana_system` built-in user (required by Elasticsearch 8.x)
    - Password is set via `KIBANA_PASSWORD` in `.env`

## Why kibana_system user?
Starting from Elasticsearch 8.x, Kibana cannot use the `elastic` superuser account directly. 
The `elastic` superuser is forbidden because it has too many privileges and cannot write to 
certain system indices that Kibana needs. Instead, Kibana must use:
- The built-in `kibana_system` user (used in this setup), OR
- A service account token

## Usage

1. **Configure Credentials**
   Copy `env.example` to `.env` and edit if you want to change the default passwords:
   ```bash
   cp env.example .env
   ```
   
   Content of `.env`:
   ```env
   ELASTIC_PASSWORD=changeme
   KIBANA_PASSWORD=kibana_changeme
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```
   
   The `es_setup` container will automatically set the `kibana_system` user password.

3. **Access Kibana**
   - URL: [http://localhost:5601](http://localhost:5601)
   - Username: `elastic`
   - Password: value of `ELASTIC_PASSWORD` in your `.env` file (default: `changeme`)

4. **Access Elasticsearch API**
   - URL: [http://localhost:9200](http://localhost:9200)
   - You must provide Basic Auth credentials.
   - Example with curl:
     ```bash
     curl -u elastic:changeme http://localhost:9200
     ```

## Managing Passwords
To change the password for the `elastic` user after initial setup:

1. Exec into the container:
   ```bash
   docker exec -it es_node /bin/bash
   ```
2. Run the password reset tool:
   ```bash
   ./bin/elasticsearch-reset-password -u elastic -i
   ```
   (Note: If you change the password, remember to update your `.env` file)

## Stopping and Cleaning Up
```bash
# Stop services
docker-compose down

# Stop and remove volumes (will delete all data!)
docker-compose down -v
```
