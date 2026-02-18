# Project Overview

> **Note**: This is a template. Fill in each section with actual project information.

## Purpose

**What problem does this project solve?**

[Describe the core problem or need this project addresses]

## Business Boundaries

### What We Do

- Core functionality 1
- Core functionality 2
- Core functionality 3

### What We Don't Do

> Important: Clearly state what is out of scope to avoid scope creep and misaligned expectations.

- Out of scope item 1
- Out of scope item 2
- Out of scope item 3

## Key User Roles

- **Role 1**: [Description and primary needs]
- **Role 2**: [Description and primary needs]
- **Role 3**: [Description and primary needs]

## Core Use Cases

1. **Use Case 1**: [Description]
   - Actor: [Who]
   - Goal: [What they want to achieve]
   - Outcome: [Expected result]

2. **Use Case 2**: [Description]
   - Actor: [Who]
   - Goal: [What they want to achieve]
   - Outcome: [Expected result]

3. **Use Case 3**: [Description]
   - Actor: [Who]
   - Goal: [What they want to achieve]
   - Outcome: [Expected result]

## Technology Stack

### Language & Runtime

- Primary language: [e.g., Go 1.21]
- Secondary languages: [e.g., Python for scripts]

### Framework

- Web framework: [e.g., Gin, Spring Boot, Express]
- Testing framework: [e.g., pytest, JUnit, Jest]

### Database

- Primary database: [e.g., PostgreSQL 15]
- Cache: [e.g., Redis 7]
- Search: [e.g., Elasticsearch 8]

### Middleware & Infrastructure

- Message queue: [e.g., RabbitMQ, Kafka]
- Service mesh: [e.g., Istio]
- Container orchestration: [e.g., Kubernetes]

### External Services

- Authentication: [e.g., Auth0, Okta]
- Payment: [e.g., Stripe]
- Email: [e.g., SendGrid]

## Deployment Model

[Choose one or describe your specific model]

- **Monolith**: Single deployable application
- **Microservices**: Multiple independent services
- **Multi-platform**: Web, mobile, desktop clients

### Deployment Topology

```
[Describe or draw deployment architecture]
- How many environments? (dev, staging, prod)
- How are services distributed?
- What's the data residency requirement?
```

## Quality Targets (SLO)

### Performance

- API response time: p95 < [X]ms, p99 < [Y]ms
- Throughput: [Z] requests/second
- Database query time: p95 < [X]ms

### Availability

- Uptime target: [e.g., 99.9%]
- Recovery Time Objective (RTO): [e.g., < 1 hour]
- Recovery Point Objective (RPO): [e.g., < 15 minutes]

### Consistency

- Consistency model: [e.g., eventual consistency, strong consistency]
- Data synchronization lag: [e.g., < 5 seconds]

### Scalability

- Expected user count: [e.g., 10K concurrent users]
- Data volume: [e.g., 1TB, growing 10GB/month]
- Peak traffic: [e.g., 10x normal during holidays]

## Compliance & Security

- Compliance requirements: [e.g., GDPR, SOC 2, HIPAA]
- Authentication method: [e.g., OAuth2 + JWT]
- Data encryption: [e.g., at rest and in transit]
- Audit logging: [e.g., all API calls logged with retention policy]

## Team & Contacts

- Product Owner: [Name/Email]
- Tech Lead: [Name/Email]
- On-call rotation: [Link to schedule]
- Slack channel: [Channel name]
- Issue tracker: [Link]

## Related Documentation

- Architecture diagrams: [Link]
- API documentation: [Link]
- Runbook: [Link to 06-runbook.md]
- ADRs: [Link to adr/ directory]

---

**Last Updated**: [Date]
**Authors**: [Names]
**Version**: 1.0
