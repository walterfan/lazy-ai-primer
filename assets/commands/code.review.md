# Code Review

Perform a thorough code review following project standards.

## Review Checklist

### Code Quality
- [ ] Follows Google Java Style Guide
- [ ] Proper naming conventions (PascalCase, camelCase, UPPER_SNAKE_CASE)
- [ ] No code duplication (DRY principle)
- [ ] Single Responsibility Principle followed
- [ ] Appropriate use of design patterns

### Spring Boot Best Practices
- [ ] Constructor injection used (not field injection)
- [ ] Services implement interfaces (I*Service pattern)
- [ ] Proper use of @Transactional
- [ ] Configuration externalized (@ConfigurationProperties or @Value)
- [ ] Appropriate use of Spring annotations

### Security
- [ ] No sensitive data in logs
- [ ] No credentials in error messages
- [ ] Input validation implemented
- [ ] SQL injection prevention (parameterized queries)
- [ ] Proper authentication/authorization checks

### Error Handling
- [ ] Custom exceptions used appropriately
- [ ] Exceptions include context
- [ ] Proper exception logging
- [ ] Meaningful error messages for clients

### Database & Persistence
- [ ] MyBatis XML mappers used (not annotations)
- [ ] Parameterized queries (#{param} not ${param})
- [ ] Transaction boundaries appropriate
- [ ] N+1 query problems avoided

### Testing
- [ ] Unit tests provided
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Test naming follows convention (Test suffix)

### Documentation
- [ ] JavaDoc for public methods
- [ ] Complex logic documented
- [ ] Configuration documented
- [ ] README updated if needed

### Performance
- [ ] Appropriate caching strategy
- [ ] Database queries optimized
- [ ] No unnecessary object creation
- [ ] Async processing where appropriate

### Integration
- [ ] AsyncMQ patterns followed correctly
- [ ] Elasticsearch queries efficient
- [ ] AWS integration handles errors gracefully
- [ ] Retry logic implemented where needed

## Review Comments Format

**Positive:**
- ✅ Good: [What's good about it]

**Issues:**
- ❌ Issue: [Problem description]
  - Suggestion: [How to fix]

**Questions:**
- ❓ Question: [Clarification needed]

## Guidelines
- Be constructive and respectful
- Focus on code, not the author
- Provide specific examples
- Suggest improvements, not just problems
- Consider project context and constraints

