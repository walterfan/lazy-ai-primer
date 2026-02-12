# Write Code

Generate code for functions, classes, interfaces, or components in this project.

## Context Requirements

When writing code, ensure:

1. **Follow Project Conventions**
   - Use Java 16 (not Java 17+ features)
   - Spring Boot 3.4.11 patterns
   - MyBatis for database operations
   - Constructor injection (not field injection)
   - Services implement interfaces (I*Service pattern)

2. **Project Structure**
   - Controllers: `com.example.project.controller.*`
   - Services: `com.example.project.service.*` (interfaces) and `service.impl.*` (implementations)
   - DAOs: `com.example.project.dao.*`
   - Domain: `com.example.project.domain.*`
   - Framework utilities: `com.example.project.framework.*`

3. **Code Quality**
   - Follow Google Java Style Guide
   - Use Lombok for boilerplate reduction
   - Include JavaDoc for public methods
   - Handle exceptions appropriately
   - Log appropriately (use @Slf4j)
   - Never log sensitive data

4. **Security**
   - Never log sensitive data (passwords, tokens, PII)
   - Use appropriate crypto utility for hashing
   - Validate inputs early
   - Sanitize user inputs

5. **Integration Patterns**
   - Message Queue: Use appropriate service for publishing
   - Elasticsearch: Extend AbstractSearchRepository (if applicable)
   - AWS: Use IAM roles, implement retry logic (if applicable)
   - Scheduled jobs: Implement appropriate job interface

## Code Generation Checklist

- [ ] Proper package structure
- [ ] Constructor injection for dependencies
- [ ] Interface implementation (if service)
- [ ] Exception handling
- [ ] Logging with context
- [ ] Input validation
- [ ] JavaDoc for public methods
- [ ] No hardcoded values
- [ ] Configuration externalized
- [ ] Thread-safe (if shared state)

## Examples

**Service Interface:**
```java
public interface IFeatureService {
    /**
     * Description of method
     * @param param Description
     * @return Description
     */
    ReturnType methodName(ParamType param);
}
```

**Service Implementation:**
```java
@Service
@Slf4j
public class FeatureServiceImpl implements IFeatureService {
    private final FeatureDao dao;
    private final ISearchService searchService;
    
    public FeatureServiceImpl(FeatureDao dao, ISearchService searchService) {
        this.dao = dao;
        this.searchService = searchService;
    }
    
    @Override
    public ReturnType methodName(ParamType param) {
        // Implementation
    }
}
```

