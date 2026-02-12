# Go Code Review

Perform a thorough code review following Go best practices and project standards.

## Review Checklist

### Code Quality
- [ ] Follows Go style guide (gofmt, golint)
- [ ] Proper naming conventions (PascalCase for exported, camelCase for unexported)
- [ ] No code duplication (DRY principle)
- [ ] Single Responsibility Principle followed
- [ ] Appropriate use of design patterns
- [ ] Exported types/functions have documentation comments

### Go Best Practices
- [ ] Proper error handling (no ignored errors)
- [ ] Context passed as first parameter where appropriate
- [ ] Interfaces used appropriately (small, focused interfaces)
- [ ] No goroutine leaks (proper cleanup and cancellation)
- [ ] Proper use of channels (sender closes, buffering appropriate)
- [ ] Mutex locks always deferred immediately after locking

### Security
- [ ] No sensitive data in logs
- [ ] No credentials in error messages
- [ ] Input validation implemented
- [ ] SQL injection prevention (parameterized queries)
- [ ] Proper authentication/authorization checks
- [ ] Uses crypto/rand for cryptographic random numbers (not math/rand)

### Error Handling
- [ ] All errors are checked and handled
- [ ] Errors wrapped with context using fmt.Errorf with %w
- [ ] Custom error types used where appropriate
- [ ] Proper error logging before returning

### Database & Persistence
- [ ] GORM parameterized queries (? placeholders)
- [ ] Proper struct tags (gorm, json, validate)
- [ ] Transaction boundaries appropriate
- [ ] N+1 query problems avoided (Preload/Joins used)
- [ ] Connection pool configured appropriately

### Concurrency
- [ ] Goroutines have proper lifecycle management
- [ ] WaitGroups or channels used to wait for goroutines
- [ ] Context cancellation implemented for long-running operations
- [ ] No data races (verified with -race flag)
- [ ] Proper use of mutexes (defer unlock immediately)
- [ ] Loop variables passed to goroutines as parameters

### Testing
- [ ] Unit tests provided (*_test.go files)
- [ ] Table-driven tests used where appropriate
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Test names follow convention (TestFunctionName)
- [ ] No race conditions in tests

### Documentation
- [ ] Godoc comments for exported types and functions
- [ ] Complex logic documented
- [ ] Configuration documented
- [ ] README updated if needed

### Performance
- [ ] Appropriate use of pointers vs values
- [ ] String concatenation uses strings.Builder for loops
- [ ] Database queries optimized
- [ ] Proper use of buffered channels
- [ ] Worker pools for bounded concurrency

### API Design (if applicable)
- [ ] REST conventions followed
- [ ] Proper HTTP status codes
- [ ] Request validation at handler level
- [ ] Swagger/OpenAPI annotations present
- [ ] Context propagated through layers

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
