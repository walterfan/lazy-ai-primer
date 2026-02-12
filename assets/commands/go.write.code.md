# Write Go Code

Generate Go code for functions, types, interfaces, or services in this project.

## Context Requirements

When writing Go code, ensure:

1. **Follow Project Conventions**
   - Use Go 1.x (check go.mod for exact version)
   - Follow standard Go project layout
   - Use appropriate frameworks (Gin, GORM, etc.)
   - Constructor functions with `New` prefix
   - Interfaces used for abstraction

2. **Project Structure**
   - Handlers: `handlers/` or `api/`
   - Services: `services/` or `service/`
   - Repositories: `repository/` or `repo/`
   - Models: `models/` or `domain/`
   - Utils: `pkg/` or `internal/`

3. **Code Quality**
   - Follow Go style guide (use gofmt, golint)
   - Minimal dependencies
   - Include godoc comments for exported items
   - Handle errors appropriately
   - Log appropriately
   - Never log sensitive data

4. **Security**
   - Never log sensitive data (passwords, tokens, PII)
   - Use crypto/rand for cryptographic randomness
   - Validate inputs early
   - Sanitize user inputs
   - Use parameterized queries

5. **Concurrency Patterns**
   - Use goroutines with proper cleanup
   - Pass context as first parameter
   - Implement cancellation with context
   - Use WaitGroups or channels to wait
   - Handle panics in goroutines

## Code Generation Checklist

- [ ] Proper package structure
- [ ] Exported items have godoc comments
- [ ] Context passed as first parameter
- [ ] Error handling implemented
- [ ] Logging with context
- [ ] Input validation
- [ ] No hardcoded values
- [ ] Configuration externalized
- [ ] Goroutine-safe (if shared state)

## Examples

**Service Interface:**
```go
// UserService defines operations for user management.
type UserService interface {
    // GetByID retrieves a user by their ID.
    GetByID(ctx context.Context, id string) (*User, error)

    // Create creates a new user.
    Create(ctx context.Context, user *User) error
}
```

**Service Implementation:**
```go
type userService struct {
    repo UserRepository
    log  *logrus.Logger
}

// NewUserService creates a new user service instance.
func NewUserService(repo UserRepository, log *logrus.Logger) UserService {
    return &userService{
        repo: repo,
        log:  log,
    }
}

func (s *userService) GetByID(ctx context.Context, id string) (*User, error) {
    if id == "" {
        return nil, fmt.Errorf("user ID is required")
    }

    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        s.log.WithError(err).WithField("user_id", id).Error("failed to get user")
        return nil, fmt.Errorf("failed to get user: %w", err)
    }

    return user, nil
}
```

**Handler:**
```go
type UserHandler struct {
    service UserService
}

func NewUserHandler(service UserService) *UserHandler {
    return &UserHandler{service: service}
}

// GetUser handles GET /users/:id
func (h *UserHandler) GetUser(c *gin.Context) {
    id := c.Param("id")

    user, err := h.service.GetByID(c.Request.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
            return
        }
        c.JSON(http.StatusInternalServerError, gin.H{"error": "internal server error"})
        return
    }

    c.JSON(http.StatusOK, user)
}
```
