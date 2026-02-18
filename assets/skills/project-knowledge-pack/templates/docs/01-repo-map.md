# Repository Map

> **Note**: This document helps developers and AI quickly locate code and understand the repository structure.

## Directory Structure

```
[Run: tree -L 3 -d or equivalent]

Example:
.
├── cmd/                  # Application entry points
├── config/              # Configuration files
├── docs/                # Documentation
├── internal/            # Private application code
│   ├── api/            # API handlers
│   ├── domain/         # Business logic
│   └── repository/     # Data access
├── pkg/                 # Public libraries
├── scripts/             # Utility scripts
└── test/                # Test files
```

## Directory Responsibilities

### `/cmd` - Application Entry Points

- `cmd/server/main.go`: Main server application
- `cmd/cli/main.go`: Command-line tool
- Purpose: [Describe]

### `/internal` - Private Application Code

- `internal/api/`: HTTP/gRPC handlers, request/response models
- `internal/domain/`: Business logic, domain entities, use cases
- `internal/repository/`: Database access, external service clients
- Purpose: [Describe]

### `/pkg` - Public Libraries

- `pkg/logger/`: Logging utilities
- `pkg/middleware/`: Reusable middleware
- Purpose: [Describe]

### `/config` - Configuration Files

- `config/dev.yaml`: Development environment config
- `config/prod.yaml`: Production environment config
- Purpose: [Describe]

### `/test` - Test Files

- `test/unit/`: Unit tests
- `test/integration/`: Integration tests
- `test/e2e/`: End-to-end tests
- Purpose: [Describe]

### `/scripts` - Utility Scripts

- `scripts/setup.sh`: Environment setup
- `scripts/migrate.sh`: Database migration
- Purpose: [Describe]

### `/docs` or `/man` - Documentation

- `docs/architecture/`: Architecture diagrams and decisions
- `docs/api/`: API specifications
- Purpose: [Describe]

## Key Entry Points

### Application Startup

- **Main entry**: `cmd/server/main.go`
  - Initializes: Configuration, database, services
  - Starts: HTTP server, background workers

### Request Handling

- **Router**: `internal/api/router.go`
  - Defines: All HTTP routes
  - Middleware: Auth, logging, rate limiting

- **Handler**: `internal/api/handler/`
  - Pattern: One file per resource (users, orders, etc.)

### Business Logic

- **Use Cases**: `internal/domain/usecase/`
  - Pattern: One file per major use case
  - Entry: `ProcessOrder()`, `CreateUser()`, etc.

### Data Access

- **Repository**: `internal/repository/`
  - Pattern: One file per entity
  - Entry: `UserRepository.Create()`, etc.

### Configuration

- **Config Loader**: `internal/config/config.go`
  - Loads from: Environment variables, YAML files
  - Validates: Required fields, formats

### Dependency Injection

- **Container**: `internal/di/container.go`
  - Wires: All dependencies
  - Pattern: [Factory / Constructor injection]

## Conventions

### Naming

#### Files

- Go: `snake_case.go` or `camelCase.go`
- Test files: `*_test.go`
- Interface files: `i_interface.go` or `interface.go`

#### Packages

- Short, lowercase, single word when possible
- Example: `api`, `domain`, `repository`

#### Functions

- Exported: `PascalCase` (public)
- Unexported: `camelCase` (private)

#### Variables

- Exported: `PascalCase` (public)
- Unexported: `camelCase` (private)
- Constants: `UPPER_SNAKE_CASE` or `PascalCase`

### Layering

This project follows [Clean Architecture / Hexagonal / Layered] architecture:

#### Presentation Layer

- Location: `internal/api/`
- Responsibility: HTTP/gRPC handlers, request validation, response formatting
- Dependencies: Can depend on domain layer

#### Domain/Business Layer

- Location: `internal/domain/`
- Responsibility: Business logic, domain entities, use cases
- Dependencies: Should not depend on infrastructure

#### Data/Infrastructure Layer

- Location: `internal/repository/`, `internal/client/`
- Responsibility: Database access, external service clients
- Dependencies: Implements interfaces defined in domain

### Common Patterns

#### CQRS (Command Query Responsibility Segregation)

- Commands: `internal/domain/command/`
- Queries: `internal/domain/query/`
- Pattern: Separate read and write operations

#### DDD (Domain-Driven Design)

- Entities: `internal/domain/entity/`
- Value Objects: `internal/domain/vo/`
- Aggregates: `internal/domain/aggregate/`
- Domain Events: `internal/domain/event/`

#### Repository Pattern

```go
// internal/domain/repository/user_repository.go
type UserRepository interface {
    Create(ctx context.Context, user *User) error
    FindByID(ctx context.Context, id string) (*User, error)
    Update(ctx context.Context, user *User) error
    Delete(ctx context.Context, id string) error
}

// internal/repository/user_repository_impl.go
type userRepositoryImpl struct {
    db *sql.DB
}
```

#### Factory Pattern

- Used for: Creating complex objects
- Location: `internal/domain/factory/`

#### Strategy Pattern

- Used for: Pluggable algorithms
- Location: `internal/domain/strategy/`

## Import Organization

### Order

1. Standard library
2. External dependencies
3. Internal packages

### Example

```go
import (
    // Standard library
    "context"
    "fmt"

    // External dependencies
    "github.com/gin-gonic/gin"
    "gorm.io/gorm"

    // Internal packages
    "github.com/yourorg/project/internal/domain"
    "github.com/yourorg/project/pkg/logger"
)
```

## Code Organization Principles

1. **Package by Feature**: Group related code together
2. **Dependency Direction**: Dependencies point inward (toward domain)
3. **Interface Segregation**: Small, focused interfaces
4. **Single Responsibility**: Each package has one clear purpose

## Related Documentation

- [Architecture](02-architecture.md) - Detailed architecture description
- [Conventions](05-conventions.md) - Coding conventions and standards
- [ADRs](adr/) - Architecture decision records

---

**Last Updated**: [Date]
**Authors**: [Names]
**Version**: 1.0
