# Generate Go Unit Test

Generate comprehensive unit tests for Go code using the standard testing package and testify.

## Test Structure

```go
package service

import (
    "context"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
)

// Mock repository
type MockUserRepository struct {
    mock.Mock
}

func (m *MockUserRepository) FindByID(ctx context.Context, id string) (*User, error) {
    args := m.Called(ctx, id)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

// Test function
func TestUserService_GetByID(t *testing.T) {
    // Table-driven tests
    tests := []struct {
        name    string
        userID  string
        setup   func(*MockUserRepository)
        want    *User
        wantErr bool
    }{
        {
            name:   "success",
            userID: "user-123",
            setup: func(m *MockUserRepository) {
                m.On("FindByID", mock.Anything, "user-123").
                    Return(&User{ID: "user-123", Email: "test@example.com"}, nil)
            },
            want:    &User{ID: "user-123", Email: "test@example.com"},
            wantErr: false,
        },
        {
            name:   "user not found",
            userID: "invalid-id",
            setup: func(m *MockUserRepository) {
                m.On("FindByID", mock.Anything, "invalid-id").
                    Return(nil, ErrNotFound)
            },
            want:    nil,
            wantErr: true,
        },
        {
            name:   "empty user ID",
            userID: "",
            setup:  func(m *MockUserRepository) {},
            want:    nil,
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Arrange
            mockRepo := new(MockUserRepository)
            if tt.setup != nil {
                tt.setup(mockRepo)
            }
            service := NewUserService(mockRepo, nil)

            // Act
            got, err := service.GetByID(context.Background(), tt.userID)

            // Assert
            if tt.wantErr {
                require.Error(t, err)
                assert.Nil(t, got)
            } else {
                require.NoError(t, err)
                assert.Equal(t, tt.want, got)
            }

            mockRepo.AssertExpectations(t)
        })
    }
}
```

## Test Coverage Requirements

1. **Happy Path**
   - Normal execution flow
   - Expected return values
   - Correct method calls

2. **Edge Cases**
   - Nil inputs
   - Empty slices/maps
   - Boundary values
   - Zero values

3. **Error Conditions**
   - Invalid inputs
   - Error scenarios
   - Error handling verification

4. **Integration Points**
   - Mock external dependencies
   - Verify service calls
   - Check database operations
   - Validate message queue operations

## Test Naming Convention

Format: `TestFunctionName_Condition_ExpectedResult` or use table-driven tests with descriptive names

Examples:
- `TestGetUserByID_ValidID_ReturnsUser`
- `TestCreateOrder_QueueFull_ReturnsError`
- `TestValidateEmail_InvalidFormat_ReturnsError`

Or with table-driven tests:
```go
tests := []struct {
    name string
    // ...
}{
    {name: "success with valid input"},
    {name: "error when user not found"},
    {name: "error with empty ID"},
}
```

## Mocking Guidelines

- Use testify/mock for interface mocking
- Mock all external dependencies (repositories, services, clients)
- Use `mock.Anything` for parameters you don't care about
- Verify interactions when important
- Clean up mocks between tests

## Assertions

- Use testify/assert for non-critical assertions
- Use testify/require for assertions that should stop the test immediately
- Check return values and state changes
- Verify method calls with `AssertExpectations()`

## Testing Best Practices

### 1. Table-Driven Tests
```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -2, -3, -5},
        {"zero", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            assert.Equal(t, tt.want, got)
        })
    }
}
```

### 2. Testing with Context
```go
func TestServiceWithTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()

    // Test with context
    result, err := service.DoWork(ctx)
    require.NoError(t, err)
}
```

### 3. Testing Goroutines
```go
func TestConcurrentOperation(t *testing.T) {
    var wg sync.WaitGroup
    results := make(chan int, 10)

    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(n int) {
            defer wg.Done()
            results <- process(n)
        }(i)
    }

    wg.Wait()
    close(results)

    // Verify results
    count := 0
    for range results {
        count++
    }
    assert.Equal(t, 10, count)
}
```

### 4. Testing with Race Detector
```bash
go test -race ./...
```

## Test Checklist

- [ ] All exported functions tested
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Mocks properly configured
- [ ] Assertions verify expected behavior
- [ ] Test names follow convention
- [ ] No hardcoded test data (use variables/constants)
- [ ] Tests are independent and isolated
- [ ] Race conditions checked with -race flag
- [ ] Context cancellation tested where applicable
