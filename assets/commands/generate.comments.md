# Generate Comments

Generate appropriate comments and JavaDoc for project code.

## Comment Types

### 1. JavaDoc Comments
For public classes, interfaces, and methods:

```java
/**
 * Brief description of the class/method.
 * 
 * More detailed description if needed.
 * 
 * @param paramName Description of parameter
 * @return Description of return value
 * @throws ExceptionType When this exception is thrown
 * @since Version (if applicable)
 */
```

### 2. Inline Comments
For complex logic or business decisions:

```java
// Why this approach was chosen, not what the code does
```

### 3. TODO Comments
For future improvements:

```java
// TODO(username): Description of what needs to be done
```

## Comment Guidelines

1. **JavaDoc Requirements**
   - All public classes and interfaces
   - All public methods
   - Include @param, @return, @throws
   - Explain purpose and behavior
   - Include usage examples for complex APIs

2. **Code Comments**
   - Explain "why" not "what"
   - Document business logic and decisions
   - Clarify non-obvious code
   - Avoid redundant comments

3. **What NOT to Comment**
   - Self-explanatory code
   - Obvious implementations
   - Code that should be refactored instead

4. **Project-Specific**
   - Document configuration dependencies
   - Explain integration logic (AWS, external services, etc.)
   - Clarify message queue handling (if applicable)
   - Document search/query logic (if applicable)
   - Explain security-sensitive operations

## Example

```java
/**
 * Processes an order and updates its status.
 * 
 * This method validates the order, processes payment, and updates inventory.
 * If the order is invalid or payment fails, an exception is thrown.
 * 
 * @param orderRequest The order request containing order details
 * @throws IllegalArgumentException if request is invalid
 * @throws PaymentException if payment processing fails
 */
public void processOrder(OrderRequest orderRequest) {
    // Validate order - early return if invalid
    if (!isValid(orderRequest)) {
        throw new IllegalArgumentException("Invalid order request");
    }
    
    // Implementation...
}
```

