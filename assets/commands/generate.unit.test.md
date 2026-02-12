# Generate Unit Test

Generate comprehensive unit tests for project code using JUnit 4 and Mockito.

## Test Structure

```java
@RunWith(MockitoJUnitRunner.class)
@Slf4j
public class ServiceImplTest {
    
    @Mock
    private DependencyDao dao;
    
    @Mock
    private ISearchService searchService;
    
    @InjectMocks
    private ServiceImpl service;
    
    @Before
    public void setUp() {
        // Setup test data
    }
    
    @Test
    public void testMethodName_whenCondition_thenExpectedResult() {
        // Arrange
        // Act
        // Assert
    }
    
    @Test(expected = ExceptionType.class)
    public void testMethodName_whenInvalidInput_thenThrowException() {
        // Test error conditions
    }
}
```

## Test Coverage Requirements

1. **Happy Path**
   - Normal execution flow
   - Expected return values
   - Correct method calls

2. **Edge Cases**
   - Null inputs
   - Empty collections
   - Boundary values
   - Maximum/minimum values

3. **Error Conditions**
   - Invalid inputs
   - Exception scenarios
   - Error handling verification

4. **Integration Points**
   - Mock external dependencies
   - Verify service calls
   - Check database operations
   - Validate AsyncMQ messages

## Test Naming Convention

Format: `testMethodName_whenCondition_thenExpectedResult`

Examples:
- `testGetUserById_whenValidId_thenReturnUser`
- `testSaveOrder_whenQueueFull_thenDropRecord`
- `testValidateEmail_whenInvalidFormat_thenThrowException`

## Mocking Guidelines

- Mock all external dependencies (DAOs, Services, AWS clients)
- Use `@Mock` for dependencies
- Use `@InjectMocks` for class under test
- Verify interactions when important
- Use `ArgumentMatchers` for flexible matching

## Assertions

- Use JUnit assertions (`assertEquals`, `assertTrue`, `assertNotNull`, etc.)
- Verify exception messages when testing error cases
- Check return values and state changes
- Verify method calls with Mockito `verify()`

## Project-Specific Considerations

- Test message queue publishing (if applicable)
- Test search/query logic (if applicable)
- Test external service integration error handling
- Test configuration property handling
- Test security-sensitive operations (without exposing sensitive data)
- Test transaction boundaries
- Test cache behavior

## Test Checklist

- [ ] All public methods tested
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Mocks properly configured
- [ ] Assertions verify expected behavior
- [ ] Test names follow convention
- [ ] No hardcoded test data (use constants)
- [ ] Tests are independent and isolated

