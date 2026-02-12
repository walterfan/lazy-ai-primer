# Generate API Test

Generate API integration tests for REST endpoints.

## Test Structure

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureMockMvc
@Slf4j
public class ControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private IService service;
    
    @Test
    public void testEndpoint_whenValidRequest_thenReturnSuccess() throws Exception {
        // Arrange
        String requestBody = "{\"field\":\"value\"}";
        
        // Act & Assert
        mockMvc.perform(post("/api/endpoint")
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));
    }
}
```

## Test Coverage

1. **HTTP Methods**
   - GET requests
   - POST requests
   - PUT requests
   - DELETE requests

2. **Status Codes**
   - 200 OK
   - 201 Created
   - 400 Bad Request
   - 401 Unauthorized
   - 403 Forbidden
   - 404 Not Found
   - 500 Internal Server Error

3. **Request Validation**
   - Valid requests
   - Invalid requests
   - Missing required fields
   - Invalid data types
   - Boundary values

4. **Response Validation**
   - Response structure
   - Data correctness
   - Error messages
   - Response headers

## MockMvc Usage

```java
// GET request
mockMvc.perform(get("/api/users/{id}", userId)
        .header("Authorization", "Bearer " + token))
        .andExpect(status().isOk());

// POST request
mockMvc.perform(post("/api/users")
        .contentType(MediaType.APPLICATION_JSON)
        .content(requestJson))
        .andExpect(status().isCreated());

// PUT request
mockMvc.perform(put("/api/users/{id}", userId)
        .contentType(MediaType.APPLICATION_JSON)
        .content(updateJson))
        .andExpect(status().isOk());

// DELETE request
mockMvc.perform(delete("/api/users/{id}", userId))
        .andExpect(status().isOk());
```

## Response Assertions

```java
.andExpect(status().isOk())
.andExpect(content().contentType(MediaType.APPLICATION_JSON))
.andExpect(jsonPath("$.success").value(true))
.andExpect(jsonPath("$.data.id").value(expectedId))
.andExpect(jsonPath("$.data.name").exists())
.andExpect(jsonPath("$.errors").isEmpty())
```

## Project-Specific Considerations

- Test authentication/authorization
- Test input validation
- Test resource access permissions
- Test business rule validation
- Test error responses (custom exceptions)
- Test pagination
- Test filtering and sorting

## Test Checklist

- [ ] All endpoints tested
- [ ] Valid and invalid requests covered
- [ ] Authentication tested
- [ ] Authorization tested
- [ ] Error responses verified
- [ ] Response structure validated
- [ ] Edge cases covered
- [ ] Test data properly set up
- [ ] Cleanup after tests

## Guidelines
- Use @SpringBootTest for integration tests
- Mock external services (AWS, Elasticsearch)
- Use test profiles for configuration
- Clean up test data
- Test both success and failure scenarios

