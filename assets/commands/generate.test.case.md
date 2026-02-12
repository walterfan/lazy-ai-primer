# Generate BDD Test Case

Generate BDD (Behavior-Driven Development) style test cases in Gherkin format based on user requirements.

## BDD Format Structure

Generate test cases using the following Gherkin syntax:

```
Feature: [Feature Name]

  Scenario: [Scenario Description]
    Given [initial context]
    When [event/action]
    Then [expected outcome]
```

## Format Guidelines

1. **Feature**: High-level description of the feature being tested
   - Should be a noun phrase describing the capability
   - Examples: "User Login", "Order Processing", "Password Reset"

2. **Scenario**: A specific test case within the feature
   - Should describe a concrete example of behavior
   - Examples: "Successful login", "Login with invalid credentials", "Password reset request"

3. **Given**: Preconditions or initial state
   - Sets up the context before the action
   - Use past tense or present perfect
   - Examples: "the user is registered", "the user enters correct credentials", "the order exists"

4. **When**: The action or event that triggers the behavior
   - Describes what the user or system does
   - Use present tense
   - Examples: "they click 'Login'", "the system processes the payment", "the user submits the form"

5. **Then**: The expected outcome or result
   - Describes what should happen
   - Use present tense or future tense
   - Examples: "they should see the dashboard", "the order status should be 'confirmed'", "an error message should appear"

## Multiple Scenarios

When generating multiple scenarios for the same feature, use this format:

```
Feature: [Feature Name]

  Scenario: [First Scenario]
    Given [context]
    When [action]
    Then [outcome]

  Scenario: [Second Scenario]
    Given [context]
    When [action]
    Then [outcome]
```

## Advanced Scenarios

For more complex scenarios, you can use:

- **And**: To add additional steps of the same type
  ```
  Given the user is registered
    And the user has valid credentials
  ```

- **But**: To add a contrasting step
  ```
  Given the user is registered
    But the account is locked
  ```

- **Background**: For common setup steps across scenarios
  ```
  Feature: User Login

    Background:
      Given the application is running
        And the database is initialized

    Scenario: Successful login
      Given the user enters correct credentials
      When they click "Login"
      Then they should see the dashboard
  ```

## Scenario Outline (Data-Driven Tests)

For testing with multiple data sets:

```
Feature: [Feature Name]

  Scenario Outline: [Scenario Description]
    Given [context with <parameter>]
    When [action]
    Then [outcome]

    Examples:
      | parameter | expected |
      | value1    | result1  |
      | value2    | result2  |
```

## Best Practices

1. **Clarity**: Use clear, business-friendly language
2. **Independence**: Each scenario should be independent and testable in isolation
3. **Completeness**: Cover happy paths, edge cases, and error scenarios
4. **Specificity**: Be specific about actions and expected outcomes
5. **User Perspective**: Write from the user's or stakeholder's perspective

## Common Test Scenarios to Consider

1. **Happy Path**: Normal successful flow
2. **Error Cases**: Invalid inputs, missing data, system errors
3. **Edge Cases**: Boundary values, empty inputs, maximum values
4. **Security**: Authentication, authorization, data validation
5. **Integration**: External service interactions, database operations

## Example Output

```
Feature: User Login

  Scenario: Successful login
    Given the user enters correct credentials
    When they click "Login"
    Then they should see the dashboard

  Scenario: Login with invalid credentials
    Given the user enters incorrect credentials
    When they click "Login"
    Then an error message should be displayed
      And they should remain on the login page

  Scenario: Login with empty credentials
    Given the user leaves credentials empty
    When they click "Login"
    Then validation errors should be shown
      And the login should not proceed
```

## Guidelines

- Analyze the user's input to understand the feature and scenarios they want to test
- Generate comprehensive test cases covering multiple scenarios
- Use clear, descriptive language that non-technical stakeholders can understand
- Follow the Gherkin syntax strictly
- Include both positive and negative test cases when appropriate
- Consider edge cases and error conditions
- Maintain consistent formatting and indentation (2 spaces)

