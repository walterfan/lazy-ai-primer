# Business Workflows

> **Note**: This document describes key business workflows with detailed process flows, data flows, and code references.

## Overview

This document contains detailed descriptions of the main business workflows in the system. Each workflow includes:
- Process steps
- Sequence diagrams
- State transitions
- Input/output specifications
- Error handling
- Code references

## Workflow Index

1. [User Registration](#workflow-1-user-registration)
2. [User Login](#workflow-2-user-login)
3. [Order Processing](#workflow-3-order-processing)
4. [Payment Processing](#workflow-4-payment-processing)

---

## Workflow 1: User Registration

### Overview

New users register by providing email, password, and basic profile information.

### Sequence Diagram

```{mermaid}
sequenceDiagram
    actor User
    participant UI
    participant API
    participant Validator
    participant UserService
    participant EmailService
    participant DB

    User->>UI: Enter Registration Details
    UI->>API: POST /api/v1/users/register

    API->>Validator: Validate Input

    alt Validation Failed
        Validator-->>API: Validation Errors
        API-->>UI: 400 Bad Request
        UI-->>User: Show Errors
    else Validation Passed
        Validator-->>API: Valid

        API->>UserService: CreateUser(details)
        UserService->>DB: Check Email Exists

        alt Email Exists
            DB-->>UserService: Email Found
            UserService-->>API: Email Already Registered
            API-->>UI: 409 Conflict
            UI-->>User: Email Already Registered
        else Email Available
            DB-->>UserService: Email Available
            UserService->>UserService: Hash Password
            UserService->>DB: Insert User
            DB-->>UserService: User Created

            UserService->>EmailService: Send Verification Email
            EmailService-->>UserService: Email Queued

            UserService-->>API: User Created
            API-->>UI: 201 Created
            UI-->>User: Check Your Email
        end
    end
```

### State Diagram

```{mermaid}
stateDiagram-v2
    [*] --> Draft: User Starts Registration
    Draft --> Validating: Submit Form

    Validating --> Invalid: Validation Failed
    Invalid --> Draft: Fix Errors

    Validating --> Creating: Validation Passed

    Creating --> Failed: Email Exists
    Failed --> Draft: Try Different Email

    Creating --> Unverified: Account Created
    Unverified --> Active: Email Verified
    Unverified --> Expired: 24h Timeout

    Expired --> [*]: Auto-Deleted
    Active --> [*]: Registration Complete
```

### Process Steps

1. **User Input** (`src/ui/pages/RegisterPage.tsx`)
   - User enters: email, password, confirm password, name
   - Client-side validation: format, password strength

2. **API Request** (`src/api/controllers/user_controller.go:CreateUser`)
   - Endpoint: `POST /api/v1/users/register`
   - Request body:
     ```json
     {
       "email": "user@example.com",
       "password": "SecurePass123!",
       "name": "John Doe"
     }
     ```

3. **Validation** (`src/validators/user_validator.go:ValidateRegistration`)
   - Email format validation
   - Password strength check (min 8 chars, special chars, numbers)
   - Name length validation

4. **Business Logic** (`src/services/user_service.go:CreateUser`)
   - Check if email already exists
   - Hash password using bcrypt
   - Generate verification token
   - Create user record

5. **Database** (`src/repositories/user_repository.go:Create`)
   - Table: `users`
   - Transaction: Begin → Insert → Commit

6. **Email Notification** (`src/services/email_service.go:SendVerificationEmail`)
   - Template: `verification_email.html`
   - Queue: `email_queue`
   - Async processing

### Input/Output

**Input**:
- Email (string, required, valid email format)
- Password (string, required, min 8 chars)
- Name (string, required, max 100 chars)

**Output Success (201 Created)**:
```json
{
  "id": "uuid-here",
  "email": "user@example.com",
  "name": "John Doe",
  "status": "unverified",
  "created_at": "2026-02-18T10:00:00Z"
}
```

**Output Error (400 Bad Request)**:
```json
{
  "error": "validation_failed",
  "details": [
    {
      "field": "password",
      "message": "Password must contain at least one special character"
    }
  ]
}
```

### Error Scenarios

| Error | Cause | HTTP Code | Response | Action |
|-------|-------|-----------|----------|--------|
| Invalid Email | Email format invalid | 400 | validation_failed | Fix email format |
| Weak Password | Password too simple | 400 | validation_failed | Strengthen password |
| Email Exists | Email already registered | 409 | email_exists | Use different email or login |
| Database Error | DB connection failed | 500 | internal_error | Retry or contact support |

### Data Entities

**Tables**:
- `users`: User account data
  - Columns: id, email, password_hash, name, status, created_at, updated_at

**Events**:
- `user.registered`: Published when user is created
  - Payload: user_id, email, timestamp

**Cache**:
- Key: `user:email:{email}` → user_id
- TTL: 15 minutes

### Code References

- Controller: `src/api/controllers/user_controller.go:CreateUser` (line 45)
- Service: `src/services/user_service.go:CreateUser` (line 120)
- Repository: `src/repositories/user_repository.go:Create` (line 78)
- Validator: `src/validators/user_validator.go:ValidateRegistration` (line 23)
- Email: `src/services/email_service.go:SendVerificationEmail` (line 156)

---

## Workflow 2: User Login

### Overview

Registered users authenticate with email and password to receive an access token.

### Sequence Diagram

```{mermaid}
sequenceDiagram
    actor User
    participant UI
    participant API
    participant AuthService
    participant DB
    participant Cache

    User->>UI: Enter Credentials
    UI->>API: POST /api/v1/auth/login

    API->>AuthService: Authenticate(email, password)
    AuthService->>DB: FindUserByEmail(email)

    alt User Not Found
        DB-->>AuthService: Not Found
        AuthService-->>API: Invalid Credentials
        API-->>UI: 401 Unauthorized
        UI-->>User: Invalid Email or Password
    else User Found
        DB-->>AuthService: User Data
        AuthService->>AuthService: Verify Password

        alt Password Incorrect
            AuthService-->>API: Invalid Credentials
            API-->>UI: 401 Unauthorized
            UI-->>User: Invalid Email or Password
        else Password Correct
            AuthService->>AuthService: Generate JWT Token
            AuthService->>Cache: Store Session
            AuthService-->>API: Token + User Info
            API-->>UI: 200 OK + Token
            UI->>UI: Store Token
            UI-->>User: Login Success → Dashboard
        end
    end
```

### Authentication Flow

```{mermaid}
flowchart TD
    Start([Login Request]) --> Validate{Validate<br/>Credentials}

    Validate -->|Invalid Format| Error1[400 Bad Request]
    Validate -->|Valid Format| FindUser[Find User in DB]

    FindUser -->|Not Found| Error2[401 Unauthorized]
    FindUser -->|Found| VerifyPassword{Verify<br/>Password}

    VerifyPassword -->|Incorrect| Error2
    VerifyPassword -->|Correct| CheckStatus{Account<br/>Status}

    CheckStatus -->|Inactive| Error3[403 Account Inactive]
    CheckStatus -->|Locked| Error4[403 Account Locked]
    CheckStatus -->|Active| GenerateToken[Generate JWT]

    GenerateToken --> StoreSession[Store Session in Cache]
    StoreSession --> LogEvent[Log Login Event]
    LogEvent --> Success[200 OK + Token]

    Error1 --> End([End])
    Error2 --> End
    Error3 --> End
    Error4 --> End
    Success --> End
```

### Token Structure

```{mermaid}
graph LR
    Token[JWT Token] --> Header[Header]
    Token --> Payload[Payload]
    Token --> Signature[Signature]

    Header --> Algo["alg: HS256"]
    Header --> Type["typ: JWT"]

    Payload --> UserID["user_id"]
    Payload --> Email["email"]
    Payload --> Roles["roles"]
    Payload --> Exp["exp: 1h"]

    Signature --> Secret["HMAC(header + payload, secret)"]
```

### Session Management

```{mermaid}
stateDiagram-v2
    [*] --> Active: Login Success

    Active --> Refreshed: Refresh Token
    Active --> Expired: Token Timeout
    Active --> Revoked: User Logout

    Refreshed --> Active: New Token Issued

    Expired --> [*]: Auto Cleanup
    Revoked --> [*]: Manual Logout
```

### Code References

- Controller: `src/api/controllers/auth_controller.go:Login` (line 34)
- Service: `src/services/auth_service.go:Authenticate` (line 89)
- JWT: `src/pkg/jwt/jwt.go:GenerateToken` (line 45)
- Cache: `src/pkg/cache/session.go:StoreSession` (line 67)

---

## Workflow 3: Order Processing

### Overview

Complete order lifecycle from creation to fulfillment.

### State Machine

```{mermaid}
stateDiagram-v2
    [*] --> Draft: Create Order
    Draft --> Pending: Submit Order

    Pending --> Confirmed: Payment Received
    Pending --> Cancelled: Payment Failed/Timeout

    Confirmed --> Processing: Start Fulfillment
    Processing --> Shipped: Shipment Created
    Shipped --> Delivered: Delivery Confirmed

    Confirmed --> Cancelled: Out of Stock
    Processing --> Cancelled: Processing Failed

    Delivered --> Completed: Auto-Complete (7 days)

    Completed --> [*]
    Cancelled --> [*]

    note right of Pending
        Timeout: 30 minutes
        Then auto-cancel
    end note

    note right of Shipped
        Track via shipment_id
        Push notifications
    end note
```

### Order Processing Flow

```{mermaid}
sequenceDiagram
    participant Customer
    participant OrderService
    participant InventoryService
    participant PaymentService
    participant ShippingService
    participant NotificationService

    Customer->>OrderService: Create Order
    OrderService->>InventoryService: Reserve Items

    alt Items Available
        InventoryService-->>OrderService: Reserved
        OrderService->>PaymentService: Process Payment

        alt Payment Success
            PaymentService-->>OrderService: Payment Confirmed
            OrderService->>OrderService: Confirm Order
            OrderService->>InventoryService: Commit Reservation
            OrderService->>ShippingService: Create Shipment
            ShippingService-->>OrderService: Shipment Created
            OrderService->>NotificationService: Send Confirmation
            NotificationService-->>Customer: Order Confirmed Email
        else Payment Failed
            PaymentService-->>OrderService: Payment Failed
            OrderService->>InventoryService: Release Reservation
            OrderService-->>Customer: Payment Failed
        end
    else Items Unavailable
        InventoryService-->>OrderService: Out of Stock
        OrderService-->>Customer: Items Unavailable
    end
```

### Distributed Transaction (Saga)

```{mermaid}
graph TB
    subgraph "Saga Orchestration"
        Start[Order Created] --> Step1[Reserve Inventory]
        Step1 --> Step2[Process Payment]
        Step2 --> Step3[Create Shipment]
        Step3 --> Success[Order Confirmed]
    end

    subgraph "Compensation Flow"
        Step3 -->|Failure| Comp3[Cancel Shipment]
        Step2 -->|Failure| Comp2[Refund Payment]
        Step1 -->|Failure| Comp1[Release Inventory]
        Comp3 --> Comp2
        Comp2 --> Comp1
        Comp1 --> Failed[Order Cancelled]
    end

    style Success fill:#c8e6c9
    style Failed fill:#ffcdd2
```

---

## Workflow 4: Payment Processing

### Overview

Secure payment processing with multiple payment methods.

### Payment Flow

```{mermaid}
flowchart TD
    Start([Payment Request]) --> ValidateAmount{Validate<br/>Amount}
    ValidateAmount -->|Invalid| Error1[400 Invalid Amount]
    ValidateAmount -->|Valid| SelectMethod{Payment<br/>Method}

    SelectMethod --> CreditCard[Credit Card]
    SelectMethod --> PayPal[PayPal]
    SelectMethod --> Wallet[Digital Wallet]

    CreditCard --> ProcessCC[Process via Gateway]
    PayPal --> ProcessPP[Process via PayPal API]
    Wallet --> ProcessWallet[Deduct from Wallet]

    ProcessCC --> Verify3DS{3D Secure<br/>Required?}
    Verify3DS -->|Yes| Redirect3DS[Redirect to 3DS]
    Verify3DS -->|No| AuthorizeCC[Authorize Payment]
    Redirect3DS --> AuthorizeCC

    ProcessPP --> AuthorizePP[Authorize Payment]
    ProcessWallet --> AuthorizeWallet[Authorize Payment]

    AuthorizeCC --> CheckResult{Result}
    AuthorizePP --> CheckResult
    AuthorizeWallet --> CheckResult

    CheckResult -->|Success| Capture[Capture Payment]
    CheckResult -->|Failed| Error2[Payment Failed]
    CheckResult -->|Pending| Wait[Wait for Async Callback]

    Capture --> UpdateOrder[Update Order Status]
    UpdateOrder --> SendReceipt[Send Receipt]
    SendReceipt --> Success[Payment Complete]

    Error1 --> End([End])
    Error2 --> End
    Success --> End
    Wait --> End
```

### Payment State Machine

```{mermaid}
stateDiagram-v2
    [*] --> Initiated: Create Payment

    Initiated --> Authorizing: Submit to Gateway
    Authorizing --> Authorized: Authorization Success
    Authorizing --> Failed: Authorization Failed

    Authorized --> Capturing: Capture Request
    Capturing --> Captured: Capture Success
    Capturing --> Failed: Capture Failed

    Captured --> Settled: Settlement Process

    Authorized --> Voided: Void Request
    Captured --> Refunding: Refund Request
    Refunding --> Refunded: Refund Complete

    Failed --> [*]
    Voided --> [*]
    Refunded --> [*]
    Settled --> [*]
```

---

## Error Handling Patterns

### Retry Strategy

```{mermaid}
flowchart TD
    Start([Operation Start]) --> Execute[Execute Operation]
    Execute --> Check{Success?}

    Check -->|Yes| Success[Operation Successful]
    Check -->|No| Retryable{Retryable<br/>Error?}

    Retryable -->|No| Fatal[Fatal Error]
    Retryable -->|Yes| CheckRetries{Retry Count<br/>< Max?}

    CheckRetries -->|Yes| Wait[Wait with Backoff]
    CheckRetries -->|No| GiveUp[Max Retries Exceeded]

    Wait --> Execute

    Success --> End([End])
    Fatal --> End
    GiveUp --> End

    style Success fill:#c8e6c9
    style Fatal fill:#ffcdd2
    style GiveUp fill:#ffcdd2
```

### Circuit Breaker Pattern

```{mermaid}
sequenceDiagram
    participant Service
    participant CircuitBreaker
    participant ExternalAPI

    Note over CircuitBreaker: State: CLOSED

    Service->>CircuitBreaker: Call API
    CircuitBreaker->>ExternalAPI: Forward Request
    ExternalAPI-->>CircuitBreaker: Response
    CircuitBreaker-->>Service: Response

    Note over ExternalAPI: API becomes unhealthy

    Service->>CircuitBreaker: Call API
    CircuitBreaker->>ExternalAPI: Forward Request
    ExternalAPI--xCircuitBreaker: Timeout/Error
    CircuitBreaker-->>Service: Error

    Note over CircuitBreaker: Failure count: 5<br/>State: OPEN

    Service->>CircuitBreaker: Call API
    CircuitBreaker-->>Service: Fast Fail (Circuit Open)

    Note over CircuitBreaker: 30s passed<br/>State: HALF_OPEN

    Service->>CircuitBreaker: Call API
    CircuitBreaker->>ExternalAPI: Test Request
    ExternalAPI-->>CircuitBreaker: Success
    CircuitBreaker-->>Service: Success

    Note over CircuitBreaker: Test successful<br/>State: CLOSED
```

## Related Documentation

- [Architecture](02-architecture.md) - System architecture
- [Data Model](04-data-and-api.md) - Database schema and APIs
- [Testing](07-testing.md) - Test scenarios for workflows

---

**Last Updated**: [Date]
**Authors**: [Names]
**Version**: 1.0
