# Project Knowledge Pack (PKP)

A comprehensive toolkit for creating AI-readable project documentation that enables AI assistants to understand, navigate, and contribute to software projects like a human engineer.

## Overview

This skill helps you create a **Project Knowledge Pack** - a structured set of documentation that enables AI to:
- **Locate**: Quickly find code, configs, and scripts (Repo Map, directory tree, key entry points)
- **Understand**: Grasp architecture, dependencies, data flow, and business logic
- **Execute**: Set up environment, run tests, modify code, and create PRs
- **Verify**: Design tests, locate bugs, and perform regression testing

## Commands

### `/pkp-init [project-root] [--sphinx]`
Initialize a Project Knowledge Pack structure in a software project.

**Examples:**
- `/pkp-init` - Initialize in current directory
- `/pkp-init /path/to/project` - Initialize in specific project
- `/pkp-init --sphinx` - Initialize with Sphinx + MyST support

**Instructions:**
1. **Determine target directory:**
   - Use current directory if no path provided
   - Use provided path if specified
   - Verify it's a valid project directory (has src/, package.json, go.mod, pom.xml, etc.)

2. **Create directory structure:**
   ```
   man/
   ├── 00-overview.md
   ├── 01-repo-map.md
   ├── 02-architecture.md
   ├── 03-workflows.md
   ├── 04-data-and-api.md
   ├── 05-conventions.md
   ├── 06-runbook.md
   ├── 07-testing.md
   ├── ai-guide.md              # AI usage guide
   ├── index.md                 # Documentation entry point
   ├── adr/
   │   ├── index.md
   │   └── template.md
   ├── changes/
   │   ├── index.md
   │   └── _template/
   │       ├── proposal.md
   │       ├── design.md
   │       ├── tasks.md
   │       └── specs/
   └── _static/                 # Sphinx static files (if --sphinx)
       └── custom.css
   ```

3. **Create template files:**
   - Generate each document with structured sections and instructions
   - Use markdown format with clear headings
   - Include examples and guidelines
   - Use MyST syntax for enhanced features (admonitions, directives, etc.)

4. **If --sphinx flag is provided:**
   - Copy Sphinx configuration files:
     - `conf.py` - Sphinx configuration
     - `requirements.txt` - Python dependencies
     - `Makefile` - Build commands (Unix/macOS)
     - `make.bat` - Build commands (Windows)
   - Create `_static/` directory with `custom.css`
   - Update all markdown files to use MyST syntax
   - Create `index.md` with toctree

5. **Output:**
   - List created files
   - If Sphinx enabled, show build instructions:
     ```bash
     cd man/
     pip install -r requirements.txt  # Install dependencies
     make html                         # Build HTML documentation
     make serve                        # Build and serve with auto-reload
     ```
   - Provide next steps
   - Suggest starting with `/pkp-overview`

### `/pkp-overview [project-root]`
Generate or update the project overview document (00-overview.md).

**Instructions:**
1. **Analyze the project:**
   - Read package.json, go.mod, pom.xml, requirements.txt, etc.
   - Identify technology stack
   - Detect project type (web app, API, CLI tool, library, etc.)
   - Check for README.md for existing descriptions

2. **Generate content including:**
   - Project purpose and business boundaries
   - What the project does NOT do (important!)
   - Key user roles and core use cases
   - Technology stack: languages, frameworks, middleware, databases
   - Deployment model: monolith / microservices / multi-platform
   - Performance / consistency / availability targets (SLO)

3. **Output format:**
   ```markdown
   # Project Overview

   ## Purpose
   [What problem does this solve?]

   ## Business Boundaries
   ### What We Do
   - ...

   ### What We Don't Do
   - ...

   ## Key User Roles
   - Role 1: ...
   - Role 2: ...

   ## Core Use Cases
   1. ...
   2. ...

   ## Technology Stack
   - Language: ...
   - Framework: ...
   - Database: ...
   - Middleware: ...

   ## Deployment Model
   [Monolith / Microservices / Multi-platform]

   ## Quality Targets
   - Performance: ...
   - Availability: ...
   - Consistency: ...
   ```

4. Save to `man/00-overview.md`

### `/pkp-repo-map [project-root]`
Generate or update the repository map document (01-repo-map.md).

**Instructions:**
1. **Analyze repository structure:**
   - Run `tree -L 3 -d` or equivalent to get directory tree
   - Identify key directories and their purposes
   - Find entry points: main files, app initialization, routers, DI containers
   - Document naming conventions and layering patterns

2. **Generate content including:**
   - Directory tree (2-3 levels deep)
   - Purpose of each major directory
   - Key entry points with file paths
   - Conventions: naming, layering, common patterns (CQRS, DDD, Clean Architecture)

3. **Output format:**
   ```markdown
   # Repository Map

   ## Directory Structure
   ```
   [tree output]
   ```

   ## Directory Responsibilities
   - `/src`: ...
   - `/tests`: ...
   - `/config`: ...
   - `/docs`: ...

   ## Key Entry Points
   - Main: `src/main.go`
   - App: `src/app/app.go`
   - Router: `src/router/router.go`
   - Config: `config/config.go`

   ## Conventions
   ### Naming
   - Files: ...
   - Functions: ...
   - Variables: ...

   ### Layering
   - Presentation layer: ...
   - Business layer: ...
   - Data layer: ...

   ### Common Patterns
   - Pattern 1: ...
   - Pattern 2: ...
   ```

4. Save to `man/01-repo-map.md`

### `/pkp-architecture [project-root]`
Generate or update the architecture document (02-architecture.md).

**Instructions:**
1. **Analyze architecture:**
   - Identify components and their boundaries
   - Map key call chains (from entry to DB/external API)
   - Document module dependency rules
   - Identify cross-cutting concerns

2. **Generate content including:**
   - Component diagram (services/modules boundaries)
   - Key call chains with sequence diagrams if needed
   - Module dependency rules (who can depend on whom)
   - Cross-cutting concerns: auth, logging, error codes, transactions, cache, retry, idempotency

3. **Output format:**
   ```markdown
   # Architecture

   ## Component Overview
   [Component diagram or description]

   ## Key Call Chains
   ### Use Case 1
   ```
   Entry → Service A → Service B → Database
   ```

   ## Module Dependencies
   - Module A can depend on: ...
   - Module B can depend on: ...

   ## Cross-Cutting Concerns
   ### Authentication
   - ...

   ### Logging
   - ...

   ### Error Handling
   - ...

   ### Transactions
   - ...

   ### Caching
   - ...
   ```

4. Save to `man/02-architecture.md`

### `/pkp-workflow <workflow-name> [project-root]`
Generate or update a business workflow document (03-workflows.md).

**Examples:**
- `/pkp-workflow user-registration`
- `/pkp-workflow order-processing`

**Instructions:**
1. **Analyze the workflow:**
   - Identify workflow steps
   - Map input/output
   - Document validations and error branches
   - Find involved tables/messages/events
   - Locate key code entry points

2. **Generate content including:**
   - Workflow steps (with sequence diagram or state machine)
   - Input/output specifications
   - Key validations and error branches
   - Involved tables/messages/events
   - Key code entry points (file paths + function names)

3. **Output format:**
   ```markdown
   # Business Workflows

   ## Workflow: [Name]

   ### Overview
   [Brief description]

   ### Steps
   1. Step 1: ...
   2. Step 2: ...
   3. Step 3: ...

   ### Input/Output
   - Input: ...
   - Output: ...

   ### Validations
   - Validation 1: ...
   - Validation 2: ...

   ### Error Branches
   - Error 1: ...
   - Error 2: ...

   ### Data Entities
   - Table 1: ...
   - Message 1: ...
   - Event 1: ...

   ### Code Entry Points
   - File: `src/service/workflow.go`
   - Function: `ProcessWorkflow()`
   ```

4. Save to `man/03-workflows.md` (append or update)

### `/pkp-data-api [project-root]`
Generate or update the data model and API contract document (04-data-and-api.md).

**Instructions:**
1. **Analyze data models:**
   - Identify database schema (tables, fields, indexes, constraints)
   - Find migration tools
   - Map domain objects to DTOs
   - Document external interfaces (OpenAPI, GraphQL, Proto)
   - Document events/messages (topics, schemas, compatibility)

2. **Generate content including:**
   - Database schema overview
   - Core tables with fields, indexes, and constraints
   - Migration tools and processes
   - Domain object to DTO mapping rules
   - External interface specifications
   - Event/message specifications

3. Save to `man/04-data-and-api.md`

### `/pkp-conventions [project-root]`
Generate or update the engineering conventions document (05-conventions.md).

**Instructions:**
1. **Analyze conventions:**
   - Code style guidelines
   - Lint/format configuration
   - Error handling patterns
   - Logging standards
   - Configuration management
   - Feature flags and versioning
   - Anti-patterns (prohibited practices)

2. **Generate content including:**
   - Code style guide
   - Lint/format rules
   - Error handling patterns
   - Logging standards (fields, traceId)
   - Config management (env, files, secrets)
   - Feature flags and versioning
   - Anti-patterns with explanations

3. Save to `man/05-conventions.md`

### `/pkp-runbook [project-root]`
Generate or update the operations runbook (06-runbook.md).

**Instructions:**
1. **Analyze operational procedures:**
   - One-command startup procedures
   - Test execution commands
   - Local debugging setup
   - Common troubleshooting scenarios

2. **Generate content including:**
   - One-command startup (docker compose, devcontainer, make)
   - Test execution (all tests, unit, integration, E2E)
   - Local debugging (ports, dependencies, mocks)
   - Common troubleshooting (DB connection, migrations, permissions, cache)

3. Save to `man/06-runbook.md`

### `/pkp-testing [project-root]`
Generate or update the testing strategy document (07-testing.md).

**Instructions:**
1. **Analyze testing approach:**
   - Test pyramid structure
   - Critical path test cases
   - Test data construction
   - Coverage thresholds
   - Flaky test handling

2. **Generate content including:**
   - Test pyramid (unit/contract/integration/E2E scopes)
   - Critical path test checklist
   - Test data construction and isolation strategies
   - Coverage thresholds
   - Flaky test handling procedures

3. Save to `man/07-testing.md`

### `/pkp-adr <title>`
Create a new Architecture Decision Record (ADR).

**Examples:**
- `/pkp-adr "Use PostgreSQL over MongoDB"`
- `/pkp-adr "Adopt event sourcing for order service"`

**Instructions:**
1. **Create ADR file:**
   - Generate sequential number (0001, 0002, etc.)
   - Create filename: `man/adr/0001-[slug].md`

2. **Generate ADR with structure:**
   ```markdown
   # ADR-0001: [Title]

   ## Status
   Proposed | Accepted | Deprecated | Superseded

   ## Context
   [Background and problem statement]

   ## Decision
   [The decision that was made]

   ## Alternatives Considered
   ### Option 1: ...
   - Pros: ...
   - Cons: ...

   ### Option 2: ...
   - Pros: ...
   - Cons: ...

   ## Consequences
   ### Positive
   - ...

   ### Negative
   - ...

   ### Neutral
   - ...

   ## Related
   - Related to: ADR-XXXX
   - Supersedes: ADR-XXXX
   - Superseded by: ADR-XXXX
   ```

3. Save to `man/adr/0001-[slug].md`

### `/pkp-change <change-id> [type]`
Create a new change proposal following OpenSpec workflow.

**Examples:**
- `/pkp-change add-payment-gateway`
- `/pkp-change update-auth-flow`

**Instructions:**
1. **Create change directory:**
   - Create `man/changes/<change-id>/`
   - Copy templates from `_template/`

2. **Generate change proposal files:**
   - `proposal.md`: Why change, what changes, impact
   - `design.md`: Technical decisions, goals, non-goals, risks
   - `tasks.md`: Implementation checklist
   - `specs/<capability>/spec.md`: Specification changes

3. **Proposal template:**
   ```markdown
   # Change Proposal: [Title]

   ## Why
   [Problem statement and motivation]

   ## What Changes
   [High-level summary of changes]

   ## Impact
   - Affected components: ...
   - Breaking changes: ...
   - Migration required: ...
   - Testing scope: ...

   ## Timeline
   - Proposed start: ...
   - Estimated completion: ...

   ## Stakeholders
   - Owner: ...
   - Reviewers: ...
   - Impacted teams: ...
   ```

4. Save files in `man/changes/<change-id>/`

### `/pkp-feed-ai [round]`
Generate prompts to feed project knowledge to AI in three progressive rounds.

**Examples:**
- `/pkp-feed-ai 1` - Round 1: Establish map
- `/pkp-feed-ai 2` - Round 2: Deep dive into workflows
- `/pkp-feed-ai 3` - Round 3: Module acceptance

**Instructions:**

**Round 1: Establish the Map**
1. Read `00-overview.md`, `01-repo-map.md`, and architecture diagrams
2. Generate prompt:
   ```
   You will serve as a senior engineer for this project. Below are the Project Overview, Repo Map, and Architecture.

   Tasks:
   1) Summarize the system's components, entry points, and key data flows in one page.
   2) List the 10 most critical pieces of information you still need (by priority).
   3) Propose a small task suitable as a "first hands-on modification" (include tests).
   ```
3. Output the prompt for user to feed to AI

**Round 2: Deep Dive into 2-3 Key Business Workflows**
1. For each workflow, read workflow document + related code files
2. Generate prompt:
   ```
   Below is the "[Workflow Name]" workflow description + related code files.

   Tasks:
   1) Draw the call chain (function/module level), marking transaction boundaries, idempotency points, and retry points.
   2) Identify 5 boundary conditions most likely to cause bugs.
   3) Provide a complete test checklist (unit/integration/E2E), explaining what each tests.
   ```
3. Output the prompt for user to feed to AI

**Round 3: Module-by-Module Acceptance**
1. For each module, read module README + tests + sample data
2. Generate prompt:
   ```
   Module: [Name]
   - Responsibilities: ...
   - API: ...
   - Dependencies: ...
   - Examples: ...

   Tasks:
   1) Provide improvement suggestions.
   2) Complete a small modification (PR-level with tests).
   ```
3. Output the prompt for user to feed to AI

### `/pkp-verify [project-root]`
Verify the completeness and quality of the Project Knowledge Pack.

**Instructions:**
1. **Check file existence:**
   - Verify all required documents exist
   - Check for ADRs
   - Check for changes (if any)

2. **Analyze content quality:**
   - Check if documents have substantial content (not just templates)
   - Verify links are valid
   - Check for code references (file:line format)

3. **Generate report:**
   ```markdown
   # PKP Verification Report

   ## Completeness
   - ✓ 00-overview.md (Complete)
   - ✗ 01-repo-map.md (Missing)
   - ⚠ 02-architecture.md (Template only)

   ## Quality Checks
   - Document length: ...
   - Code references: ...
   - Valid links: ...

   ## Recommendations
   1. Complete missing documents
   2. Add more ADRs for key decisions
   3. Create runbook for common operations

   ## Next Steps
   - `/pkp-repo-map` to generate repo map
   - `/pkp-adr "Decision title"` to document decisions
   ```

4. Output the report

### `/pkp-build [format]`
Build the Project Knowledge Pack documentation using Sphinx + MyST.

**Examples:**
- `/pkp-build` - Build HTML documentation (default)
- `/pkp-build html` - Build HTML documentation
- `/pkp-build pdf` - Build PDF documentation
- `/pkp-build serve` - Build and serve with auto-reload

**Prerequisites:**
- Project must be initialized with `--sphinx` flag
- Sphinx dependencies must be installed

**Instructions:**
1. **Check prerequisites:**
   - Verify `man/conf.py` exists
   - Verify `man/requirements.txt` exists
   - Check if dependencies are installed

2. **Install dependencies if needed:**
   ```bash
   cd man/
   pip install -r requirements.txt
   ```

3. **Build documentation:**
   - HTML: `make html` or `sphinx-build -b html . _build/html`
   - PDF: `make latexpdf` or build LaTeX then convert to PDF
   - Serve: `make serve` or `sphinx-autobuild . _build/html`

4. **Output:**
   - Show build progress
   - Report any errors or warnings
   - Provide URL to view documentation:
     - HTML: `file://man/_build/html/index.html`
     - Serve: `http://localhost:8000`

5. **Handle errors:**
   - Missing dependencies: Show installation command
   - Syntax errors: Point to problematic files
   - Build warnings: Explain and suggest fixes

### `/pkp-help`
Display help information for all Project Knowledge Pack commands.

**Instructions:**
1. Display a summary of all available PKP commands
2. Show examples for each command
3. Provide tips and best practices
4. Include file structure information
5. Reference the methodology article

## File Structure

```
man/
├── 00-overview.md          # Project purpose, stack, deployment
├── 01-repo-map.md          # Directory structure, entry points
├── 02-architecture.md      # Components, dependencies, patterns
├── 03-workflows.md         # Business workflows with code refs
├── 04-data-and-api.md      # DB schema, APIs, events
├── 05-conventions.md       # Code style, error handling, anti-patterns
├── 06-runbook.md           # Setup, testing, debugging
├── 07-testing.md           # Test strategy, coverage, flaky tests
├── adr/                    # Architecture Decision Records
│   ├── README.md
│   ├── template.md
│   └── 0001-decision.md
└── changes/                # Change proposals (OpenSpec style)
    ├── README.md
    ├── _template/
    │   ├── proposal.md
    │   ├── design.md
    │   ├── tasks.md
    │   └── specs/
    └── add-feature-x/
        ├── proposal.md
        ├── design.md
        ├── tasks.md
        └── specs/
```

## Methodology

This skill implements the methodology described in:
- Article: [如何让 AI 真正"懂"你的项目？](../../content/journal/journal_20260216_ai-project-knowledge-pack.md)
- Related: [OpenSpec](https://github.com/jpoehnelt/openspec) for change management

## Tips

1. **Start with the basics:**
   - Run `/pkp-init` first to create structure
   - Then `/pkp-overview` and `/pkp-repo-map`
   - These two are the foundation for AI understanding

2. **Document as you go:**
   - Create ADRs when making important decisions
   - Update workflows when adding new features
   - Keep runbook up to date with operational changes

3. **Feed AI progressively:**
   - Use `/pkp-feed-ai` to generate appropriate prompts
   - Don't dump all code at once
   - Let AI build understanding layer by layer

4. **Keep it maintained:**
   - Run `/pkp-verify` regularly
   - Update docs alongside code changes
   - Treat docs as "living documentation"

5. **Leverage for onboarding:**
   - Use the same docs for new team members
   - Knowledge Pack serves both humans and AI
   - Reduces onboarding time significantly

## Integration with Other Skills

- **OpenSpec**: For change management workflow
- **SpecKit**: For software project planning
- **Living Documentation**: For auto-generating docs from code
- **TDD/MDD**: For test-driven and metrics-driven development

## Best Practices

1. **Write for both humans and AI:**
   - Use clear, structured language
   - Include concrete examples
   - Reference actual code locations

2. **Document the "why", not just "what":**
   - ADRs are critical for understanding decisions
   - Explain trade-offs and constraints
   - Record what was considered but rejected

3. **Keep it actionable:**
   - Include commands, not just descriptions
   - Provide troubleshooting steps
   - Add test checklists

4. **Make it discoverable:**
   - Link documents together
   - Use consistent naming
   - Maintain a clear structure

5. **Version control everything:**
   - All docs in git with code
   - Update docs in same PR as code
   - Review doc changes like code changes
