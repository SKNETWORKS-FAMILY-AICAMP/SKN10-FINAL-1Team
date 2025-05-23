# Database Schema Documentation

This document provides an overview of all database models in the backend applications.

## Table of Contents
- [Accounts](#accounts)
- [Conversations](#conversations)
- [Knowledge](#knowledge)
- [MLOps](#mlops)

---

## Accounts

### Organization
- **Table**: `organizations`
- **Description**: Represents an organization that can have multiple users.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `name`: CharField (unique)
  - `created_at`: DateTimeField (auto_now_add)

### User
- **Table**: `users`
- **Description**: Custom user model extending AbstractBaseUser with organization-based access control.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `org`: ForeignKey to Organization
  - `email`: EmailField (unique)
  - `name`: CharField
  - `role`: CharField (choices: admin, engineer, analyst, guest)
  - `created_at`: DateTimeField (auto_now_add)
  - `last_login`: DateTimeField (nullable)
  - `is_active`: BooleanField
  - `is_staff`: BooleanField

---

## Conversations

### AgentType (Enum)
- code
- rag
- analytics

### ChatSession
- **Table**: `chat_sessions`
- **Description**: Represents a chat conversation session.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `user`: ForeignKey to User
  - `agent_type`: CharField (choices: code, rag, analytics)
  - `started_at`: DateTimeField (auto_now_add)
  - `ended_at`: DateTimeField (nullable)

### ChatMessage
- **Table**: `chat_messages`
- **Description**: Individual messages within a chat session.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `session`: ForeignKey to ChatSession
  - `role`: CharField (user/assistant/system)
  - `content`: TextField
  - `created_at`: DateTimeField (auto_now_add)

### LlmCall
- **Table**: `llm_calls`
- **Description**: Logs of LLM API calls.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `session`: ForeignKey to ChatSession
  - `provider`: CharField
  - `model`: CharField
  - `prompt_tokens`: PositiveIntegerField
  - `completion_tokens`: PositiveIntegerField
  - `cost_usd`: DecimalField
  - `latency_ms`: PositiveIntegerField (nullable)
  - `called_at`: DateTimeField (auto_now_add)

---

## Knowledge

### DocType (Enum)
- policy
- product
- tech_manual

### Document
- **Table**: `documents`
- **Description**: Represents an uploaded document.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `org`: ForeignKey to Organization
  - `title`: CharField
  - `doc_type`: CharField (choices: policy, product, tech_manual)
  - `s3_key`: TextField (unique)
  - `version`: CharField
  - `pinecone_ns`: CharField (Pinecone namespace)
  - `uploaded_by`: ForeignKey to User (nullable)
  - `created_at`: DateTimeField (auto_now_add)

### GitRepository
- **Table**: `git_repositories`
- **Description**: Tracks Git repositories for code indexing.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `org`: ForeignKey to Organization
  - `repo_url`: TextField (unique)
  - `default_branch`: CharField (default: 'main')
  - `fetched_at`: DateTimeField (nullable)

### CodeFile
- **Table**: `code_files`
- **Description**: Represents a file in a Git repository.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `repo`: ForeignKey to GitRepository
  - `file_path`: TextField
  - `language`: CharField
  - `latest_commit`: CharField
  - `loc`: PositiveIntegerField (nullable)

### EmbedChunk
- **Table**: `embed_chunks`
- **Description**: Represents an embedded chunk of text from a document or code file.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `document`: ForeignKey to Document (nullable)
  - `file`: ForeignKey to CodeFile (nullable)
  - `chunk_index`: PositiveIntegerField
  - `pinecone_id`: CharField
  - `hash`: CharField (unique)

---

## MLOps

### ResultType (Enum)
- churn_pred
- viz_image
- timeseries_forecast

### AnalyticsResult
- **Table**: `analytics_results`
- **Description**: Stores results from analytics operations.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `user`: ForeignKey to User
  - `result_type`: CharField (choices: churn_pred, viz_image, timeseries_forecast)
  - `s3_key`: TextField
  - `meta`: JSONField (nullable)
  - `created_at`: DateTimeField (auto_now_add)

### ModelStage (Enum)
- staging
- production
- archived

### ModelArtifact
- **Table**: `model_artifacts`
- **Description**: Tracks machine learning model artifacts.
- **Fields**:
  - `id`: UUID (Primary Key)
  - `name`: CharField
  - `version`: CharField
  - `s3_key`: TextField
  - `stage`: CharField (choices: staging, production, archived)
  - `metrics`: JSONField (nullable)
  - `created_by`: ForeignKey to User (nullable)
  - `created_at`: DateTimeField (auto_now_add)

---

## Relationships

- **Organization** 1:N **User**
- **User** 1:N **ChatSession**
- **ChatSession** 1:N **ChatMessage**
- **ChatSession** 1:N **LlmCall**
- **Organization** 1:N **Document**
- **Organization** 1:N **GitRepository**
- **GitRepository** 1:N **CodeFile**
- **Document** 1:N **EmbedChunk**
- **CodeFile** 1:N **EmbedChunk**
- **User** 1:N **AnalyticsResult**
- **User** 1:N **ModelArtifact**
