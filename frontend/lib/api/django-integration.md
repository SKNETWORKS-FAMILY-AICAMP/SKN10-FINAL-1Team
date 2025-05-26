# Django Backend Integration Guide

This document explains how to connect the AI Agent Platform frontend to the Django backend based on the provided database schema.

## Overview

The frontend connects to the Django backend through a set of API services that map to the database models. The main integration points are:

1. **Authentication** - Connects to the User and Organization models
2. **Chat Sessions** - Connects to the ChatSession and ChatMessage models
3. **Knowledge Base** - Connects to the Document, GitRepository, CodeFile, and EmbedChunk models
4. **Analytics** - Connects to the AnalyticsResult and ModelArtifact models

## API Services

We've created the following API service files to handle communication with the Django backend:

- `auth-service.ts` - Authentication and user management
- `chat-service.ts` - Chat sessions and messages
- `document-service.ts` - Document management and retrieval
- `code-service.ts` - Code repository management
- `analytics-service.ts` - Business analytics and visualizations

## Django Backend Setup

### 1. Django REST Framework

The backend should use Django REST Framework to expose API endpoints for the frontend to consume. Install it with:

\`\`\`bash
pip install djangorestframework
\`\`\`

Add it to your `INSTALLED_APPS` in `settings.py`:

\`\`\`python
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'rest_framework.authtoken',
    # ...
]
\`\`\`

### 2. Authentication

The frontend expects token-based authentication. Configure it in `settings.py`:

\`\`\`python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
}
\`\`\`

### 3. CORS Configuration

To allow the frontend to communicate with the backend, install django-cors-headers:

\`\`\`bash
pip install django-cors-headers
\`\`\`

Add it to your `INSTALLED_APPS` and configure it in `settings.py`:

\`\`\`python
INSTALLED_APPS = [
    # ...
    'corsheaders',
    # ...
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    # ... other middleware
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    # Add your frontend URL here
]
\`\`\`

## API Endpoints

The frontend expects the following API endpoints:

### Authentication

- `POST /api/auth/login/` - Login with email and password
- `GET /api/auth/me/` - Get current user
- `POST /api/auth/logout/` - Logout
- `PATCH /api/auth/profile/` - Update user profile

### Chat

- `GET /api/chat/sessions/` - Get all chat sessions
- `POST /api/chat/sessions/` - Create a new chat session
- `GET /api/chat/sessions/{id}/messages/` - Get messages for a session
- `POST /api/chat/sessions/{id}/messages/` - Send a message
- `PATCH /api/chat/sessions/{id}/` - Update a chat session (e.g., end it)

### Knowledge

- `GET /api/knowledge/documents/` - Get all documents
- `POST /api/knowledge/documents/` - Upload a new document
- `GET /api/knowledge/documents/{id}/summary/` - Get document summary
- `GET /api/knowledge/repositories/` - Get all repositories
- `POST /api/knowledge/repositories/` - Add a new repository
- `GET /api/knowledge/repositories/{id}/files/` - Get files for a repository
- `GET /api/knowledge/files/{id}/content/` - Get file content

### Analytics

- `GET /api/mlops/results/` - Get analytics results
- `GET /api/mlops/results/{id}/` - Get a specific analytics result
- `POST /api/mlops/generate-chart/` - Generate a business chart
- `GET /api/mlops/models/` - Get model artifacts

## Django Models to API Mapping

The frontend API services map to the Django models as follows:

- `User` → `auth-service.ts`
- `Organization` → `auth-service.ts`
- `ChatSession` → `chat-service.ts`
- `ChatMessage` → `chat-service.ts`
- `LlmCall` → (handled by backend)
- `Document` → `document-service.ts`
- `GitRepository` → `code-service.ts`
- `CodeFile` → `code-service.ts`
- `EmbedChunk` → (handled by backend)
- `AnalyticsResult` → `analytics-service.ts`
- `ModelArtifact` → `analytics-service.ts`

## Implementation Notes

1. The frontend uses TypeScript interfaces that match the Django model fields
2. API calls include authentication tokens in the headers
3. Error handling is implemented with toast notifications
4. Loading states are managed for API operations
5. The frontend supports the same enum values as the Django models (AgentType, DocType, etc.)
