// Document service for interacting with Django backend
// Connects to the Document and EmbedChunk models in the Knowledge section

import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "./mock-data"

// Types that match Django models
export type DocType = "policy" | "product" | "tech_manual" // Matches DocType enum in schema

export interface Document {
  id: string
  org: string // Organization ID
  title: string
  doc_type: DocType
  s3_key: string
  version: string
  pinecone_ns: string
  uploaded_by: string | null // User ID
  created_at: string
}

export interface DocumentSummary {
  id: string
  title: string
  type: DocType
  date: string
  summary?: string
  keyPoints?: string[]
  source?: string
}

// Mock document summaries for preview environment
const mockDocumentSummaries: Record<string, DocumentSummary> = {
  doc1: {
    id: "doc1",
    title: "Employee Handbook 2024",
    type: "policy",
    date: "January 15, 2024",
    summary: "This document outlines the company's policies, benefits, and expectations for all employees.",
    keyPoints: [
      "Updated remote work policy allows for 3 days of remote work per week",
      "New mental health benefits added to healthcare package",
      "Annual performance reviews will now be conducted quarterly",
      "Updated code of conduct with emphasis on inclusive workplace practices",
    ],
    source: "HR Department / company-policies/employee-handbook-2024.pdf",
  },
  doc2: {
    id: "doc2",
    title: "Product Roadmap Q2 2024",
    type: "product",
    date: "March 28, 2024",
    summary: "Strategic plan for product development in Q2 2024, including key features and release timelines.",
    keyPoints: [
      "AI-powered recommendation engine scheduled for May release",
      "Mobile app redesign to be completed by end of June",
      "API v2 deprecation planned for mid-Q2",
      "New enterprise tier features to be prioritized",
    ],
    source: "Product Team / roadmaps/q2-2024-roadmap.pptx",
  },
}

// Base URL for API calls
const API_BASE_URL = "/api"

/**
 * Get authentication headers for API calls
 */
function getAuthHeaders() {
  const token = localStorage.getItem("token")
  return {
    "Content-Type": "application/json",
    Authorization: `Token ${token}`,
  }
}

/**
 * Get all documents for the current organization
 * Maps to Django Document model filtered by organization
 */
export async function getDocuments(): Promise<Document[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve([
      {
        id: "doc1",
        org: "org123",
        title: "Employee Handbook 2024",
        doc_type: "policy",
        s3_key: "documents/handbook.pdf",
        version: "1.0",
        pinecone_ns: "org123_docs",
        uploaded_by: "user123",
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(), // 30 days ago
      },
      {
        id: "doc2",
        org: "org123",
        title: "Product Roadmap Q2 2024",
        doc_type: "product",
        s3_key: "documents/roadmap.pptx",
        version: "1.0",
        pinecone_ns: "org123_docs",
        uploaded_by: "user123",
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 7 days ago
      },
    ])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/documents/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch documents")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching documents:", error)
    toast({
      title: "Error",
      description: "Failed to load documents",
      variant: "destructive",
    })
    return []
  }
}

/**
 * Get document summary
 * This would be processed by the backend, likely using the EmbedChunk model
 * to retrieve relevant chunks and summarize them
 */
export async function getDocumentSummary(documentId: string): Promise<DocumentSummary | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve(mockDocumentSummaries[documentId] || null)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/documents/${documentId}/summary/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch document summary")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching document summary:", error)
    toast({
      title: "Error",
      description: "Failed to load document summary",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Upload a new document
 * Creates a new Document record and triggers processing to create EmbedChunks
 */
export async function uploadDocument(file: File, docType: DocType, title?: string): Promise<Document | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const newDoc: Document = {
      id: `doc-${Date.now()}`,
      org: "org123",
      title: title || file.name,
      doc_type: docType,
      s3_key: `documents/${file.name}`,
      version: "1.0",
      pinecone_ns: "org123_docs",
      uploaded_by: "user123",
      created_at: new Date().toISOString(),
    }

    // Create a mock summary for this document
    mockDocumentSummaries[newDoc.id] = {
      id: newDoc.id,
      title: newDoc.title,
      type: docType,
      date: new Date().toLocaleDateString(),
      summary: "This is an automatically generated summary for the uploaded document.",
      keyPoints: [
        "This is a mock document summary for preview purposes",
        "In production, this would be generated by processing the document content",
        "The document would be chunked and embedded for retrieval",
      ],
      source: `Uploaded by user / ${file.name}`,
    }

    return Promise.resolve(newDoc)
  }

  try {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("doc_type", docType)

    if (title) {
      formData.append("title", title)
    }

    const token = localStorage.getItem("token")

    const response = await fetch(`${API_BASE_URL}/knowledge/documents/`, {
      method: "POST",
      headers: {
        Authorization: `Token ${token}`,
      },
      body: formData,
    })

    if (!response.ok) {
      throw new Error("Failed to upload document")
    }

    return await response.json()
  } catch (error) {
    console.error("Error uploading document:", error)
    toast({
      title: "Error",
      description: "Failed to upload document",
      variant: "destructive",
    })
    return null
  }
}
