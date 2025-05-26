// Code service for interacting with Django backend
// Connects to the GitRepository and CodeFile models in the Knowledge section

import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "./mock-data"

// Types that match Django models
export interface GitRepository {
  id: string
  org: string // Organization ID
  repo_url: string
  default_branch: string
  fetched_at: string | null
}

export interface CodeFile {
  id: string
  repo: string // Repository ID
  file_path: string
  language: string
  latest_commit: string
  loc: number | null
}

// Mock repositories for preview environment
const mockRepositories: GitRepository[] = [
  {
    id: "repo1",
    org: "org123",
    repo_url: "https://github.com/example/project",
    default_branch: "main",
    fetched_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
  },
]

// Mock files for preview environment
const mockFiles: Record<string, CodeFile[]> = {
  repo1: [
    {
      id: "file1",
      repo: "repo1",
      file_path: "src/main.py",
      language: "python",
      latest_commit: "abc123",
      loc: 120,
    },
    {
      id: "file2",
      repo: "repo1",
      file_path: "src/utils.js",
      language: "javascript",
      latest_commit: "def456",
      loc: 85,
    },
  ],
}

// Mock file content for preview environment
const mockFileContent: Record<string, string> = {
  file1: `def fibonacci(n):
    """Generate fibonacci sequence up to n"""
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result

print(fibonacci(100))`,
  file2: `function fetchUserData(userId) {
  return fetch(\`/api/users/\${userId}\`)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('User data:', data);
      return data;
    })
    .catch(error => {
      console.error('Error fetching user data:', error);
    });
}`,
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
 * Get all repositories for the current organization
 * Maps to Django GitRepository model filtered by organization
 */
export async function getRepositories(): Promise<GitRepository[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve([...mockRepositories])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/repositories/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch repositories")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching repositories:", error)
    toast({
      title: "Error",
      description: "Failed to load repositories",
      variant: "destructive",
    })
    return []
  }
}

/**
 * Add a new repository
 * Creates a new GitRepository record and triggers fetching and indexing
 */
export async function addRepository(repoUrl: string, defaultBranch = "main"): Promise<GitRepository | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const newRepo: GitRepository = {
      id: `repo-${Date.now()}`,
      org: "org123",
      repo_url: repoUrl,
      default_branch: defaultBranch,
      fetched_at: new Date().toISOString(),
    }

    // Add to mock repositories
    mockRepositories.push(newRepo)

    // Initialize empty files array for this repo
    mockFiles[newRepo.id] = []

    return Promise.resolve(newRepo)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/repositories/`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        repo_url: repoUrl,
        default_branch: defaultBranch,
      }),
    })

    if (!response.ok) {
      throw new Error("Failed to add repository")
    }

    return await response.json()
  } catch (error) {
    console.error("Error adding repository:", error)
    toast({
      title: "Error",
      description: "Failed to add repository",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Get files for a specific repository
 * Maps to Django CodeFile model filtered by repository
 */
export async function getRepositoryFiles(repoId: string): Promise<CodeFile[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve(mockFiles[repoId] || [])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/repositories/${repoId}/files/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch repository files")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching repository files:", error)
    toast({
      title: "Error",
      description: "Failed to load repository files",
      variant: "destructive",
    })
    return []
  }
}

/**
 * Get file content
 * This would fetch the actual content of a file from the backend
 */
export async function getFileContent(fileId: string): Promise<string | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve(mockFileContent[fileId] || null)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/knowledge/files/${fileId}/content/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch file content")
    }

    const data = await response.json()
    return data.content
  } catch (error) {
    console.error("Error fetching file content:", error)
    toast({
      title: "Error",
      description: "Failed to load file content",
      variant: "destructive",
    })
    return null
  }
}
