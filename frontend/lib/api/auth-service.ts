// Authentication service for interacting with Django backend
// Connects to the User and Organization models in the Accounts section of the schema

import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment, mockUser } from "./mock-data"

// Types that match Django models
export interface User {
  id: string
  org: {
    id: string
    name: string
  }
  email: string
  name: string
  role: "admin" | "engineer" | "analyst" | "guest"
  created_at: string
  last_login: string | null
  is_active: boolean
  is_staff: boolean
  avatar?: string // Not in DB schema but useful for UI
}

// Login credentials
interface LoginCredentials {
  email: string
  password: string
}

// Login response from Django
interface LoginResponse {
  user: User
  access: string
  refresh: string
}

// Base URL for API calls
const API_BASE_URL = "/api"

/**
 * Login user with Django backend
 * Connects to Django's authentication system (likely using DRF TokenAuthentication)
 */
export async function loginUser(credentials: LoginCredentials): Promise<User | null> {
  // 디버깅 모드에서는 항상 실제 API 사용
  const forceRealApi = false; // 로그인은 모의 API 우선 사용 (isPreviewEnvironment 조건에 따라)
  
  // Use mock data in preview environment - but only if not forcing real API
  if (!forceRealApi && isPreviewEnvironment()) {
    console.log("테스트 모드 - 모의 로그인 사용")
    // Simple validation for demo
    if (credentials.email && credentials.password.length >= 4) {
      // Store mock user data
      localStorage.setItem("user", JSON.stringify(mockUser))
      localStorage.setItem("accessToken", "mock-token-123")
      localStorage.setItem("refreshToken", "mock-refresh-token-123")
      return mockUser
    }
    return null
  }
  
  // 실제 API 사용을 위해 상태 설정
  localStorage.setItem("forceRealApi", "true");
  localStorage.setItem("backendAvailable", "true");

  try {
    const response = await fetch(`${API_BASE_URL}/auth/login/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(credentials),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Login failed")
    }

    const data: LoginResponse = await response.json()

    // Store JWT tokens in localStorage for subsequent API calls
    localStorage.setItem("accessToken", data.access)
    localStorage.setItem("refreshToken", data.refresh)

    // Store user data
    localStorage.setItem("user", JSON.stringify(data.user))

    return data.user
  } catch (error: any) {
    console.error("Login error:", error)
    toast({
      title: "Login failed",
      description: error.message || "Please check your credentials and try again",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Get current user from localStorage or refresh from backend
 */
export async function getCurrentUser(): Promise<User | null> {
  const storedUser = localStorage.getItem("user")
  if (storedUser) {
    return JSON.parse(storedUser)
  }

  // If in preview environment, return null (not logged in)
  if (isPreviewEnvironment()) {
    return null
  }

  // If no stored user but access token exists, fetch user data
  const accessToken = localStorage.getItem("accessToken")
  if (accessToken) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/me/`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      })

      if (!response.ok) {
        // If token is expired, try refreshing it
        if (response.status === 401) {
          const refreshed = await refreshToken()
          if (refreshed) {
            // Retry with new token
            return getCurrentUser()
          }
        }
        throw new Error("Failed to get user data")
      }

      const user: User = await response.json()
      localStorage.setItem("user", JSON.stringify(user))
      return user
    } catch (error: any) {
      console.error("Error fetching current user:", error)
      // Clear invalid tokens
      localStorage.removeItem("accessToken")
      localStorage.removeItem("refreshToken")
      localStorage.removeItem("user")
      return null
    }
  }

  return null
}

/**
 * Logout user
 * Calls Django logout endpoint to invalidate token
 */
export async function logoutUser(): Promise<boolean> {
  // In preview environment, just clear local storage
  if (isPreviewEnvironment()) {
    localStorage.removeItem("accessToken")
    localStorage.removeItem("refreshToken")
    localStorage.removeItem("user")
    return true
  }

  const accessToken = localStorage.getItem("accessToken")

  if (accessToken) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/logout/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      })

      if (!response.ok) {
        console.warn("Logout API call failed, clearing local storage anyway")
      }
    } catch (error: any) {
      console.error("Logout error:", error)
    }
  }

  // Clear local storage regardless of API success
  localStorage.removeItem("accessToken")
  localStorage.removeItem("refreshToken")
  localStorage.removeItem("user")

  return true
}

/**
 * Refresh the JWT token using the refresh token
 * @returns boolean indicating if the refresh was successful
 */
async function refreshToken(): Promise<boolean> {
  const refreshToken = localStorage.getItem("refreshToken")
  
  if (!refreshToken) {
    return false
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/auth/token/refresh/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh: refreshToken }),
    })
    
    if (!response.ok) {
      throw new Error("Failed to refresh token")
    }
    
    const data = await response.json()
    localStorage.setItem("accessToken", data.access)
    
    // If the API also returns a new refresh token, update it
    if (data.refresh) {
      localStorage.setItem("refreshToken", data.refresh)
    }
    
    return true
  } catch (error: any) {
    console.error("Token refresh error:", error)
    // Clear tokens if refresh fails
    localStorage.removeItem("accessToken")
    localStorage.removeItem("refreshToken")
    localStorage.removeItem("user")
    return false
  }
}

/**
 * Update user profile
 * Connects to Django User model update endpoint
 */
export async function updateUserProfile(userData: Partial<User>): Promise<User | null> {
  // In preview environment, update the mock user
  if (isPreviewEnvironment()) {
    const storedUser = localStorage.getItem("user")
    if (storedUser) {
      const currentUser = JSON.parse(storedUser)
      const updatedUser = { ...currentUser, ...userData }
      localStorage.setItem("user", JSON.stringify(updatedUser))
      return updatedUser
    }
    return null
  }

  const accessToken = localStorage.getItem("accessToken")

  if (!accessToken) {
    toast({
      title: "Authentication error",
      description: "You must be logged in to update your profile",
      variant: "destructive",
    })
    return null
  }

  try {
    const response = await fetch(`${API_BASE_URL}/auth/profile/`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(userData),
    })

    if (!response.ok) {
      // If token is expired, try refreshing it
      if (response.status === 401) {
        const refreshed = await refreshToken()
        if (refreshed) {
          // Retry with new token
          return updateUserProfile(userData)
        }
      }
      throw new Error("Failed to update profile")
    }

    const updatedUser: User = await response.json()

    // Update stored user data
    localStorage.setItem("user", JSON.stringify(updatedUser))

    return updatedUser
  } catch (error: any) {
    console.error("Profile update error:", error)
    toast({
      title: "Update failed",
      description: error.message || "Failed to update profile",
      variant: "destructive",
    })
    return null
  }
}
