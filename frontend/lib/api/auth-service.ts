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
  token: string
}

// Base URL for API calls
const API_BASE_URL = "/api"

/**
 * Login user with Django backend
 * Connects to Django's authentication system (likely using DRF TokenAuthentication)
 */
export async function loginUser(credentials: LoginCredentials): Promise<User | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    // Simple validation for demo
    if (credentials.email && credentials.password.length >= 4) {
      // Store mock user data
      localStorage.setItem("user", JSON.stringify(mockUser))
      localStorage.setItem("token", "mock-token-123")
      return mockUser
    }
    return null
  }

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

    // Store token in localStorage for subsequent API calls
    localStorage.setItem("token", data.token)

    // Store user data
    localStorage.setItem("user", JSON.stringify(data.user))

    return data.user
  } catch (error) {
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

  // If no stored user but token exists, fetch user data
  const token = localStorage.getItem("token")
  if (token) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/me/`, {
        headers: {
          Authorization: `Token ${token}`,
        },
      })

      if (!response.ok) {
        throw new Error("Failed to get user data")
      }

      const user: User = await response.json()
      localStorage.setItem("user", JSON.stringify(user))
      return user
    } catch (error) {
      console.error("Error fetching current user:", error)
      // Clear invalid token
      localStorage.removeItem("token")
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
    localStorage.removeItem("token")
    localStorage.removeItem("user")
    return true
  }

  const token = localStorage.getItem("token")

  if (token) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/logout/`, {
        method: "POST",
        headers: {
          Authorization: `Token ${token}`,
        },
      })

      if (!response.ok) {
        console.warn("Logout API call failed, clearing local storage anyway")
      }
    } catch (error) {
      console.error("Logout error:", error)
    }
  }

  // Clear local storage regardless of API success
  localStorage.removeItem("token")
  localStorage.removeItem("user")

  return true
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

  const token = localStorage.getItem("token")

  if (!token) {
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
        Authorization: `Token ${token}`,
      },
      body: JSON.stringify(userData),
    })

    if (!response.ok) {
      throw new Error("Failed to update profile")
    }

    const updatedUser: User = await response.json()

    // Update stored user data
    localStorage.setItem("user", JSON.stringify(updatedUser))

    return updatedUser
  } catch (error) {
    console.error("Profile update error:", error)
    toast({
      title: "Update failed",
      description: error.message || "Failed to update profile",
      variant: "destructive",
    })
    return null
  }
}
