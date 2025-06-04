// Analytics service for interacting with Django backend
// Connects to the AnalyticsResult and ModelArtifact models in the MLOps section

import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "./mock-data"

// Types that match Django models
export type ResultType = "churn_pred" | "viz_image" | "timeseries_forecast" // Matches ResultType enum in schema

export interface AnalyticsResult {
  id: string
  user: string // User ID
  result_type: ResultType
  s3_key: string
  meta: any | null // JSONField
  created_at: string
}

export type ModelStage = "staging" | "production" | "archived" // Matches ModelStage enum in schema

export interface ModelArtifact {
  id: string
  name: string
  version: string
  s3_key: string
  stage: ModelStage
  metrics: any | null // JSONField
  created_by: string | null // User ID
  created_at: string
}

// For UI display
export interface BusinessChartData {
  title: string
  subtitle?: string
  chartType: "bar" | "line" | "pie"
  chartData: any[]
  series?: any[]
  insights?: string[]
}

// Mock business charts for preview environment
const mockBusinessCharts: Record<string, BusinessChartData> = {
  monthly_active_users: {
    title: "Monthly Active Users (MAU)",
    subtitle: "Last 6 months trend analysis",
    chartType: "line",
    chartData: [
      { name: "Jan", value: 4200 },
      { name: "Feb", value: 4500 },
      { name: "Mar", value: 5100 },
      { name: "Apr", value: 5400 },
      { name: "May", value: 6200 },
      { name: "Jun", value: 7100 },
    ],
    series: [{ name: "Monthly Active Users", dataKey: "value", color: "#3b82f6" }],
    insights: [
      "69% increase in MAU over the last 6 months",
      "Highest growth rate observed after the March product update",
      "Mobile users account for 64% of total active users",
      "Retention rate improved from 72% to 78% in this period",
    ],
  },
  revenue_by_product: {
    title: "Revenue by Product Category",
    subtitle: "Current quarter breakdown",
    chartType: "pie",
    chartData: [
      { name: "Enterprise", value: 45 },
      { name: "Professional", value: 30 },
      { name: "Basic", value: 15 },
      { name: "Add-ons", value: 10 },
    ],
    insights: [
      "Enterprise tier generates 45% of total revenue",
      "Add-on services show 25% growth compared to previous quarter",
      "Professional tier conversion rate increased by 12%",
      "Average revenue per user (ARPU) is $42 for Basic, $120 for Professional, and $550 for Enterprise",
    ],
  },
  customer_acquisition: {
    title: "Customer Acquisition Channels",
    subtitle: "Performance comparison",
    chartType: "bar",
    chartData: [
      { name: "Organic Search", acquisition: 320, cost: 0 },
      { name: "Paid Search", acquisition: 280, cost: 14000 },
      { name: "Social Media", acquisition: 240, cost: 9600 },
      { name: "Email", acquisition: 180, cost: 3600 },
      { name: "Referral", acquisition: 120, cost: 2400 },
    ],
    series: [{ name: "New Customers", dataKey: "acquisition", color: "#3b82f6" }],
    insights: [
      "Organic search remains the most cost-effective channel",
      "Social media campaigns show 18% higher conversion rate than last quarter",
      "Email marketing has the lowest customer acquisition cost (CAC) at $20",
      "Referral program shows strong ROI despite lower absolute numbers",
    ],
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
 * Get analytics results for the current user
 * Maps to Django AnalyticsResult model filtered by user
 */
export async function getAnalyticsResults(): Promise<AnalyticsResult[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve([
      {
        id: "result1",
        user: "user123",
        result_type: "viz_image",
        s3_key: "analytics/monthly_active_users.png",
        meta: { chartId: "monthly_active_users" },
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(), // 2 days ago
      },
      {
        id: "result2",
        user: "user123",
        result_type: "viz_image",
        s3_key: "analytics/revenue_by_product.png",
        meta: { chartId: "revenue_by_product" },
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(), // 5 days ago
      },
    ])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/mlops/results/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch analytics results")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching analytics results:", error)
    toast({
      title: "Error",
      description: "Failed to load analytics results",
      variant: "destructive",
    })
    return []
  }
}

/**
 * Get a specific analytics result
 * Retrieves a single AnalyticsResult record
 */
export async function getAnalyticsResult(resultId: string): Promise<AnalyticsResult | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const mockResults = await getAnalyticsResults()
    return Promise.resolve(mockResults.find((r) => r.id === resultId) || null)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/mlops/results/${resultId}/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch analytics result")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching analytics result:", error)
    toast({
      title: "Error",
      description: "Failed to load analytics result",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Generate a business chart from natural language query
 * This would trigger backend processing to generate a chart
 * and create an AnalyticsResult record
 */
export async function generateBusinessChart(query: string): Promise<BusinessChartData | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    // Simple keyword matching to return appropriate mock chart
    if (query.toLowerCase().includes("monthly") || query.toLowerCase().includes("active user")) {
      return Promise.resolve(mockBusinessCharts["monthly_active_users"])
    } else if (query.toLowerCase().includes("revenue") || query.toLowerCase().includes("product")) {
      return Promise.resolve(mockBusinessCharts["revenue_by_product"])
    } else if (query.toLowerCase().includes("acquisition") || query.toLowerCase().includes("channel")) {
      return Promise.resolve(mockBusinessCharts["customer_acquisition"])
    }

    // Default to monthly active users if no match
    return Promise.resolve(mockBusinessCharts["monthly_active_users"])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/mlops/generate-chart/`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ query }),
    })

    if (!response.ok) {
      throw new Error("Failed to generate business chart")
    }

    return await response.json()
  } catch (error) {
    console.error("Error generating business chart:", error)
    toast({
      title: "Error",
      description: "Failed to generate business chart",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Get available model artifacts
 * Maps to Django ModelArtifact model
 */
export async function getModelArtifacts(): Promise<ModelArtifact[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve([
      {
        id: "model1",
        name: "churn_prediction",
        version: "1.2.0",
        s3_key: "models/churn_prediction_v1.2.0.pkl",
        stage: "production",
        metrics: { accuracy: 0.87, f1: 0.84, precision: 0.82, recall: 0.86 },
        created_by: "user123",
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(), // 30 days ago
      },
      {
        id: "model2",
        name: "revenue_forecast",
        version: "0.9.1",
        s3_key: "models/revenue_forecast_v0.9.1.pkl",
        stage: "staging",
        metrics: { mape: 0.12, rmse: 0.09 },
        created_by: "user123",
        created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 7 days ago
      },
    ])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/mlops/models/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch model artifacts")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching model artifacts:", error)
    toast({
      title: "Error",
      description: "Failed to load model artifacts",
      variant: "destructive",
    })
    return []
  }
}
