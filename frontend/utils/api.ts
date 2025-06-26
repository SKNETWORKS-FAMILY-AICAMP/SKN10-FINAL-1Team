export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "/api"
export function getCookie(name: string): string | null {
  if (typeof document === "undefined") return null
  const value = `; ${document.cookie}`
  const parts = value.split(`; ${name}=`)
  if (parts.length === 2) {
    return parts.pop()?.split(";").shift() || null
  }
  return null
}

export const isJsonResponse = (resp: Response) =>
  resp.headers.get("content-type")?.toLowerCase().includes("application/json")
