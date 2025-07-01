// 백엔드 API 기본 URL 설정
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || `${BACKEND_URL}/api`

// 개발 모드인지 확인
export const IS_DEVELOPMENT = process.env.NODE_ENV === "development"

console.log("🔗 Backend URL:", BACKEND_URL)
console.log("🔗 API Base URL:", API_BASE)
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
