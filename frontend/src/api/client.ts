/**
 * Typed client for the PulmoGuard backend API.
 *
 * Centralizes the base URL, auth header injection, and error handling so
 * components never construct fetch() calls directly. Types here mirror
 * backend/app/schemas/*.py exactly - if the backend contract changes,
 * update both sides together.
 */

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
}

export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  normalized_entropy: number;
  abstain: boolean;
  class_probabilities: Record<string, number>;
  mc_dropout_passes: number;
  filename: string;
  message: string;
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return response.statusText;
  }
}

/**
 * Authenticate against the backend's OAuth2 password-grant endpoint.
 * Uses application/x-www-form-urlencoded, matching FastAPI's OAuth2PasswordRequestForm.
 */
export async function login(username: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams();
  body.set("username", username);
  body.set("password", password);

  const response = await fetch(`${API_BASE_URL}/api/v1/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

/**
 * Upload a chest X-ray image for triage prediction. Requires a valid bearer token.
 */
export async function predict(file: File, token: string): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export interface ReadinessResponse {
  status: string;
  model_loaded: boolean;
  environment: string;
  version: string;
}

export async function checkReadiness(): Promise<ReadinessResponse> {
  const response = await fetch(`${API_BASE_URL}/health/ready`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}
