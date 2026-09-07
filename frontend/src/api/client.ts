/**
 * Typed client for the PulmoGuard backend API.
 *
 * Auth uses an httpOnly session cookie set by the backend on login - this
 * client never reads or stores the JWT itself (it's not accessible to JS
 * at all, by design; see docs/ARCHITECTURE.md §5.3). Every request that
 * needs auth passes `credentials: "include"` so the browser attaches the
 * cookie automatically.
 */

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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

export interface UserResponse {
  username: string;
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
 * The backend sets an httpOnly session cookie on success - `credentials:
 * "include"` is required for the browser to store it (and to send it back
 * on subsequent requests).
 */
export async function login(username: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams();
  body.set("username", username);
  body.set("password", password);

  const response = await fetch(`${API_BASE_URL}/api/v1/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    credentials: "include",
    body,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

/**
 * Check whether the current session cookie (if any) is still valid, and
 * return the authenticated username. Used on app load so a page refresh
 * doesn't force a re-login.
 */
export async function getCurrentUser(): Promise<UserResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
    credentials: "include",
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

/**
 * Clear the session cookie server-side and end the browser session.
 */
export async function logout(): Promise<void> {
  await fetch(`${API_BASE_URL}/api/v1/auth/logout`, {
    method: "POST",
    credentials: "include",
  });
}

/**
 * Upload a chest X-ray image for triage prediction. Relies on the session
 * cookie for auth - no token handling needed here.
 */
export async function predict(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    credentials: "include",
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
