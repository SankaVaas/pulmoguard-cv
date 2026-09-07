import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isCheckingSession } = useAuth();

  // Avoid redirecting to /login before we've had a chance to ask the
  // backend whether an existing session cookie is still valid - otherwise
  // every page refresh would briefly bounce an authenticated user out.
  if (isCheckingSession) {
    return <div className="session-check">Checking session…</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}
