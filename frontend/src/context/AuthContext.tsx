import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import {
  login as apiLogin,
  logout as apiLogout,
  getCurrentUser,
  ApiError,
} from "../api/client";

/**
 * Auth state now lives entirely server-side, in an httpOnly session cookie
 * the browser manages automatically - this context never touches the JWT
 * itself. On mount, it asks the backend "am I still logged in?" via
 * GET /api/v1/auth/me (which succeeds or fails based on the cookie), so a
 * page refresh doesn't force a re-login without any client-side token
 * storage. See docs/ARCHITECTURE.md §5.3 for why this replaced the earlier
 * sessionStorage-based approach.
 */

interface AuthContextValue {
  username: string | null;
  isAuthenticated: boolean;
  isCheckingSession: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [username, setUsername] = useState<string | null>(null);
  const [isCheckingSession, setIsCheckingSession] = useState(true);

  useEffect(() => {
    getCurrentUser()
      .then((user) => setUsername(user.username))
      .catch(() => setUsername(null))
      .finally(() => setIsCheckingSession(false));
  }, []);

  const login = useCallback(async (usernameInput: string, password: string) => {
    await apiLogin(usernameInput, password);
    setUsername(usernameInput);
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } catch (err) {
      // Best-effort: even if the network call fails, clear local state so
      // the UI reflects "logged out" - a stale cookie will still fail
      // server-side auth checks on its own.
      if (!(err instanceof ApiError)) {
        console.error("Logout request failed:", err);
      }
    } finally {
      setUsername(null);
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{ username, isAuthenticated: username !== null, isCheckingSession, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
