import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { login as apiLogin, type TokenResponse } from "../api/client";

/**
 * Auth state is kept in React state (source of truth for the running app)
 * and mirrored to sessionStorage so a page refresh doesn't force re-login,
 * while still clearing automatically when the browser tab closes -
 * a reasonable middle ground for an internal clinical tool. See
 * docs/ARCHITECTURE.md for the tradeoffs vs. httpOnly cookies.
 */

const STORAGE_KEY = "pulmoguard_token";

interface AuthContextValue {
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => sessionStorage.getItem(STORAGE_KEY));

  const login = useCallback(async (username: string, password: string) => {
    const response: TokenResponse = await apiLogin(username, password);
    sessionStorage.setItem(STORAGE_KEY, response.access_token);
    setToken(response.access_token);
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem(STORAGE_KEY);
    setToken(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, isAuthenticated: token !== null, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
