import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../api/client";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(username, password);
      navigate("/triage");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.status === 401 ? "Incorrect username or password." : err.message);
      } else {
        setError("Unable to reach the server. Is the backend running?");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <h1>PulmoGuard</h1>
        <p className="auth-subtitle">Chest X-ray triage — sign in to continue</p>

        <label htmlFor="username">Username</label>
        <input
          id="username"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          required
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          required
        />

        {error && <div className="auth-error">{error}</div>}

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
