import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Register from "./pages/Register.jsx";
import Verify from "./pages/Verify.jsx";
import Users from "./pages/Users.jsx";

const TITLES = {
  "/register": "Registration",
  "/verify": "Verification",
  "/users": "Users",
};

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const isHome = location.pathname === "/";

  return (
    <div className="app">
      <header className="topbar">
        {isHome ? (
          <Link to="/" className="brand" style={{ color: "inherit", textDecoration: "none" }}>
            <span className="brand-mark">H2A</span>
            Biometrics
          </Link>
        ) : (
          <>
            <button
              type="button"
              className="btn ghost small"
              onClick={() => navigate("/")}
              aria-label="Back to home"
              style={{ padding: "6px 10px" }}
            >
              &larr;
            </button>
            <h1>{TITLES[location.pathname] || "Biometrics"}</h1>
          </>
        )}
      </header>

      <main className="grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/register" element={<Register />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/users" element={<Users />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
