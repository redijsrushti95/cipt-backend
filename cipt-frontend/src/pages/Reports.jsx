import React, { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";

const ReportsPage = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API_BASE_URL}/api/reports`, {
          method: "GET",
          credentials: "include",
          headers: { Accept: "application/json" },
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || "Failed to load reports");
        }

        const data = await response.json();
        setReports(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Error loading reports:", err);
        setError(err.message || "Error loading reports");
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const formatDate = (ms) => {
    const d = new Date(ms);
    return d.toLocaleString();
  };

  if (loading) {
    return (
      <div className="loading-state">
        <div className="loading-content">
          <div className="spinner"></div>
          <h2>Loading Your Reports...</h2>
          <p>Please wait while we fetch all your generated PDF reports.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="container" style={{ maxWidth: "900px" }}>
        <h2>Your Interview Reports</h2>

        {error && (
          <div
            style={{
              margin: "10px 0",
              padding: "10px 15px",
              background: "#ffebee",
              borderLeft: "3px solid #c62828",
              borderRadius: "6px",
              color: "#c62828",
              fontSize: "0.9rem",
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {reports.length === 0 ? (
          <p>No reports found for your account yet. Please complete an assessment to generate a report.</p>
        ) : (
          <div
            style={{
              marginTop: "15px",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: "15px",
            }}
          >
            {reports.map((report, idx) => (
              <div
                key={`${report.name}-${idx}`}
                style={{
                  background: "#ffffff",
                  borderRadius: "8px",
                  padding: "12px 14px",
                  boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  minHeight: "120px",
                }}
              >
                <div>
                  <div
                    style={{
                      fontWeight: 600,
                      marginBottom: "4px",
                      wordBreak: "break-word",
                    }}
                  >
                    {report.name}
                  </div>
                  <div
                    style={{
                      fontSize: "0.8rem",
                      color: "#555",
                    }}
                  >
                    Created: {formatDate(report.createdAt)}
                  </div>
                </div>

                <div
                  style={{
                    marginTop: "10px",
                    display: "flex",
                    gap: "8px",
                  }}
                >
                  <a
                    href={report.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "8px 10px",
                      background: "#3b82f6",
                      color: "#fff",
                      borderRadius: "6px",
                      textDecoration: "none",
                      fontSize: "0.9rem",
                      fontWeight: 600,
                    }}
                  >
                    View PDF
                  </a>
                  <a
                    href={report.url}
                    download
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "8px 10px",
                      background: "#10b981",
                      color: "#fff",
                      borderRadius: "6px",
                      textDecoration: "none",
                      fontSize: "0.9rem",
                      fontWeight: 600,
                    }}
                  >
                    Download
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportsPage;

