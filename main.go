package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
	"golang.org/x/crypto/pbkdf2"

	_ "modernc.org/sqlite"
)

// Provider represents an external AI model API provider (e.g., OpenAI, Anthropic).
type Provider struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	Endpoint    string `json:"endpoint"`
	APIKey      string `json:"api_key"`
	ExtraConfig string `json:"extra_config"`
	Models      string `json:"models"`
	TokenUsage  int    `json:"token_usage"`
}

// Source represents external data sources available for agents to retrieve context.
type Source struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Type     string `json:"type"`
	APIKey   string `json:"api_key"`
	FilePath string `json:"file_path"`
}

// Output defines instruction templates for standardizing agent responses.
type Output struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Instruction string `json:"instruction"`
	Type        string `json:"type"` // "Command" (default) or "Webhook"
	CommandTpl  string `json:"command_tpl"`
	WebhookURL  string `json:"webhook_url"`
}

// Skill represents a dynamic capability executable by the agent (Bash, Go, API, etc.).
type Skill struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	Instruction string `json:"instruction"`
	Content     string `json:"content"`
	APIMethod   string `json:"api_method"`
	APIURL      string `json:"api_url"`
	APIHeaders  string `json:"api_headers"`
	APIBody     string `json:"api_body"`
	UseDocker   bool   `json:"use_docker"`
}

// RAG represents a retrieval-augmented generation vector database configuration.
type RAG struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	TableName   string `json:"table_name"`
}

// Preference stores persistent memory and key-value attributes for agents.
type Preference struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	TableName   string `json:"table_name"`
}

// Agent represents an AI entity combining a specific model, system prompt, and capabilities.
type Agent struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	ProviderID       string  `json:"provider_id"`
	Model            string  `json:"model"`
	Sources          string  `json:"sources"`
	Skills           string  `json:"skills"`
	Outputs          string  `json:"outputs"`
	MCPs             string  `json:"mcps"`
	MCPTools         string  `json:"mcp_tools"` // JSON array of {mcpId, tool} assigned MCP commands
	Projects         string  `json:"projects"`
	SystemPrompt     string  `json:"system_prompt"`
	TokenUsage       int     `json:"token_usage"`
	InputMethods     string  `json:"input_methods"`
	OutputMethods    string  `json:"output_methods"`
	UserPromptPrefix string  `json:"user_prompt_prefix"`
	Temperature      float64 `json:"temperature"`
	MaxTokens        int     `json:"max_tokens"`
	TopP             float64 `json:"top_p"`
	PrivacyEnabled   bool    `json:"privacy_enabled"`
	CanCreateSkills  bool    `json:"can_create_skills"`
	StreamEnabled    bool    `json:"stream_enabled"`
}

// Task defines an automated background job monitored and executed by the system.
type Task struct {
	ID      string `json:"id"`
	AgentID string `json:"agent_id"`
	Regex   string `json:"regex"`
	Command string `json:"command"`
	Repeat  bool   `json:"repeat"`
	Active  bool   `json:"active"`
}

// Project represents a visual routing pipeline linking multiple nodes and agents.
type Project struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	FlowJSON    string `json:"flow_json"`
	Tags        string `json:"tags"`
	IsActive    bool   `json:"is_active"`
}

// MCPServer defines a Model Context Protocol endpoint for external tool aggregation.
type MCPServer struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	URL     string `json:"url"`
	Command string `json:"command"`
	Tools   string `json:"tools"` // JSON array of resolved tool/command names
}

// ChatMessage represents a single message exchange within an agent's history.
type ChatMessage struct {
	ID        int    `json:"id"`
	AgentID   string `json:"agent_id"`
	SessionID string `json:"session_id"`
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
}

// ChatSession represents a conversation session within an agent.
type ChatSession struct {
	ID        string `json:"id"`
	AgentID   string `json:"agent_id"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

// App encapsulates the global database connections for the framework.
type App struct {
	ConfigDB *sql.DB
	MemoryDB *sql.DB
}

// APIKey manages external access credentials and granular permission scopes.
type APIKey struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Token       string `json:"token"`
	Permissions string `json:"permissions"`
	RateLimit   int    `json:"rate_limit"` // max requests/minute, 0 = unlimited
}

// DBTask encapsulates a database operation to be processed sequentially.
type DBTask struct {
	DB    *sql.DB
	Query string
	Args  []interface{}
}

// Thread-safe channels and mutexes for resolving concurrent SQLite locks and requests.
// contextKeyAPIKeyPerms is the context key under which authMiddleware stores the
// raw permissions JSON string for an authenticated external API key request.
type contextKeyAPIKeyPerms struct{}

type contextKeyReauthVerified struct{}

type SecurityEvent struct {
	Timestamp string `json:"timestamp"`
	Action    string `json:"action"`
	Path      string `json:"path"`
	Actor     string `json:"actor"`
	Detail    string `json:"detail"`
}

// bcryptHash returns a bcrypt hash of the given password at the default cost.
func bcryptHash(password string) (string, error) {
	b, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// bcryptCompare returns true if the plaintext password matches the stored bcrypt hash.
func bcryptCompare(hash, password string) bool {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

// generateRandomPassword generates a cryptographically random 24-character password.
func generateRandomPassword() (string, error) {
	const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
	b := make([]byte, 24)
	for i := range b {
		n, err := rand.Int(rand.Reader, big.NewInt(int64(len(chars))))
		if err != nil {
			return "", err
		}
		b[i] = chars[n.Int64()]
	}
	return string(b), nil
}

// safeTableNameRe matches only identifiers that are safe to interpolate into SQL
// table/column names — alphanumerics and underscores only.
var safeTableNameRe = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// sanitizeTableName validates that a table name loaded from the DB is safe to
// interpolate directly into a SQL query (no SQLi via a poisoned DB value).
// Returns the name unchanged if valid, or an error otherwise.
func sanitizeTableName(name string) (string, error) {
	if name == "" || !safeTableNameRe.MatchString(name) {
		return "", fmt.Errorf("unsafe table name: %q", name)
	}
	return name, nil
}

// allowedUploadExts is the explicit allowlist of file extensions accepted by
// handleFileUpload. Extensions not in this map are rejected.
var allowedUploadExts = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".gif": true,
	".pdf": true, ".txt": true, ".md": true, ".csv": true,
	".json": true, ".yaml": true, ".yml": true, ".html": true,
	".mp3": true, ".wav": true, ".mp4": true,
}

// isPrivateHost returns true if the hostname/IP belongs to a loopback or
// private-range address that should not be reachable via outbound HTTP calls
// triggered by user-supplied URLs (SSRF prevention).
func isPrivateHost(rawURL string) bool {
	parsed, err := url.ParseRequestURI(rawURL)
	if err != nil {
		return true // treat unparseable as unsafe
	}
	h := strings.ToLower(parsed.Hostname())

	// Check DNS A/AAAA records to prevent DNS rebinding attacks.
	if ips, err := net.LookupIP(h); err == nil && len(ips) > 0 {
		for _, ip := range ips {
			if isPrivateIP(ip.String()) {
				return true
			}
		}
		// Use first IPv4 for prefix matching below.
		for _, ip := range ips {
			if v4 := ip.To4(); v4 != nil {
				h = v4.String()
				break
			}
		}
	}

	for _, blocked := range []string{
		"localhost", "127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.",
		"172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
		"172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
		"0.0.0.0", "169.254.", "100.64.", "198.18.", "198.19.",
	} {
		if strings.HasPrefix(h, blocked) || h == strings.TrimSuffix(blocked, ".") {
			return true
		}
	}
	return false
}

// isPrivateIP checks if an IP address is in private, loopback, or reserved ranges.
func isPrivateIP(ipStr string) bool {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return false
	}
	// Check loopback.
	if ip.IsLoopback() {
		return true
	}
	// Check private ranges (10/8, 172.16/12, 192.168/16).
	private := net.ParseIP("10.0.0.0")
	if ip.Equal(private) || ip.To4() != nil && ip[0] == 10 {
		return true
	}
	if ip.To4() != nil {
		b := ip.To4()
		// 172.16.0.0 - 172.31.255.255
		if b[0] == 172 && b[1] >= 16 && b[1] <= 31 {
			return true
		}
		// 192.168.0.0 - 192.168.255.255
		if b[0] == 192 && b[1] == 168 {
			return true
		}
	}
	// Check IPv6 unique local (fc00::/7) and link-local (fe80::/10).
	if ip.IsGlobalUnicast() == false {
		return true
	}
	return false
}

// safeHTTPClient returns an http.Client that resolves DNS once and dials the
// resolved IP directly, preventing DNS rebinding (TOCTOU) attacks.
func safeHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, port, err := net.SplitHostPort(addr)
				if err != nil {
					return nil, err
				}
				// Resolve DNS once.
				ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
				if err != nil {
					return nil, err
				}
				if len(ips) == 0 {
					return nil, fmt.Errorf("no IPs found for %s", host)
				}
				// Try each resolved IP.
				var lastErr error
				for _, ipAddr := range ips {
				    if isPrivateIP(ipAddr.IP.String()) {
				        return nil, fmt.Errorf("security block: resolved to private IP %s", ipAddr.IP.String())
				    }
				    
				    conn, err := net.DialTimeout(network, net.JoinHostPort(ipAddr.IP.String(), port), timeout)
				    if err == nil {
				        return conn, nil
 				   }
				    lastErr = err
				}
				return nil, lastErr
			},
		},
	}
}

// securityHeadersMiddleware adds standard defensive HTTP response headers to
// every response: X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
// Content-Security-Policy (restrictive default), and Permissions-Policy.
// When cors_enabled is 'true' in the settings table it also adds
// Access-Control-Allow-Origin: * and handles OPTIONS preflight requests.
func (a *App) securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read CORS setting live so changes take effect without restart.
		var corsVal string
		a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='cors_enabled'").Scan(&corsVal)
		corsEnabled := corsVal == "true"

		if corsEnabled {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept")
			// Short-circuit OPTIONS preflight — no further processing needed.
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}

		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "SAMEORIGIN")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		w.Header().Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
		// Allow inline scripts/styles only for the single-page UI served at "/".
		// API and asset paths get the strictest policy.
		if r.URL.Path == "/" || r.URL.Path == "/index.html" || r.URL.Path == "/api/add" {
			w.Header().Set("Content-Security-Policy",
				"default-src 'self'; "+
					"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.tailwindcss.com https://unpkg.com; "+
					"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://unpkg.com https://cdn.tailwindcss.com; "+
					"font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net https://unpkg.com; "+
					"img-src 'self' data: blob: https:; "+
					"connect-src 'self' https:; "+
					"frame-ancestors 'self'")
		} else {
			w.Header().Set("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
		}
		next.ServeHTTP(w, r)
	})
}

// rateBucket tracks per-API-key request counts for rate limiting.
type rateBucket struct {
	count     int
	windowEnd time.Time
}

var (
	rateBuckets   = make(map[string]*rateBucket)
	rateBucketsMu sync.Mutex
	securityLogMu sync.Mutex
	securityLogs  []SecurityEvent
)

// CurrentVersion is the running instance version — compared against GitHub releases.
const CurrentVersion = "v1.4.1-beta.1"

// updateInfo caches the latest release info from GitHub.
type updateInfo struct {
	LatestVersion string `json:"latest_version"`
	DownloadURL   string `json:"download_url"`
	ReleaseURL    string `json:"release_url"`
	HasUpdate     bool   `json:"has_update"`
	CheckedAt     string `json:"checked_at"`
}

var (
	latestUpdate   updateInfo
	latestUpdateMu sync.Mutex
)

// checkForUpdate fetches the latest release tag from GitHub and caches the result.
func checkForUpdate() {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", "https://api.github.com/repos/synasties/zuver/releases/latest", nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	resp, err := client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	var release struct {
		TagName string `json:"tag_name"`
		HTMLURL string `json:"html_url"`
		Assets  []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
			Name               string `json:"name"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return
	}
	if release.TagName == "" {
		return
	}
	downloadURL := release.HTMLURL
	for _, a := range release.Assets {
		if strings.Contains(a.Name, "linux") && strings.Contains(a.Name, "amd64") {
			downloadURL = a.BrowserDownloadURL
			break
		}
	}
	latestUpdateMu.Lock()
	latestUpdate = updateInfo{
		LatestVersion: release.TagName,
		DownloadURL:   downloadURL,
		ReleaseURL:    release.HTMLURL,
		HasUpdate:     release.TagName != CurrentVersion,
		CheckedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	latestUpdateMu.Unlock()
}

var (
	dbWriteQueue  = make(chan DBTask, 10000)
	adminToken    string
	adminTokenExp time.Time
	adminTokenMu  sync.RWMutex
	loginAttempts = make(map[string]int)
	lockoutTime   = make(map[string]time.Time)
	authMu        sync.Mutex
	dbWriteMu     sync.Mutex
)

const adminTokenTTL = 24 * time.Hour

// CSRF token store: token → timestamp
var (
	csrfTokens   = make(map[string]time.Time)
	csrfTokensMu sync.Mutex
)

// generateCSRFToken creates a single-use CSRF token valid for 10 minutes.
func generateCSRFToken() string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		log.Printf("[Security] CSRF token generation failed: %v", err)
		return ""
	}
	token := hex.EncodeToString(b)
	csrfTokensMu.Lock()
	csrfTokens[token] = time.Now().Add(10 * time.Minute)
	csrfTokensMu.Unlock()
	return token
}

// validateCSRFToken checks and consumes a CSRF token.
// Returns true if valid and removes it from the store (one-time use).
func validateCSRFToken(token string) bool {
	csrfTokensMu.Lock()
	defer csrfTokensMu.Unlock()
	expiresAt, ok := csrfTokens[token]
	if !ok {
		return false
	}
	if time.Now().After(expiresAt) {
		delete(csrfTokens, token)
		return false
	}
	delete(csrfTokens, token)
	return true
}

// cleanupCSRFTokens removes expired tokens periodically.
func cleanupCSRFTokens() {
	csrfTokensMu.Lock()
	defer csrfTokensMu.Unlock()
	now := time.Now()
	for token, expiresAt := range csrfTokens {
		if now.After(expiresAt) {
			delete(csrfTokens, token)
		}
	}
}

// cleanupAuthState removes stale login attempts and lockout entries.
func cleanupAuthState() {
	authMu.Lock()
	defer authMu.Unlock()
	now := time.Now()
	for ip, expiresAt := range lockoutTime {
		if now.After(expiresAt.Add(5 * time.Minute)) {
			delete(lockoutTime, ip)
			delete(loginAttempts, ip)
		}
	}
}

// SyncDBExec executes a synchronous database write protected by a global mutex.
func SyncDBExec(db *sql.DB, query string, args ...interface{}) (sql.Result, error) {
	dbWriteMu.Lock()
	defer dbWriteMu.Unlock()
	return db.Exec(query, args...)
}

// permissionScopeForPath maps an HTTP route to (resource, itemID, action).
// Actions: "read" (GET), "write" (POST/PUT), "delete" (DELETE), "execute" (POST on run/chat).
// itemID is non-empty only for routes that address a specific item (e.g. agents/{id}/run).
// Returns ("", "", "") if the route is allowed for any valid key.
// Returns ("admin_only", "", "") to block all external API keys unconditionally.
func permissionScopeForPath(method, path string) (resource string, itemID string, action string) {
	// Strip /api/v1/ prefix and split into up to 4 segments.
	trimmed := strings.TrimPrefix(path, "/api/v1/")
	parts := strings.SplitN(trimmed, "/", 4)
	seg := parts[0]

	// Normalize HTTP method to semantic action for permission checks.
	toAction := func(m string) string {
		switch m {
		case "GET":
			return "read"
		case "POST":
			return "write"
		case "PUT", "PATCH":
			return "write"
		case "DELETE":
			return "delete"
		default:
			return "read"
		}
	}

	switch seg {
	case "agents":
		// agents/{id}/run — execute on a specific agent.
		if len(parts) >= 3 && parts[2] == "run" {
			return "agents", parts[1], "execute"
		}
		// agents/{id} — scoped read/write/delete on a specific agent.
		if len(parts) >= 2 && parts[1] != "" {
			return "agents", parts[1], toAction(method)
		}
		return "agents", "", toAction(method)
	case "chat":
		// Chat: agent_id comes from the request body; checked post-parse in handleChat.
		// We return "*" as a sentinel — middleware will pass, handleChat enforces per-agent.
		return "agents", "*", "execute"
	case "history":
		return "agents", "", "read"
	case "skills":
		return "skills", "", toAction(method)
	case "sources":
		if len(parts) >= 2 && parts[1] == "update" {
			return "sources", "", "write"
		}
		return "sources", "", toAction(method)
	case "outputs":
		return "outputs", "", toAction(method)
	case "mcp":
		return "mcps", "", toAction(method)
	case "projects":
		// projects/{id}/run — execute on a specific application.
		if len(parts) >= 3 && parts[2] == "run" {
			return "projects", parts[1], "execute"
		}
		// projects/{id}/trigger — webhook trigger (execute permission).
		if len(parts) >= 3 && parts[2] == "trigger" {
			return "projects", parts[1], "execute"
		}
		if len(parts) >= 2 && parts[1] != "" {
			return "projects", parts[1], toAction(method)
		}
		return "projects", "", toAction(method)
	case "preferences":
		return "preferences", "", toAction(method)
	case "rags":
		return "rags", "", toAction(method)
	case "providers", "apikeys", "settings", "wipe-database", "import":
		// Admin-only: never accessible via external API keys.
		return "admin_only", "", ""
	default:
		// stats, sysinfo, analytics, upload — permitted for any valid key.
		return "", "", ""
	}
}

func (a *App) appendSecurityEvent(action, path, actor, detail string) {
	ts := time.Now().UTC().Format(time.RFC3339)
	// Write to DB for persistence.
	AsyncDBExec(a.ConfigDB, "INSERT INTO security_logs (timestamp, action, path, actor, detail) VALUES (?, ?, ?, ?, ?)", ts, action, path, actor, detail)
	// Prune DB to keep only the latest 200 records.
	AsyncDBExec(a.ConfigDB, "DELETE FROM security_logs WHERE id NOT IN (SELECT id FROM security_logs ORDER BY id DESC LIMIT 200)")

	// Keep in-memory buffer for fast API reads (capped at 200).
	securityLogMu.Lock()
	defer securityLogMu.Unlock()
	securityLogs = append(securityLogs, SecurityEvent{
		Timestamp: ts,
		Action:    action,
		Path:      path,
		Actor:     actor,
		Detail:    detail,
	})
	if len(securityLogs) > 200 {
		securityLogs = append([]SecurityEvent(nil), securityLogs[len(securityLogs)-200:]...)
	}
}

func requiresRecentReauth(r *http.Request) bool {
	if r.Method == http.MethodGet || r.Method == http.MethodOptions {
		return false
	}
	for _, prefix := range []string{
		"/api/v1/providers",
		"/api/v1/sources",
		"/api/v1/apikeys",
		"/api/v1/wipe-database",
	} {
		if strings.HasPrefix(r.URL.Path, prefix) {
			return true
		}
	}
	return false
}

func (a *App) verifyAdminPassword(password string) bool {
	var dbHash string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='admin_password'").Scan(&dbHash)
	if dbHash == "" {
		return false
	}
	// Only accept bcrypt hashes. Legacy SHA-256 hashes are auto-upgraded
	// on first successful login via handleLogin — we never verify them here.
	if !strings.HasPrefix(dbHash, "$2") {
		return false
	}
	return bcryptCompare(dbHash, password)
}

// authMiddleware enforces API security and authenticates administrative or API requests.
// For external API keys, it also validates granular permission scopes.
func (a *App) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/v1/") || r.URL.Path == "/api/v1/login" {
			next.ServeHTTP(w, r)
			return
		}

		// Accept token from Authorization header OR ?token= query param (used for download links).
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if token == "" {
			token = r.URL.Query().Get("token")
		}

		// Admin token always passes without scope restriction.
		adminTokenMu.RLock()
		currentAdminToken := adminToken
		tokenExp := adminTokenExp
		adminTokenMu.RUnlock()
		if currentAdminToken != "" && token == currentAdminToken && time.Now().Before(tokenExp) {
			if requiresRecentReauth(r) {
				reauthOK := false
				if strings.EqualFold(strings.TrimSpace(r.Header.Get("X-Reauth-Verified")), "true") {
					reauthOK = true
				} else if pw := r.Header.Get("X-Reauth-Password"); pw != "" {
					reauthOK = a.verifyAdminPassword(pw)
				}
				if !reauthOK {
					a.appendSecurityEvent("reauth_required", r.URL.Path, "admin", "sensitive action blocked until reauthentication")
					w.Header().Set("Content-Type", "application/json")
					http.Error(w, `{"error": "Reauthentication required"}`, http.StatusUnauthorized)
					return
				}
				a.appendSecurityEvent("reauth_success", r.URL.Path, "admin", "recent reauthentication accepted")
				r = r.WithContext(context.WithValue(r.Context(), contextKeyReauthVerified{}, true))
			}
			next.ServeHTTP(w, r)
			return
		}

		// Validate external API key and load its permission document + rate limit.
		var permJSON string
		var rateLimit int
		tokenHash := fmt.Sprintf("%x", sha256.Sum256([]byte(token)))
		err := a.ConfigDB.QueryRow("SELECT permissions, COALESCE(rate_limit,0) FROM api_keys WHERE token_hash=?", tokenHash).Scan(&permJSON, &rateLimit)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Unauthorized"}`, http.StatusUnauthorized)
			return
		}

		// Enforce per-minute rate limit when configured (0 = unlimited).
		if rateLimit > 0 {
			rateBucketsMu.Lock()
			b, exists := rateBuckets[token]
			now := time.Now()
			if !exists || now.After(b.windowEnd) {
				rateBuckets[token] = &rateBucket{count: 1, windowEnd: now.Add(time.Minute)}
			} else {
				b.count++
				if b.count > rateLimit {
					rateBucketsMu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", rateLimit))
					http.Error(w, `{"error": "Rate limit exceeded"}`, http.StatusTooManyRequests)
					return
				}
			}
			rateBucketsMu.Unlock()
		}

		// Attach the raw permissions JSON to the context so downstream handlers
		// (e.g. handleChat) can enforce per-resource checks after parsing the body.
		ctx := context.WithValue(r.Context(), contextKeyAPIKeyPerms{}, permJSON)
		r = r.WithContext(ctx)

		// Determine required scope for this endpoint.
		resource, itemID, action := permissionScopeForPath(r.Method, r.URL.Path)

		// Admin-only routes are never accessible via API keys.
		if resource == "admin_only" {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Forbidden: admin-only endpoint"}`, http.StatusForbidden)
			return
		}

		// Routes with no specific scope requirement pass through for any valid key.
		if resource == "" {
			next.ServeHTTP(w, r)
			return
		}

		// Parse permissions document.
		var perms map[string]interface{}
		if json.Unmarshal([]byte(permJSON), &perms) != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Forbidden: malformed permissions"}`, http.StatusForbidden)
			return
		}

		// checkActionMap returns true if actionMap ({"read":true,...}) grants action.
		checkActionMap := func(actionMap map[string]interface{}) bool {
			if b, ok := actionMap[action].(bool); ok {
				return b
			}
			return false
		}

		allowed := false
		if val, ok := perms[resource]; ok {
			switch v := val.(type) {
			case map[string]interface{}:
				switch resource {
				case "preferences":
					// preferences: {pref_id: {"read":true,"write":true}, ...}
					// Any matching pref entry with the right action grants access.
					for _, entryRaw := range v {
						if entry, ok2 := entryRaw.(map[string]interface{}); ok2 {
							if checkActionMap(entry) {
								allowed = true
							}
						}
					}

				case "agents", "projects":
					// Per-item permissions: {"*": {"execute":true}, "agent_id": {"execute":true}}
					// Check wildcard "*" first, then the specific itemID.
					if wildcardRaw, hasWild := v["*"]; hasWild {
						if wm, ok2 := wildcardRaw.(map[string]interface{}); ok2 && checkActionMap(wm) {
							allowed = true
						}
					}
					// For specific item operations, also check the exact item entry.
					if !allowed && itemID != "" && itemID != "*" {
						if itemRaw, hasItem := v[itemID]; hasItem {
							if im, ok2 := itemRaw.(map[string]interface{}); ok2 && checkActionMap(im) {
								allowed = true
							}
						}
					}
					// itemID == "*" means the body carries the real agent ID (e.g. /api/v1/chat).
					// The middleware can't know the agent ID yet — pass through if *any*
					// per-agent entry grants the action. handleChat enforces the specific check.
					if !allowed && itemID == "*" {
						for _, entryRaw := range v {
							if em, ok2 := entryRaw.(map[string]interface{}); ok2 && checkActionMap(em) {
								allowed = true
								break
							}
						}
					}
					// If itemID is empty (list endpoint), wildcard already checked above.
					// Grant access to list endpoint if *any* item-level entry has the action.
					if !allowed && itemID == "" {
						for _, entryRaw := range v {
							if em, ok2 := entryRaw.(map[string]interface{}); ok2 && checkActionMap(em) {
								allowed = true
								break
							}
						}
					}

				default:
					// Flat action map: {"read":true,"write":true,...}
					if checkActionMap(v) {
						allowed = true
					}
				}

			case []interface{}:
				// Legacy list format: presence grants access.
				if len(v) > 0 {
					allowed = true
				}
			case bool:
				allowed = v
			}
		}

		if !allowed {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Forbidden: insufficient permissions for action '`+action+`' on '`+resource+`'"}`, http.StatusForbidden)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// getClientIP extracts the real client IP from the request, considering reverse proxy headers.
// Only trusts X-Forwarded-For/X-Real-IP when the immediate peer is a known proxy (loopback/private).
func getClientIP(r *http.Request) string {
	// Extract immediate peer IP from RemoteAddr.
	peer := r.RemoteAddr
	if h, _, err := net.SplitHostPort(peer); err == nil {
		peer = h
	}
	peer = strings.Trim(peer, "[]")

	// Only trust proxy headers if the immediate peer is loopback or private (i.e., behind a proxy).
	if peer == "127.0.0.1" || peer == "::1" || peer == "localhost" {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			// X-Forwarded-For may contain a chain: "client, proxy1, proxy2"
			// Take the first (leftmost) IP as the real client.
			parts := strings.Split(xff, ",")
			if len(parts) > 0 {
				return strings.TrimSpace(parts[0])
			}
		}
		if xri := r.Header.Get("X-Real-IP"); xri != "" {
			return strings.TrimSpace(xri)
		}
	}
	return peer
}

// isLoopbackRemoteAddr reports whether the request appears to originate from the same host.
// This is used to restrict first-time admin password initialization to local access only.
func isLoopbackRemoteAddr(remoteAddr string) bool {
	host := strings.TrimSpace(remoteAddr)
	if h, _, err := net.SplitHostPort(remoteAddr); err == nil {
		host = h
	}
	host = strings.Trim(host, "[]")
	if host == "" {
		return false
	}
	if host == "::1" || host == "127.0.0.1" || strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// handleLogin manages system authentication and rate-limiting against brute force attacks.
func (a *App) handleLogin(w http.ResponseWriter, r *http.Request) {
	// Security: limit body size to prevent DoS.
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req struct {
		Password string `json:"password"`
		TOTPCode string `json:"totp_code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "Invalid request"}`, http.StatusBadRequest)
		return
	}

	// Security: reject obviously-empty passwords.
	if strings.TrimSpace(req.Password) == "" {
		http.Error(w, `{"error": "Password required"}`, http.StatusBadRequest)
		return
	}

	// Get real client IP (handles reverse proxy headers).
	ip := getClientIP(r)

	authMu.Lock()
	defer authMu.Unlock()

	if time.Now().Before(lockoutTime[ip]) {
		http.Error(w, `{"error": "Too many failed attempts. Try again in 1 minute."}`, http.StatusTooManyRequests)
		return
	}

	var dbHash string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='admin_password'").Scan(&dbHash)

	// First boot: generate a random password automatically.
	// Security: only allow bootstrap from a local loopback request so a remotely
	// exposed fresh instance cannot be claimed by the first network caller.
	if dbHash == "" {
		if !isLoopbackRemoteAddr(getClientIP(r)) {
			a.appendSecurityEvent("remote_init_denied", r.URL.Path, ip, "remote first-time admin initialization blocked")
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Admin password has not been initialized yet. Complete first-time setup from localhost."}`, http.StatusForbidden)
			return
		}
		// Generate random password for first boot.
		generatedPassword, genErr := generateRandomPassword()
		if genErr != nil {
			http.Error(w, `{"error": "Server error"}`, http.StatusInternalServerError)
			return
		}
		newHash, hashErr := bcryptHash(generatedPassword)
		if hashErr != nil {
			http.Error(w, `{"error": "Server error"}`, http.StatusInternalServerError)
			return
		}
		SyncDBExec(a.ConfigDB, "INSERT INTO settings (key, value) VALUES ('admin_password', ?)", newHash)
		dbHash = newHash
		// Return the generated password so the user can see it.
		loginAttempts[ip] = 0
		b := make([]byte, 32)
		rand.Read(b)
		adminTokenMu.Lock()
		adminToken = "tok_" + hex.EncodeToString(b)
		adminTokenExp = time.Now().Add(adminTokenTTL)
		newAdminToken := adminToken
		adminTokenMu.Unlock()
		a.appendSecurityEvent("first_login", r.URL.Path, ip, "first-time admin login - auto-generated password")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"token":         newAdminToken,
			"password":      generatedPassword,
			"password_set":  "true",
		})
		return
	}

	// Migrate legacy plain SHA-256 hashes on first successful login.
	// A bcrypt hash always starts with "$2"; SHA-256 hex is 64 hex chars.
	if !strings.HasPrefix(dbHash, "$2") {
		// Legacy SHA-256 hash detected — log security event and enforce stricter lockout.
		a.appendSecurityEvent("legacy_password", r.URL.Path, ip, "legacy SHA-256 admin hash detected — upgrade recommended")
		// Verify against old SHA-256 hash for backward compat.
		oldHash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Password)))
		if oldHash != dbHash {
			loginAttempts[ip]++
			// Stricter lockout for legacy SHA-256: 3 attempts instead of 5.
			if loginAttempts[ip] >= 3 {
				lockoutTime[ip] = time.Now().Add(5 * time.Minute)
			}
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Invalid password. Access Denied."}`, http.StatusUnauthorized)
			return
		}
		// Upgrade to bcrypt immediately.
		newHash, _ := bcryptHash(req.Password)
		SyncDBExec(a.ConfigDB, "UPDATE settings SET value=? WHERE key='admin_password'", newHash)
		dbHash = newHash
		a.appendSecurityEvent("password_upgrade", r.URL.Path, ip, "legacy SHA-256 hash upgraded to bcrypt")
	}

	if bcryptCompare(dbHash, req.Password) {
		// Check if TOTP 2FA is enabled — require TOTP code if so.
		var totpEnabled string
		a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='totp_enabled'").Scan(&totpEnabled)
		if totpEnabled == "true" {
			if req.TOTPCode == "" {
				http.Error(w, `{"error": "TOTP code required", "totp_required": true}`, http.StatusUnauthorized)
				return
			}
			var totpSecret string
			a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='totp_secret'").Scan(&totpSecret)
			if !verifyTOTP(totpSecret, req.TOTPCode) {
				a.appendSecurityEvent("totp_verify_fail", r.URL.Path, ip, "invalid TOTP code on login")
				http.Error(w, `{"error": "Invalid TOTP code"}`, http.StatusUnauthorized)
				return
			}
		}

		loginAttempts[ip] = 0
		// Use crypto/rand for an unpredictable session token.
		b := make([]byte, 32)
		if _, err := rand.Read(b); err != nil {
			log.Printf("[Security] Admin token generation failed: %v", err)
			http.Error(w, `{"error": "Server error"}`, http.StatusInternalServerError)
			return
		}
		adminTokenMu.Lock()
		adminToken = "tok_" + hex.EncodeToString(b)
		adminTokenExp = time.Now().Add(adminTokenTTL)
		currentToken := adminToken
		adminTokenMu.Unlock()
		a.appendSecurityEvent("login_success", r.URL.Path, ip, "admin login succeeded")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"token": currentToken})
	} else {
		loginAttempts[ip]++
		if loginAttempts[ip] >= 5 {
			lockoutTime[ip] = time.Now().Add(1 * time.Minute)
		}
		a.appendSecurityEvent("login_failure", r.URL.Path, ip, "invalid admin password")
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error": "Invalid password. Access Denied."}`, http.StatusUnauthorized)
	}
}

// startDBWorker initializes the background processor for resolving asynchronous database operations.
func startDBWorker() {
	for task := range dbWriteQueue {
		_, err := task.DB.Exec(task.Query, task.Args...)
		if err != nil {
			log.Printf("[DB Queue Error] %v | Query: %s", err, task.Query)
		}
	}
}

// AsyncDBExec appends a query to the database writing queue to prevent file locking.
func AsyncDBExec(db *sql.DB, query string, args ...interface{}) {
	select {
	case dbWriteQueue <- DBTask{DB: db, Query: query, Args: args}:
	default:
		log.Printf("[DB Queue Full] Dropping write: %s", query)
	}
}

var placeholderRe = regexp.MustCompile(`<([^>]+)>`)

// cosineSimilarity calculates the mathematical proximity between two vector embeddings.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// getEmbedding requests vector generation from the primary configured OpenAI provider.
func getEmbedding(text string, db *sql.DB) []float32 {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	// Check embedding cache first.
	var cacheEnabled string
	db.QueryRow("SELECT value FROM settings WHERE key='rag_embedding_cache'").Scan(&cacheEnabled)
	cacheKey := fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
	if cacheEnabled == "true" {
		var cachedEmbedding string
		if db.QueryRow("SELECT embedding FROM embedding_cache WHERE hash=? AND model='default'", cacheKey).Scan(&cachedEmbedding) == nil {
			var vec []float32
			if json.Unmarshal([]byte(cachedEmbedding), &vec) == nil && len(vec) > 0 {
				return vec
			}
		}
	}

	// Check if local embedding is enabled.
	var localEnabled, localURL string
	db.QueryRow("SELECT value FROM settings WHERE key='rag_local_embedding'").Scan(&localEnabled)
	db.QueryRow("SELECT value FROM settings WHERE key='rag_local_embedding_url'").Scan(&localURL)

	var vec []float32
	if localEnabled == "true" && localURL != "" {
		vec = getEmbeddingLocal(text, localURL)
	} else {
		vec = getEmbeddingOpenAI(text, db)
	}

	// Store in cache.
	if cacheEnabled == "true" && len(vec) > 0 {
		if vBytes, err := json.Marshal(vec); err == nil {
			db.Exec("INSERT OR REPLACE INTO embedding_cache (hash, embedding, model) VALUES (?, ?, 'default')", cacheKey, string(vBytes))
		}
	}

	return vec
}

// generateImage calls an OpenAI-compatible image generation API (DALL-E, etc.)
// and returns a base64-encoded image or a URL.
func generateImage(prompt, model, apiKey, endpoint string) (string, string, error) {
	if model == "" {
		model = "dall-e-3"
	}
	apiURL := strings.TrimRight(endpoint, "/") + "/images/generations"
	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"n":      1,
		"size":   "1024x1024",
	})
	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 120 * time.Second}).Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if resp.StatusCode != 200 {
		return "", "", fmt.Errorf("image generation failed: HTTP %d", resp.StatusCode)
	}
	var result struct {
		Data []struct {
			URL string `json:"url"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", "", err
	}
	if len(result.Data) == 0 {
		return "", "", fmt.Errorf("no image returned")
	}
	return result.Data[0].URL, "image/png", nil
}

func getEmbeddingLocal(text, endpoint string) []float32 {
	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":  "nomic-embed-text",
		"prompt": text,
	})
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Post(endpoint, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var res struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil || len(res.Embedding) == 0 {
		return nil
	}
	return res.Embedding
}

func getEmbeddingOpenAI(text string, db *sql.DB) []float32 {
	var apiKey, endpoint string
	err := db.QueryRow("SELECT api_key, endpoint FROM providers WHERE type='OpenAI' LIMIT 1").Scan(&apiKey, &endpoint)
	if err != nil || apiKey == "" {
		return nil
	}
	apiKey = decryptSecret(apiKey)
	apiURL := strings.TrimRight(endpoint, "/") + "/v1/embeddings"
	reqBody, _ := json.Marshal(map[string]interface{}{
		"input": text,
		"model": "text-embedding-3-small",
	})
	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := safeHTTPClient(10 * time.Second).Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var res struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil || len(res.Data) == 0 {
		return nil
	}
	return res.Data[0].Embedding
}

// extractPlaceholders parses input parameters using regular expressions.
func extractPlaceholders(text string) []string {
	matches := placeholderRe.FindAllString(text, -1)
	seen := make(map[string]bool)
	var result []string
	for _, m := range matches {
		if !seen[m] {
			seen[m] = true
			result = append(result, m)
		}
	}
	return result
}

// estimateTokens provides a rudimentary token calculation fallback.
func estimateTokens(text string) int { return len(text) / 3 }

// chunkText splits text into overlapping chunks of approximately chunkSize characters.
func chunkText(text string, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}
	runes := []rune(text)
	var chunks []string
	for i := 0; i < len(runes); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}

var (
	paginationStore  = make(map[string][]string)
	paginationStep   = make(map[string]int)
	paginationMu     sync.Mutex
	skillEditorMu    sync.Mutex
	skillEditorStore = make(map[string]map[string]string)
)

// triggerPagination breaks large responses into readable chunks for context window optimization.
func triggerPagination(agentID string, data string, pageSize int) string {
	if pageSize <= 0 || len(data) <= pageSize {
		return data
	}
	var chunks []string
	runes := []rune(data)
	for i := 0; i < len(runes); i += pageSize {
		end := i + pageSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	paginationMu.Lock()
	paginationStore[agentID] = chunks
	paginationStep[agentID] = 0
	paginationMu.Unlock()
	return fmt.Sprintf("%s\n\n[Page 1 of %d, reply \"/nextPage\" to view next page.]", chunks[0], len(chunks))
}

// loggingMiddleware records inbound HTTP routing behavior.
// Global per-IP rate limiter for all endpoints.
var (
	globalRateBuckets = make(map[string]*rateBucket)
	globalRateMu      sync.Mutex
	globalRateLimit   = 60 // requests per minute per IP
)

func globalRateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip rate limiting for login and static files.
		if r.URL.Path == "/api/v1/login" || !strings.HasPrefix(r.URL.Path, "/api/v1/") {
			next.ServeHTTP(w, r)
			return
		}
		ip := getClientIP(r)
		globalRateMu.Lock()
		now := time.Now()
		b, exists := globalRateBuckets[ip]
		if !exists || now.After(b.windowEnd) {
			globalRateBuckets[ip] = &rateBucket{count: 1, windowEnd: now.Add(time.Minute)}
		} else {
			b.count++
			if b.count > globalRateLimit {
				globalRateMu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "60")
				http.Error(w, `{"error":"Rate limit exceeded"}`, http.StatusTooManyRequests)
				return
			}
		}
		globalRateMu.Unlock()
		next.ServeHTTP(w, r)
	})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[HTTP] %s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func main() {
	if u, err := user.Current(); err == nil && (u.Uid == "0" || u.Username == "root") {
		log.Println("=================================================================")
		log.Println("[SECURITY WARNING] System is currently running as ROOT user.")
		log.Println("Native Go/Bash skills will execute with maximum system privileges.")
		log.Println("For production environments, ensure Docker container isolation.")
		log.Println("=================================================================")
	}

	db, err := sql.Open("sqlite", "./zuver.db")
	if err != nil {
		log.Fatal("Failed to open main database:", err)
	}
	db.SetMaxOpenConns(1)
	db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
	defer db.Close()

	ragDb, err := sql.Open("sqlite", "./rag.db")
	if err != nil {
		log.Fatal("Failed to open RAG database:", err)
	}
	ragDb.SetMaxOpenConns(1)
	ragDb.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
	defer ragDb.Close()

	tables := []string{
		`CREATE TABLE IF NOT EXISTS providers (id TEXT PRIMARY KEY, name TEXT, type TEXT DEFAULT 'OpenAI', endpoint TEXT, api_key TEXT, extra_config TEXT, token_usage INTEGER DEFAULT 0)`,
		`CREATE TABLE IF NOT EXISTS source_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT, data TEXT, timestamp TEXT, is_read INTEGER DEFAULT 0)`,
		`CREATE TABLE IF NOT EXISTS skills (id TEXT PRIMARY KEY, name TEXT, type TEXT, instruction TEXT, content TEXT, api_method TEXT, api_url TEXT, api_headers TEXT, api_body TEXT)`,
		`CREATE TABLE IF NOT EXISTS rags (id TEXT PRIMARY KEY, name TEXT, description TEXT, table_name TEXT)`,
		`CREATE TABLE IF NOT EXISTS preferences (id TEXT PRIMARY KEY, name TEXT, description TEXT, table_name TEXT)`,
		`CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, description TEXT, flow_json TEXT, tags TEXT DEFAULT '', is_active INTEGER DEFAULT 1)`,
		`CREATE TABLE IF NOT EXISTS mcp_servers (id TEXT PRIMARY KEY, name TEXT, url TEXT, command TEXT)`,
		`CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)`,
		`CREATE TABLE IF NOT EXISTS outputs (id TEXT PRIMARY KEY, name TEXT, instruction TEXT, command_tpl TEXT)`,
		`CREATE TABLE IF NOT EXISTS response_cache (hash TEXT PRIMARY KEY, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)`,
		`CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)`,
		`CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, name TEXT, type TEXT, api_key TEXT, file_path TEXT)`,
		`CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, name TEXT, provider_id TEXT, model TEXT, sources TEXT DEFAULT '[]', skills TEXT DEFAULT '[]', outputs TEXT DEFAULT '[]', mcps TEXT DEFAULT '[]', projects TEXT DEFAULT '[]', system_prompt TEXT, token_usage INTEGER DEFAULT 0, input_methods TEXT DEFAULT '["Text"]', output_methods TEXT DEFAULT '["Text"]', user_prompt_prefix TEXT DEFAULT '', temperature REAL DEFAULT 0.7, max_tokens INTEGER DEFAULT 4096, top_p REAL DEFAULT 1.0, privacy_enabled INTEGER DEFAULT 0)`,
		`CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, agent_id TEXT, regex TEXT, command TEXT, repeat INTEGER DEFAULT 0, active INTEGER DEFAULT 1)`,
		`CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, name TEXT, description TEXT, token TEXT, permissions TEXT)`,
		`CREATE TABLE IF NOT EXISTS analytics_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, entity_type TEXT, entity_id TEXT, tokens INT, is_success BOOLEAN)`,
		`CREATE TABLE IF NOT EXISTS security_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, action TEXT, path TEXT, actor TEXT, detail TEXT)`,
		`CREATE TABLE IF NOT EXISTS embedding_cache (hash TEXT PRIMARY KEY, embedding TEXT, model TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)`,
		`CREATE TABLE IF NOT EXISTS chat_sessions (id TEXT PRIMARY KEY, agent_id TEXT, name TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)`,
	}
	for _, q := range tables {
		if _, err := db.Exec(q); err != nil {
			log.Printf("[DB Schema Error] %v", err)
		}
	}

	autoMigrateColumn(db, "agents", "input_methods", "TEXT DEFAULT '[\"Text\"]'")
	autoMigrateColumn(db, "agents", "output_methods", "TEXT DEFAULT '[\"Text\"]'")
	autoMigrateColumn(db, "agents", "projects", "TEXT DEFAULT '[]'")
	autoMigrateColumn(db, "sources", "type", "TEXT DEFAULT 'Text'")
	autoMigrateColumn(db, "sources", "api_key", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "sources", "file_path", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "providers", "type", "TEXT DEFAULT 'OpenAI'")
	autoMigrateColumn(db, "providers", "extra_config", "TEXT DEFAULT '{}'")
	autoMigrateColumn(db, "providers", "models", "TEXT DEFAULT '[]'")
	autoMigrateColumn(db, "analytics_logs", "cost", "REAL DEFAULT 0")
	autoMigrateColumn(db, "chat_history", "session_id", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "agents", "can_create_skills", "INTEGER DEFAULT 0")
	autoMigrateColumn(db, "agents", "stream_enabled", "INTEGER DEFAULT 1")
	autoMigrateColumn(db, "projects", "tags", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "projects", "is_active", "INTEGER DEFAULT 1")
	autoMigrateColumn(db, "api_keys", "rate_limit", "INTEGER DEFAULT 0") // requests/minute, 0 = unlimited
	autoMigrateColumn(db, "api_keys", "token_hash", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "outputs", "type", "TEXT DEFAULT 'Command'")   // Command | Webhook
	autoMigrateColumn(db, "outputs", "webhook_url", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "mcp_servers", "tools", "TEXT DEFAULT '[]'") // JSON array of resolved tool names
	autoMigrateColumn(db, "agents", "mcp_tools", "TEXT DEFAULT '[]'")  // JSON array of {mcpId, tool} pairs assigned to this agent
	autoMigrateColumn(db, "skills", "use_docker", "INTEGER DEFAULT 0")  // Run skill inside Docker container

	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('presidio_enabled', 'false'), ('presidio_analyzer', 'http://localhost:3000'), ('presidio_anonymizer', 'http://localhost:3001')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('cors_enabled', 'false')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('docker_enabled', 'false'), ('docker_host', 'tcp://localhost:2375'), ('docker_image', 'zuver-skill:latest')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('error_output_id', '')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('rag_chunking', 'false'), ('rag_chunk_size', '500'), ('rag_chunk_overlap', '50')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('rag_embedding_cache', 'false')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('rag_hybrid_search', 'false')")
	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('rag_local_embedding', 'false'), ('rag_local_embedding_url', 'http://localhost:11434/api/embeddings')")

	var pCount int
	db.QueryRow("SELECT COUNT(*) FROM projects").Scan(&pCount)
	if pCount == 0 {
		defaultFlow := `{"nodes":[{"id":"start","type":"Start","x":50,"y":150},{"id":"end","type":"End","x":450,"y":150}],"edges":[]}`
		db.Exec("INSERT INTO projects (id, name, description, flow_json) VALUES (?, ?, ?, ?)", "proj_default", "Default Router", "System default routing pipeline.", defaultFlow)
	}

	loadOrCreateMasterKey()
	go startDBWorker()
	go func() {
		for {
			time.Sleep(5 * time.Minute)
			cleanupCSRFTokens()
			cleanupAuthState()
		}
	}()

	// Background update checker — checks GitHub releases every 6 hours.
	go func() {
		checkForUpdate() // check immediately on startup
		for {
			time.Sleep(6 * time.Hour)
			checkForUpdate()
		}
	}()

	// Background task monitor — starts AFTER ListenAndServe would block, so we run it before.
	go func() {
		for {
			time.Sleep(10 * time.Second)
			rows, err := db.Query("SELECT id, agent_id, regex, command, repeat FROM tasks WHERE active=1")
			if err != nil {
				continue
			}
			for rows.Next() {
				var tID, aID, regx, cmd string
				var repeat int
				rows.Scan(&tID, &aID, &regx, &cmd, &repeat)
				var lastMsg string
				db.QueryRow("SELECT content FROM chat_history WHERE agent_id=? ORDER BY id DESC LIMIT 1", aID).Scan(&lastMsg)
				if matched, _ := regexp.MatchString(regx, lastMsg); matched {
					db.Exec("INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", aID, "user", cmd)
					if repeat == 0 {
						db.Exec("UPDATE tasks SET active=0 WHERE id=?", tID)
					}
				}
			}
			rows.Close()
		}
	}()

	app := &App{ConfigDB: db, MemoryDB: ragDb}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.ServeFile(w, r, "index.html")
		} else {
			http.NotFound(w, r)
		}
	})
	mux.HandleFunc("GET /api/v1/stats", app.handleGetStats)
	mux.HandleFunc("GET /api/v1/providers", app.handleGetProviders)
	mux.HandleFunc("POST /api/v1/providers", app.handleCreateProvider)
	mux.HandleFunc("PUT /api/v1/providers/{id}", app.handleUpdateProvider)
	mux.HandleFunc("DELETE /api/v1/providers/{id}", app.handleDeleteProvider)
	mux.HandleFunc("GET /api/v1/sources", app.handleGetSources)
	mux.HandleFunc("POST /api/v1/sources", app.handleCreateSource)
	mux.HandleFunc("PUT /api/v1/sources/{id}", app.handleUpdateSource)
	mux.HandleFunc("GET /api/v1/outputs", app.handleGetOutputs)
	mux.HandleFunc("POST /api/v1/outputs", app.handleCreateOutput)
	mux.HandleFunc("PUT /api/v1/outputs/{id}", app.handleUpdateOutput)
	mux.HandleFunc("DELETE /api/v1/outputs/{id}", app.handleDeleteOutput)
	mux.HandleFunc("DELETE /api/v1/sources/{id}", app.handleDeleteSource)
	mux.HandleFunc("POST /api/v1/sources/update/{name}", app.handleSourceUpdate)
	mux.HandleFunc("GET /api/v1/skills", app.handleGetSkills)
	mux.HandleFunc("POST /api/v1/skills", app.handleCreateSkill)
	mux.HandleFunc("PUT /api/v1/skills/{id}", app.handleUpdateSkill)
	mux.HandleFunc("DELETE /api/v1/skills/{id}", app.handleDeleteSkill)
	mux.HandleFunc("GET /api/v1/rags", app.handleGetRAGs)
	mux.HandleFunc("POST /api/v1/rags", app.handleCreateRAG)
	mux.HandleFunc("PUT /api/v1/rags/{id}", app.handleUpdateRAG)
	mux.HandleFunc("DELETE /api/v1/rags/{id}", app.handleDeleteRAG)
	mux.HandleFunc("POST /api/v1/rags/{id}/query", app.handleQueryRAG)
	mux.HandleFunc("GET /api/v1/preferences", app.handleGetPreferences)
	mux.HandleFunc("POST /api/v1/preferences", app.handleCreatePreference)
	mux.HandleFunc("PUT /api/v1/preferences/{id}", app.handleUpdatePreference)
	mux.HandleFunc("DELETE /api/v1/preferences/{id}", app.handleDeletePreference)
	mux.HandleFunc("GET /api/v1/agents", app.handleGetAgents)
	mux.HandleFunc("POST /api/v1/agents", app.handleCreateAgent)
	mux.HandleFunc("PUT /api/v1/agents/{id}", app.handleUpdateAgent)
	mux.HandleFunc("DELETE /api/v1/agents/{id}", app.handleDeleteAgent)

	// POST /api/v1/agents/{id}/clone — duplicates an agent with a new ID.
	mux.HandleFunc("POST /api/v1/agents/{id}/clone", func(w http.ResponseWriter, r *http.Request) {
		srcID := r.PathValue("id")
		var ag Agent
		err := app.ConfigDB.QueryRow(
			`SELECT name, provider_id, model, sources, skills, outputs, mcps, projects,
			        system_prompt, token_usage, input_methods, output_methods, user_prompt_prefix,
			        temperature, max_tokens, top_p, privacy_enabled, can_create_skills, stream_enabled
			 FROM agents WHERE id=?`, srcID,
		).Scan(&ag.Name, &ag.ProviderID, &ag.Model, &ag.Sources, &ag.Skills, &ag.Outputs,
			&ag.MCPs, &ag.Projects, &ag.SystemPrompt, &ag.TokenUsage,
			&ag.InputMethods, &ag.OutputMethods, &ag.UserPromptPrefix,
			&ag.Temperature, &ag.MaxTokens, &ag.TopP,
			&ag.PrivacyEnabled, &ag.CanCreateSkills, &ag.StreamEnabled)
		if err != nil {
			http.Error(w, `{"error":"agent not found"}`, http.StatusNotFound)
			return
		}
		newID := fmt.Sprintf("ag_%d", time.Now().UnixNano())
		newName := ag.Name + " (Copy)"
		_, dbErr := SyncDBExec(app.ConfigDB,
			`INSERT INTO agents (id, name, provider_id, model, sources, skills, outputs, mcps, projects,
			  system_prompt, token_usage, input_methods, output_methods, user_prompt_prefix,
			  temperature, max_tokens, top_p, privacy_enabled, can_create_skills, stream_enabled)
			 VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?,?)`,
			newID, newName, ag.ProviderID, ag.Model, ag.Sources, ag.Skills, ag.Outputs,
			ag.MCPs, ag.Projects, ag.SystemPrompt,
			ag.InputMethods, ag.OutputMethods, ag.UserPromptPrefix,
			ag.Temperature, ag.MaxTokens, ag.TopP,
			ag.PrivacyEnabled, ag.CanCreateSkills, ag.StreamEnabled)
		if dbErr != nil {
			http.Error(w, `{"error":"failed to clone agent"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": newID, "name": newName})
	})

	// GET /api/v1/providers/{id}/health — pings the provider's chat endpoint and returns latency.
	mux.HandleFunc("GET /api/v1/providers/{id}/health", func(w http.ResponseWriter, r *http.Request) {
		var endpoint, apiKey, pType string
		if err := app.ConfigDB.QueryRow(
			"SELECT endpoint, api_key, COALESCE(type,'OpenAI') FROM providers WHERE id=?",
			r.PathValue("id"),
		).Scan(&endpoint, &apiKey, &pType); err != nil {
			http.Error(w, `{"error":"provider not found"}`, http.StatusNotFound)
			return
		}
		// SSRF guard: Ollama is intentionally local, but block private hosts for other provider types.
		parsedEP, epErr := url.ParseRequestURI(strings.TrimRight(endpoint, "/"))
		if epErr != nil || (parsedEP.Scheme != "http" && parsedEP.Scheme != "https") {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "error": "invalid endpoint scheme"})
			return
		}
		if strings.ToLower(pType) != "ollama" && isPrivateHost(endpoint) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "error": "private host blocked for non-local providers"})
			return
		}
		// Build a minimal models/ping request appropriate for the provider type.
		pingURL := strings.TrimRight(endpoint, "/")
		var pingReq *http.Request
		switch strings.ToLower(pType) {
		case "claude":
			pingURL += "/v1/models"
			pingReq, _ = http.NewRequest("GET", pingURL, nil)
			pingReq.Header.Set("x-api-key", apiKey)
			pingReq.Header.Set("anthropic-version", "2023-06-01")
		case "ollama":
			pingURL += "/api/tags"
			pingReq, _ = http.NewRequest("GET", pingURL, nil)
		default:
			pingURL += "/models"
			pingReq, _ = http.NewRequest("GET", pingURL, nil)
			if apiKey != "" {
				pingReq.Header.Set("Authorization", "Bearer "+apiKey)
			}
		}
	client := safeHTTPClient(8 * time.Second)
	t0 := time.Now()
	resp, err := client.Do(pingReq)
		latency := time.Since(t0).Milliseconds()
		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "latency_ms": latency, "error": err.Error()})
			return
		}
		resp.Body.Close()
		ok := resp.StatusCode < 500
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": ok, "latency_ms": latency, "status": resp.StatusCode})
	})

	mux.HandleFunc("POST /api/v1/agents/{id}/run", func(w http.ResponseWriter, r *http.Request) {
		agentID := r.PathValue("id")
		if agentID == "" {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"agent id is required"}`, http.StatusBadRequest)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
		var req struct {
			Input   string `json:"input"`   // primary field
			Message string `json:"message"` // alias for compatibility
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		// Support "message" as an alias for "input".
		if req.Input == "" {
			req.Input = req.Message
		}
		if strings.TrimSpace(req.Input) == "" {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"'input' field is required and must not be empty"}`, http.StatusBadRequest)
			return
		}

		// Build a synthetic /api/v1/chat body and re-use handleChat via an
		// in-process ResponseRecorder so we get the full pipeline for free
		// (skill execution, RAG, MCP, privacy filters, caching, analytics).
		chatBody, _ := json.Marshal(map[string]interface{}{
			"agent_id": agentID,
			"message":  req.Input,
			"stream":   false, // force non-streaming for REST response
		})
		synthetic, _ := http.NewRequest("POST", "/api/v1/chat", bytes.NewReader(chatBody))
		synthetic.Header.Set("Content-Type", "application/json")
		// Copy Authorization header so any downstream auth checks pass.
		synthetic.Header.Set("Authorization", r.Header.Get("Authorization"))

		rec := &responseRecorder{header: make(http.Header), code: http.StatusOK}
		app.handleChat(rec, synthetic)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(rec.code)
		w.Write(rec.body)
	})
	mux.HandleFunc("GET /api/v1/projects", app.handleGetProjects)
	mux.HandleFunc("POST /api/v1/projects", app.handleCreateProject)
	mux.HandleFunc("PUT /api/v1/projects/{id}", app.handleUpdateProject)
	mux.HandleFunc("DELETE /api/v1/projects/{id}", app.handleDeleteProject)
	mux.HandleFunc("POST /api/v1/projects/{id}/run", app.handleRunProject)
	mux.HandleFunc("POST /api/v1/projects/{id}/trigger", app.handleTriggerProject)
	mux.HandleFunc("GET /api/v1/mcp", app.handleGetMCP)
	mux.HandleFunc("POST /api/v1/mcp", app.handleCreateMCP)
	mux.HandleFunc("PUT /api/v1/mcp/{id}", app.handleUpdateMCP)
	mux.HandleFunc("POST /api/v1/mcp/{id}/resolve", app.handleResolveMCP)
	mux.HandleFunc("DELETE /api/v1/mcp/{id}", app.handleDeleteMCP)
	mux.HandleFunc("GET /api/v1/settings", app.handleGetSettings)
	mux.HandleFunc("POST /api/v1/settings", app.handleUpdateSettings)
	mux.HandleFunc("POST /api/v1/settings/password", app.handleChangePassword)
	mux.HandleFunc("POST /api/v1/chat", app.handleChat)
	mux.HandleFunc("GET /api/v1/history/{agent_id}", app.handleGetChatHistory)
	mux.HandleFunc("DELETE /api/v1/history/{agent_id}", app.handleClearChatHistory)
	mux.HandleFunc("GET /api/v1/history/{agent_id}/export", app.handleExportChatHistory)

	mux.HandleFunc("GET /api/v1/sessions/{agent_id}", app.handleGetSessions)
	mux.HandleFunc("POST /api/v1/sessions/{agent_id}", app.handleCreateSession)
	mux.HandleFunc("DELETE /api/v1/sessions/{session_id}", app.handleDeleteSession)

	mux.HandleFunc("GET /api/v1/tasks", app.handleGetTasks)
	mux.HandleFunc("POST /api/v1/tasks", app.handleCreateTask)
	mux.HandleFunc("PUT /api/v1/tasks/{id}", app.handleUpdateTask)
	mux.HandleFunc("DELETE /api/v1/tasks/{id}", app.handleDeleteTask)

	mux.HandleFunc("GET /api/v1/apikeys", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query("SELECT id, name, description, token, permissions, COALESCE(rate_limit,0) FROM api_keys")
		if err != nil {
			json.NewEncoder(w).Encode([]APIKey{})
			return
		}
		defer rows.Close()
		var list []APIKey
		for rows.Next() {
			var i APIKey
			rows.Scan(&i.ID, &i.Name, &i.Description, &i.Token, &i.Permissions, &i.RateLimit)
			i.Token = maskSecret(decryptSecret(i.Token))
			list = append(list, i)
		}
		if list == nil {
			list = []APIKey{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(list)
	})

	mux.HandleFunc("POST /api/v1/apikeys", func(w http.ResponseWriter, r *http.Request) {
		var i APIKey
		if err := json.NewDecoder(r.Body).Decode(&i); err != nil {
			http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
			return
		}
		i.ID = fmt.Sprintf("ak_%d", time.Now().UnixNano())
		i.Token = "zuv-" + hex.EncodeToString([]byte(i.ID))
		if i.RateLimit == 0 {
			i.RateLimit = 60
		}
		app.ConfigDB.Exec("INSERT INTO api_keys (id, name, description, token, token_hash, permissions, rate_limit) VALUES (?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Description, encryptSecret(i.Token), fmt.Sprintf("%x", sha256.Sum256([]byte(i.Token))), i.Permissions, i.RateLimit)
		app.appendSecurityEvent("apikey_create", r.URL.Path, "admin", fmt.Sprintf("created API key %s (%s)", i.ID, i.Name))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("PUT /api/v1/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		var i APIKey
		json.NewDecoder(r.Body).Decode(&i)
		app.ConfigDB.Exec("UPDATE api_keys SET name=?, description=?, permissions=?, rate_limit=? WHERE id=?", i.Name, i.Description, i.Permissions, i.RateLimit, r.PathValue("id"))
		app.appendSecurityEvent("apikey_update", r.URL.Path, "admin", fmt.Sprintf("updated API key %s", r.PathValue("id")))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /api/v1/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		app.ConfigDB.Exec("DELETE FROM api_keys WHERE id=?", r.PathValue("id"))
		app.appendSecurityEvent("apikey_delete", r.URL.Path, "admin", fmt.Sprintf("deleted API key %s", r.PathValue("id")))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/v1/login", app.handleLogin)
	mux.HandleFunc("POST /api/v1/logout", func(w http.ResponseWriter, r *http.Request) {
		adminTokenMu.Lock()
		adminToken = ""
		adminTokenExp = time.Time{}
		adminTokenMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// TOTP 2FA endpoints.
	mux.HandleFunc("POST /api/v1/settings/totp/setup", func(w http.ResponseWriter, r *http.Request) {
		secret := generateTOTPSecret()
		qrURL := getTOTPQRCodeURL(secret, "zuver@admin")
		// Store the secret temporarily (not yet enabled).
		app.ConfigDB.Exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('totp_secret', ?)", secret)
		app.ConfigDB.Exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('totp_enabled', 'false')")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"secret": secret, "qr_url": qrURL})
	})

	mux.HandleFunc("POST /api/v1/settings/totp/verify", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Code string `json:"code"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		var secret string
		app.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='totp_secret'").Scan(&secret)
		if secret == "" {
			http.Error(w, `{"error":"TOTP not set up"}`, http.StatusBadRequest)
			return
		}
		if !verifyTOTP(secret, req.Code) {
			app.appendSecurityEvent("totp_verify_fail", r.URL.Path, "admin", "invalid TOTP code")
			http.Error(w, `{"error":"Invalid TOTP code"}`, http.StatusUnauthorized)
			return
		}
		// Enable TOTP.
		app.ConfigDB.Exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('totp_enabled', 'true')")
		app.appendSecurityEvent("totp_enabled", r.URL.Path, "admin", "TOTP 2FA enabled")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /api/v1/settings/totp", func(w http.ResponseWriter, r *http.Request) {
		app.ConfigDB.Exec("DELETE FROM settings WHERE key='totp_secret'")
		app.ConfigDB.Exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('totp_enabled', 'false')")
		app.appendSecurityEvent("totp_disabled", r.URL.Path, "admin", "TOTP 2FA disabled")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("GET /api/v1/settings/totp", func(w http.ResponseWriter, r *http.Request) {
		var enabled string
		app.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='totp_enabled'").Scan(&enabled)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]bool{"enabled": enabled == "true"})
	})

	mux.HandleFunc("GET /api/v1/security/logs", func(w http.ResponseWriter, r *http.Request) {
		limit := 100
		if l := r.URL.Query().Get("limit"); l != "" {
			if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 && parsed <= 500 {
				limit = parsed
			}
		}
		offset := 0
		if o := r.URL.Query().Get("offset"); o != "" {
			if parsed, err := strconv.Atoi(o); err == nil && parsed >= 0 {
				offset = parsed
			}
		}
		action := r.URL.Query().Get("action")
		var rows *sql.Rows
		var err error
		if action != "" {
			rows, err = app.ConfigDB.Query("SELECT id, timestamp, action, path, actor, detail FROM security_logs WHERE action=? ORDER BY id DESC LIMIT ? OFFSET ?", action, limit, offset)
		} else {
			rows, err = app.ConfigDB.Query("SELECT id, timestamp, action, path, actor, detail FROM security_logs ORDER BY id DESC LIMIT ? OFFSET ?", limit, offset)
		}
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"events": []SecurityEvent{}, "total": 0})
			return
		}
		defer rows.Close()
		var events []SecurityEvent
		for rows.Next() {
			var e SecurityEvent
			rows.Scan(&e.Timestamp, &e.Timestamp, &e.Action, &e.Path, &e.Actor, &e.Detail)
			events = append(events, e)
		}
		if events == nil {
			events = []SecurityEvent{}
		}
		var total int
		if action != "" {
			app.ConfigDB.QueryRow("SELECT COUNT(*) FROM security_logs WHERE action=?", action).Scan(&total)
		} else {
			app.ConfigDB.QueryRow("SELECT COUNT(*) FROM security_logs").Scan(&total)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"events": events, "total": total, "limit": limit, "offset": offset})
	})

	mux.HandleFunc("GET /api/v1/sysinfo", func(w http.ResponseWriter, r *http.Request) {
		latestUpdateMu.Lock()
		u := latestUpdate
		latestUpdateMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"name":        "Zuver",
			"version":     CurrentVersion,
			"description": "Next-gen Generative AI Framework, built for possibilities.",
			"has_update":  u.HasUpdate,
			"latest":      u.LatestVersion,
		})
	})

	mux.HandleFunc("GET /api/v1/update-check", func(w http.ResponseWriter, r *http.Request) {
		// Allow manual refresh by triggering a check.
		if r.URL.Query().Get("refresh") == "true" {
			go checkForUpdate()
		}
		latestUpdateMu.Lock()
		u := latestUpdate
		latestUpdateMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(u)
	})

	mux.HandleFunc("GET /api/v1/analytics/summary", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query(`
			SELECT entity_id, entity_type,
			       COALESCE(SUM(tokens), 0) as tokens_24h,
			       COUNT(*) as calls_24h,
			       MAX(timestamp) as last_call,
			       COALESCE(SUM(CASE WHEN is_success THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 100) as success_rate,
			       COALESCE(SUM(cost), 0) as cost_24h
			FROM analytics_logs
			WHERE timestamp >= datetime('now', '-1 day')
			GROUP BY entity_id, entity_type
		`)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"entities": map[string]interface{}{}, "global_tokens_24h": 0, "global_cost_24h": 0})
			return
		}
		defer rows.Close()

		type Stat struct {
			Tokens24h   int     `json:"tokens_24h"`
			Calls24h    int     `json:"calls_24h"`
			LastCall    string  `json:"last_call"`
			SuccessRate float64 `json:"success_rate"`
			Cost24h     float64 `json:"cost_24h"`
		}
		stats := make(map[string]Stat)
		for rows.Next() {
			var id, eType, lastCall string
			var tokens, calls int
			var rate, cost float64
			rows.Scan(&id, &eType, &tokens, &calls, &lastCall, &rate, &cost)
			stats[id] = Stat{Tokens24h: tokens, Calls24h: calls, LastCall: lastCall, SuccessRate: rate, Cost24h: cost}
		}

		var globalTokens int
		var globalCost float64
		app.ConfigDB.QueryRow(`SELECT COALESCE(SUM(tokens), 0) FROM analytics_logs WHERE timestamp >= datetime('now', '-1 day')`).Scan(&globalTokens)
		app.ConfigDB.QueryRow(`SELECT COALESCE(SUM(cost), 0) FROM analytics_logs WHERE timestamp >= datetime('now', '-1 day')`).Scan(&globalCost)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"entities": stats, "global_tokens_24h": globalTokens, "global_cost_24h": globalCost})
	})

	mux.HandleFunc("POST /api/v1/upload", app.handleFileUpload)
	os.MkdirAll("./uploads", 0755)

	mux.HandleFunc("GET /api/v1/analytics/chart", func(w http.ResponseWriter, r *http.Request) {
		targetID := r.URL.Query().Get("id")
		query := `
		WITH RECURSIVE dates(d) AS (
			SELECT date('now', '-6 days')
			UNION ALL
			SELECT date(d, '+1 day') FROM dates WHERE d < date('now')
		)
		SELECT dates.d, COALESCE(SUM(a.tokens), 0), COUNT(a.id)
		FROM dates
		LEFT JOIN analytics_logs a ON date(a.timestamp) = dates.d `

		var dbRows *sql.Rows
		var err error
		if targetID != "" && targetID != "global" {
			query += " AND a.entity_id = ? GROUP BY dates.d ORDER BY dates.d ASC"
			dbRows, err = app.ConfigDB.Query(query, targetID)
		} else {
			query += " GROUP BY dates.d ORDER BY dates.d ASC"
			dbRows, err = app.ConfigDB.Query(query)
		}

		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"labels": []string{}, "tokens": []int{}, "calls": []int{}})
			return
		}
		defer dbRows.Close()

		var labels []string
		var tokens, calls []int
		for dbRows.Next() {
			var d string
			var t, c int
			dbRows.Scan(&d, &t, &c)
			labels = append(labels, d)
			tokens = append(tokens, t)
			calls = append(calls, c)
		}
		if labels == nil {
			labels = []string{}
			tokens = []int{}
			calls = []int{}
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"labels": labels, "tokens": tokens, "calls": calls})
	})

	mux.HandleFunc("POST /api/v1/wipe-database", func(w http.ResponseWriter, r *http.Request) {
		actor := getClientIP(r)
		if _, ok := r.Context().Value(contextKeyReauthVerified{}).(bool); ok {
			actor = "admin"
		}
		app.appendSecurityEvent("wipe_database", r.URL.Path, actor, "database wipe requested")
		tableNames := []string{"agents", "skills", "sources", "source_logs", "projects", "rags", "outputs", "mcp_servers", "chat_history", "tasks", "response_cache"}
		for _, t := range tableNames {
			// Security: table names are hardcoded — safe from injection.
			app.ConfigDB.Exec("DELETE FROM " + t)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Backup: export all data as a single JSON file, optionally encrypted with a password.
	// Usage: GET /api/v1/backup (plaintext) or POST /api/v1/backup (encrypted with password in body)
	mux.HandleFunc("GET /api/v1/backup", func(w http.ResponseWriter, r *http.Request) {
		backup := buildBackupData(app)
		out, _ := json.MarshalIndent(backup, "", "  ")
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", `attachment; filename="zuver_backup.json"`)
		w.Write(out)
	})

	mux.HandleFunc("POST /api/v1/backup", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Password string `json:"password"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Password == "" {
			http.Error(w, `{"error":"password required"}`, http.StatusBadRequest)
			return
		}
		backup := buildBackupData(app)
		out, _ := json.MarshalIndent(backup, "", "  ")
		encrypted, err := encryptBackup(out, req.Password)
		if err != nil {
			http.Error(w, `{"error":"encryption failed"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Disposition", `attachment; filename="zuver_backup.enc"`)
		w.Write(encrypted)
	})

	// Restore: import data from a JSON backup file.
	mux.HandleFunc("POST /api/v1/restore", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 50<<20) // 50 MB max
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
			return
		}

		// Try to parse as JSON first. If it fails, try to decrypt with password.
		var backup map[string]interface{}
		if err := json.Unmarshal(rawBody, &backup); err != nil {
			// Not valid JSON — try to decrypt as encrypted backup.
			var req struct {
				Password string `json:"password"`
				Data     string `json:"data"` // base64-encoded encrypted data
			}
			if err2 := json.Unmarshal(rawBody, &req); err2 != nil || req.Data == "" || req.Password == "" {
				http.Error(w, `{"error":"invalid JSON or missing password/data"}`, http.StatusBadRequest)
				return
			}
			encrypted, decErr := base64.StdEncoding.DecodeString(req.Data)
			if decErr != nil {
				http.Error(w, `{"error":"invalid base64 data"}`, http.StatusBadRequest)
				return
			}
			plaintext, decErr := decryptBackup(encrypted, req.Password)
			if decErr != nil {
				http.Error(w, `{"error":"decryption failed: wrong password?"}`, http.StatusUnauthorized)
				return
			}
			if err := json.Unmarshal(plaintext, &backup); err != nil {
				http.Error(w, `{"error":"decrypted data is not valid JSON"}`, http.StatusBadRequest)
				return
			}
		}

		app.appendSecurityEvent("database_restore", r.URL.Path, "admin", "database restore initiated")
		tables := []string{"agents", "skills", "sources", "projects", "rags", "outputs", "mcp_servers", "tasks", "api_keys", "settings", "providers"}
		imported := 0
		for _, t := range tables {
			rowsRaw, ok := backup[t]
			if !ok {
				continue
			}
			rows, ok := rowsRaw.([]interface{})
			if !ok || len(rows) == 0 {
				continue
			}
			// Get column names from first row.
			firstRow, ok := rows[0].(map[string]interface{})
			if !ok {
				continue
			}
			cols := make([]string, 0, len(firstRow))
			for k := range firstRow {
				// Security: validate column names to prevent SQL injection.
				if !safeTableNameRe.MatchString(k) {
					continue
				}
				cols = append(cols, k)
			}
			if len(cols) == 0 {
				continue
			}
			// Clear existing data.
			app.ConfigDB.Exec("DELETE FROM " + t)
			// Insert each row.
			placeholders := make([]string, len(cols))
			for i := range placeholders {
				placeholders[i] = "?"
			}
			query := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)", t, strings.Join(cols, ","), strings.Join(placeholders, ","))
			for _, rowRaw := range rows {
				row, ok := rowRaw.(map[string]interface{})
				if !ok {
					continue
				}
				args := make([]interface{}, len(cols))
				for i, col := range cols {
					args[i] = row[col]
				}
				app.ConfigDB.Exec(query, args...)
				imported++
			}
		}
		app.appendSecurityEvent("database_restore_done", r.URL.Path, "admin", fmt.Sprintf("restored %d rows", imported))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "rows_imported": imported})
	})

	// ------------------------------------------------------------------
	// /api/add — public confirmation page for one-click config import.
	// Usage: /api/add?configURL=https://example.com/agent.json
	//        Everything after "configURL=" is treated as the target URL,
	//        so query parameters inside that URL are preserved verbatim.
	//        The item type (agent/skill) is auto-detected from the JSON.
	// ------------------------------------------------------------------
	mux.HandleFunc("GET /api/add", func(w http.ResponseWriter, r *http.Request) {
		// Extract everything after "configURL=" from the raw query string so
		// that any "?" or "&" inside the remote URL is preserved intact.
		rawQuery := r.URL.RawQuery
		_, after, found := strings.Cut(rawQuery, "configURL=")
		if !found || after == "" {
			http.Error(w, "Missing configURL parameter. Usage: /api/add?configURL=https://...", http.StatusBadRequest)
			return
		}
		// URL-decode the value (browsers encode special chars when building the link).
		configURL, decErr := url.QueryUnescape(after)
		if decErr != nil {
			configURL = after // fall back to raw value if decoding fails
		}

		// SSRF guard: configURL must be a valid https URL pointing to a public host.
		parsedCfgURL, urlErr := url.ParseRequestURI(configURL)
		if urlErr != nil || parsedCfgURL.Scheme != "https" {
			http.Error(w, "configURL must be a valid https:// URL.", http.StatusBadRequest)
			return
		}
		// Block requests to private/loopback ranges.
		hostLower := strings.ToLower(parsedCfgURL.Hostname())
		for _, blocked := range []string{"localhost", "127.", "10.", "192.168.", "172.16.", "::1", "0.0.0.0", "169.254."} {
			if strings.HasPrefix(hostLower, blocked) || hostLower == blocked {
				http.Error(w, "configURL must point to a public host.", http.StatusForbidden)
				return
			}
		}

		// Generate CSRF token for this import session.
		csrfToken := generateCSRFToken()

		// Fetch the remote config JSON (timeout 10 s).
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Get(configURL)
		if err != nil || resp.StatusCode != http.StatusOK {
			http.Error(w, "Failed to fetch config from remote URL.", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512<<10)) // 512 KB max

		// Validate it is parseable JSON.
		var preview map[string]interface{}
		if err := json.Unmarshal(body, &preview); err != nil {
			http.Error(w, "Remote URL did not return valid JSON.", http.StatusBadRequest)
			return
		}

		// Auto-detect item type from the JSON content.
		// Agents: have system_prompt, model, or user_prompt_prefix.
		// Skills: have type field matching known skill types, or instruction field.
		itemType := "agent" // default
		if t, ok := preview["type"].(string); ok && allowedSkillTypes[t] {
			itemType = "skill"
		} else if _, hasInstruction := preview["instruction"]; hasInstruction {
			if _, hasSystemPrompt := preview["system_prompt"]; !hasSystemPrompt {
				itemType = "skill"
			}
		}

		// Validate structure and sanitize the remote payload before displaying or importing.
		if err := validateImportPayload(itemType, preview); err != nil {
			http.Error(w, "Remote config failed validation: "+err.Error(), http.StatusUnprocessableEntity)
			return
		}

		title, _ := preview["name"].(string)
		desc, _ := preview["description"].(string)
		if title == "" {
			title = "(unnamed)"
		}

		// Pretty-print JSON for the code preview pane.
		prettyJSON, _ := json.MarshalIndent(preview, "", "  ")
		// Escape backticks and backslashes so the JSON can be safely embedded
		// in a JS template literal (backtick string) without breaking the syntax.
		safeJSONStr := strings.ReplaceAll(strings.ReplaceAll(string(prettyJSON), `\`, `\\`), "`", "\\`")

		// Inline HTML confirmation page — no external dependencies on this host.
		escapedTitle := strings.ReplaceAll(title, `"`, `&quot;`)
		escapedDesc := strings.ReplaceAll(desc, `"`, `&quot;`)
		escapedConfigURL := strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(configURL, "&", "&amp;"), "\"", "&quot;"), "<", "&lt;")

		html := `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Import ` + itemType + ` — Zuver</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',sans-serif;background:#0a0a0a;color:#e5e5e5;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
  .card{background:#111;border:1px solid #222;border-radius:12px;max-width:560px;width:100%;overflow:hidden;box-shadow:0 24px 48px rgba(0,0,0,.6)}
  .header{padding:24px 28px 20px;border-bottom:1px solid #1e1e1e}
  .badge{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:3px 10px;border-radius:20px;margin-bottom:12px}
  .badge.skill{background:rgba(249,115,22,.15);color:#f97316;border:1px solid rgba(249,115,22,.3)}
  .badge.agent{background:rgba(59,130,246,.15);color:#3b82f6;border:1px solid rgba(59,130,246,.3)}
  .badge svg{width:12px;height:12px}
  h1{font-size:20px;font-weight:700;color:#fff;margin-bottom:6px}
  .desc{font-size:13px;color:#888;line-height:1.5}
  .origin{font-size:11px;color:#555;margin-top:8px;font-family:monospace;word-break:break-all}
  .body{padding:24px 28px}
  .preview-label{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:8px}
  .code-wrap{background:#0a0a0a;border:1px solid #1e1e1e;border-radius:6px;overflow:auto;max-height:260px}
  pre{font-size:11.5px;font-family:'Menlo','Monaco',monospace;color:#9ca3af;padding:14px 16px;white-space:pre;line-height:1.55}
  .form{margin-top:20px}
  label{font-size:12px;color:#666;display:block;margin-bottom:6px;font-weight:600}
  input[type=password]{width:100%;background:#0a0a0a;border:1px solid #2a2a2a;color:#fff;font-size:14px;padding:10px 14px;border-radius:7px;outline:none;transition:border-color .15s}
  input[type=password]:focus{border-color:#3b82f6}
  .actions{display:flex;gap:10px;margin-top:16px}
  .btn{flex:1;padding:11px 0;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:opacity .15s}
  .btn:hover{opacity:.85}
  .btn-primary{background:#3b82f6;color:#fff}
  .btn-danger{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.25)}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .msg{margin-top:12px;font-size:12px;padding:8px 12px;border-radius:6px;display:none}
  .msg.error{background:rgba(239,68,68,.1);color:#ef4444;border:1px solid rgba(239,68,68,.2);display:block}
  .msg.success{background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);display:block}
  .footer{padding:16px 28px;border-top:1px solid #1a1a1a;font-size:11px;color:#444;text-align:center}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <span class="badge ` + itemType + `">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
      ` + itemType + `
    </span>
    <h1>` + escapedTitle + `</h1>
    <p class="desc">` + escapedDesc + `</p>
    <p class="origin">Source: ` + escapedConfigURL + `</p>
  </div>
  <div class="body">
    <p class="preview-label">Configuration Preview</p>
    <div class="code-wrap"><pre id="json-preview"></pre></div>
    <form class="form" onsubmit="doImport(event)">
      <label>Admin Password</label>
      <input type="password" id="pw" placeholder="Enter admin password to confirm" required autofocus>
      <div class="actions">
        <button type="submit" class="btn btn-primary" id="confirm-btn">Import ` + itemType + `</button>
        <button type="button" class="btn btn-danger" onclick="window.close()">Cancel</button>
      </div>
      <div id="msg" class="msg"></div>
    </form>
  </div>
  <div class="footer">Zuver Framework &mdash; Importing will add this ` + itemType + ` to your instance permanently.</div>
</div>
<script>
const RAW_JSON = ` + "`" + safeJSONStr + "`" + `;
const ITEM_TYPE = "` + itemType + `";
const CONFIG_URL = "` + escapedConfigURL + `";
const CSRF_TOKEN = "` + csrfToken + `";
document.getElementById('json-preview').textContent = RAW_JSON;
async function doImport(e) {
  e.preventDefault();
  const pw = document.getElementById('pw').value;
  const btn = document.getElementById('confirm-btn');
  const msg = document.getElementById('msg');
  msg.className = 'msg';
  btn.disabled = true;
  btn.textContent = 'Importing…';
  try {
    // Step 1: authenticate to get a token.
    const loginResp = await fetch('/api/v1/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({password: pw})
    });
    if (!loginResp.ok) {
      const e = await loginResp.json().catch(()=>({error:'Wrong password'}));
      throw new Error(e.error || 'Authentication failed');
    }
    const {token} = await loginResp.json();
    // Step 2: import via the API with CSRF token.
    const importResp = await fetch('/api/v1/import', {
      method: 'POST',
      headers: {'Content-Type':'application/json','Authorization':'Bearer '+token,'X-CSRF-Token':CSRF_TOKEN},
      body: JSON.stringify({type: ITEM_TYPE, data: JSON.parse(RAW_JSON)})
    });
    if (!importResp.ok) {
      const e = await importResp.json().catch(()=>({error:'Import failed'}));
      throw new Error(e.error || 'Import failed');
    }
    msg.textContent = '` + itemType + ` imported successfully. You can close this page.';
    msg.className = 'msg success';
    btn.textContent = 'Imported';
  } catch(err) {
    msg.textContent = err.message;
    msg.className = 'msg error';
    btn.disabled = false;
    btn.textContent = 'Import ` + itemType + `';
  }
}
</script>
</body>
</html>`
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, html)
	})

	// ------------------------------------------------------------------
	// POST /api/v1/import — authenticated one-shot import endpoint.
	// Body: {"type":"skill"|"agent", "data":{...config object...}}
	// ------------------------------------------------------------------
	mux.HandleFunc("POST /api/v1/import", func(w http.ResponseWriter, r *http.Request) {
		// CSRF validation: require valid one-time CSRF token from /api/add page.
		csrfToken := r.Header.Get("X-CSRF-Token")
		if csrfToken == "" || !validateCSRFToken(csrfToken) {
			http.Error(w, `{"error":"invalid or missing CSRF token"}`, http.StatusForbidden)
			return
		}

		// Enforce body size limit — 256 KB is more than enough for a config object.
		r.Body = http.MaxBytesReader(w, r.Body, 256<<10)

		var req struct {
			Type string                 `json:"type"`
			Data map[string]interface{} `json:"data"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Data == nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}

		// Validate structure and sanitize before any DB write.
		if err := validateImportPayload(req.Type, req.Data); err != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"validation failed: `+err.Error()+`"}`, http.StatusUnprocessableEntity)
			return
		}

		str := func(key string) string {
			v, _ := req.Data[key].(string)
			return v
		}
		id := str("id")
		if id == "" {
			id = fmt.Sprintf("%s_%d", req.Type[:2], time.Now().UnixNano())
		}
		w.Header().Set("Content-Type", "application/json")
		switch req.Type {
		case "skill":
			_, err := SyncDBExec(app.ConfigDB,
				"INSERT OR REPLACE INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body) VALUES (?,?,?,?,?,?,?,?,?)",
				id, str("name"), str("type"), str("instruction"), str("content"),
				str("api_method"), str("api_url"), str("api_headers"), str("api_body"),
			)
			if err != nil {
				http.Error(w, `{"error":"database error: `+err.Error()+`"}`, http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id})
		case "agent":
			// Agents are imported without provider_id — user must assign one after import.
			_, err := SyncDBExec(app.ConfigDB,
				`INSERT OR REPLACE INTO agents
				 (id, name, provider_id, model, system_prompt, user_prompt_prefix,
				  temperature, max_tokens, top_p, privacy_enabled, can_create_skills,
				  stream_enabled, sources, skills, outputs, mcps, input_methods, output_methods)
				 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
				id, str("name"), "", str("model"), str("system_prompt"), str("user_prompt_prefix"),
				floatOr(req.Data["temperature"], 0.7), intOr(req.Data["max_tokens"], 4096),
				floatOr(req.Data["top_p"], 1.0), false, false, boolOr(req.Data["stream_enabled"], true),
				"[]", "[]", "[]", "[]", str("input_methods"), str("output_methods"),
			)
			if err != nil {
				http.Error(w, `{"error":"database error: `+err.Error()+`"}`, http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id, "note": "Provider not set — assign one in the Agent editor."})
		default:
			http.Error(w, `{"error":"unsupported type"}`, http.StatusBadRequest)
		}
	})

	// ------------------------------------------------------------------
	// GET /api/v1/agents/export  — download all agents as sanitized JSON
	// GET /api/v1/skills/export  — download all skills as sanitized JSON
	// ------------------------------------------------------------------
	mux.HandleFunc("GET /api/v1/agents/export", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query(
			`SELECT id, name, model, system_prompt, user_prompt_prefix,
			        temperature, max_tokens, top_p, stream_enabled,
			        input_methods, output_methods
			 FROM agents`)
		if err != nil {
			http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
			return
		}
		defer rows.Close()
		type ExportedAgent struct {
			ID               string  `json:"id"`
			Name             string  `json:"name"`
			Model            string  `json:"model"`
			SystemPrompt     string  `json:"system_prompt"`
			UserPromptPrefix string  `json:"user_prompt_prefix"`
			Temperature      float64 `json:"temperature"`
			MaxTokens        int     `json:"max_tokens"`
			TopP             float64 `json:"top_p"`
			StreamEnabled    bool    `json:"stream_enabled"`
			InputMethods     string  `json:"input_methods"`
			OutputMethods    string  `json:"output_methods"`
		}
		var list []ExportedAgent
		for rows.Next() {
			var a ExportedAgent
			rows.Scan(&a.ID, &a.Name, &a.Model, &a.SystemPrompt, &a.UserPromptPrefix,
				&a.Temperature, &a.MaxTokens, &a.TopP, &a.StreamEnabled,
				&a.InputMethods, &a.OutputMethods)
			list = append(list, a)
		}
		if list == nil {
			list = []ExportedAgent{}
		}
		out, _ := json.MarshalIndent(list, "", "  ")
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", `attachment; filename="zuver_agents.json"`)
		w.Write(out)
	})

	mux.HandleFunc("GET /api/v1/skills/export", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query(
			`SELECT id, name, type, instruction, content, api_method, api_url, api_headers, api_body
			 FROM skills`)
		if err != nil {
			http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
			return
		}
		defer rows.Close()
		var list []Skill
		for rows.Next() {
			var s Skill
			rows.Scan(&s.ID, &s.Name, &s.Type, &s.Instruction, &s.Content,
				&s.APIMethod, &s.APIURL, &s.APIHeaders, &s.APIBody)
			list = append(list, s)
		}
		if list == nil {
			list = []Skill{}
		}
		out, _ := json.MarshalIndent(list, "", "  ")
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", `attachment; filename="zuver_skills.json"`)
		w.Write(out)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "18806"
	}

	handler := app.securityHeadersMiddleware(globalRateLimitMiddleware(loggingMiddleware(app.authMiddleware(mux))))
	server := &http.Server{
		Addr:              ":" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		// Keep WriteTimeout long enough for streamed chat/SSE responses and large model replies.
		WriteTimeout:   10 * time.Minute,
		IdleTimeout:    60 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}

	log.Printf("Starting Zuver OS Framework on port %s", port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal("HTTP server failed:", err)
	}
}

// --------------------------------------------------------------------------
// validateImportPayload enforces strict schema checks on agent/skill JSON
// before it is committed to the database, guarding against injection,
// oversized blobs, and structurally invalid configs.
// --------------------------------------------------------------------------

// allowedSkillTypes is the closed set of values the "type" field may hold.
var allowedSkillTypes = map[string]bool{
	"API": true, "Go": true, "Bash": true, "Python": true, "JavaScript": true, "Text": true, "Prompt": true, "MD": true, "Agent": true, "Placeholder": true,
}

// dangerousPatterns lists substrings that must not appear inside any string
// field — they indicate XSS / script injection attempts.
var dangerousPatterns = []string{
	"<script", "</script", "javascript:", "data:text/html", "onerror=", "onload=",
	"eval(", "document.cookie", "window.location", "__proto__", "constructor[",
}

// validateImportPayload returns a non-nil error if the payload is unsafe or malformed.
func validateImportPayload(itemType string, data map[string]interface{}) error {
	// Total encoded size guard (independent of the earlier 512 KB HTTP body limit).
	if encoded, _ := json.Marshal(data); len(encoded) > 256<<10 {
		return fmt.Errorf("payload exceeds 256 KB limit")
	}

	// Helper: extract a string field and enforce a max byte length.
	str := func(key string, maxLen int) (string, error) {
		v, _ := data[key].(string)
		if len(v) > maxLen {
			return "", fmt.Errorf("field '%s' exceeds maximum length of %d characters", key, maxLen)
		}
		return v, nil
	}

	// Scan every string value in the top-level map for injection patterns.
	for k, raw := range data {
		s, ok := raw.(string)
		if !ok {
			continue
		}
		sLower := strings.ToLower(s)
		for _, pat := range dangerousPatterns {
			if strings.Contains(sLower, pat) {
				return fmt.Errorf("field '%s' contains disallowed pattern: %s", k, pat)
			}
		}
	}

	// Require a non-empty name.
	name, err := str("name", 128)
	if err != nil {
		return err
	}
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("field 'name' is required")
	}

	switch itemType {
	case "skill":
		// Validate skill-specific fields.
		skillType, _ := data["type"].(string)
		if skillType != "" && !allowedSkillTypes[skillType] {
			return fmt.Errorf("skill type '%s' is not allowed; must be one of: API, Go, Bash, Python, JavaScript, Text, Prompt", skillType)
		}
		if _, err := str("instruction", 32768); err != nil {
			return err
		}
		if _, err := str("content", 65536); err != nil {
			return err
		}
		// Validate api_url if present.
		if rawURL, _ := data["api_url"].(string); rawURL != "" {
			u, err := url.ParseRequestURI(rawURL)
			if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
				return fmt.Errorf("field 'api_url' must be a valid http/https URL")
			}
		}
		// api_method must be one of the standard HTTP verbs if set.
		if method, _ := data["api_method"].(string); method != "" {
			allowed := map[string]bool{"GET": true, "POST": true, "PUT": true, "PATCH": true, "DELETE": true}
			if !allowed[strings.ToUpper(method)] {
				return fmt.Errorf("field 'api_method' must be a valid HTTP method")
			}
		}

	case "agent":
		if _, err := str("system_prompt", 65536); err != nil {
			return err
		}
		if _, err := str("user_prompt_prefix", 4096); err != nil {
			return err
		}
		if _, err := str("model", 128); err != nil {
			return err
		}
		// Temperature must be 0–2 if present.
		if t, ok := data["temperature"].(float64); ok && (t < 0 || t > 2) {
			return fmt.Errorf("field 'temperature' must be between 0 and 2")
		}
		// max_tokens must be positive if present.
		if mt, ok := data["max_tokens"].(float64); ok && mt <= 0 {
			return fmt.Errorf("field 'max_tokens' must be a positive integer")
		}

	default:
		return fmt.Errorf("unsupported type '%s'; must be 'skill' or 'agent'", itemType)
	}

	return nil
}

// floatOr safely casts an interface{} to float64, falling back to def.
func floatOr(v interface{}, def float64) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	}
	return def
}

// intOr safely casts an interface{} to int, falling back to def.
func intOr(v interface{}, def int) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	}
	return def
}

// boolOr safely casts an interface{} to bool, falling back to def.
func boolOr(v interface{}, def bool) bool {
	if b, ok := v.(bool); ok {
		return b
	}
	return def
}

// responseRecorder is a minimal http.ResponseWriter that buffers the response
// body and status code so handleChat can be called in-process.
type responseRecorder struct {
	header http.Header
	code   int
	body   []byte
}

func (rr *responseRecorder) Header() http.Header  { return rr.header }
func (rr *responseRecorder) WriteHeader(code int) { rr.code = code }
func (rr *responseRecorder) Write(b []byte) (int, error) {
	rr.body = append(rr.body, b...)
	return len(b), nil
}

// --------------------------------------------------------------------------
// handleChat: main conversational pipeline with streaming support.
// --------------------------------------------------------------------------
func (a *App) handleChat(w http.ResponseWriter, r *http.Request) {
	// Limit body size to 32 MB (for file-inline payloads).
	r.Body = http.MaxBytesReader(w, r.Body, 32<<20)

	var req struct {
		AgentID         string   `json:"agent_id"`
		SessionID       string   `json:"session_id"`
		Message         string   `json:"message"`
		FilePath        string   `json:"file_path"`  // legacy single file
		FilePaths       []string `json:"file_paths"` // multi-file array (takes precedence)
		UseCache        bool     `json:"use_cache"`
		PresidioEnabled bool     `json:"presidio_enabled"`
		AnalyzerURL     string   `json:"presidio_analyzer"`
		AnonymizerURL   string   `json:"presidio_anonymizer"`
		InputType       string   `json:"input_type"`
		Stream          *bool    `json:"stream"` // optional override from caller
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Bad request body.", "logs": []string{}})
		return
	}

	// --- Load Agent ---
	var agent Agent
	err := a.ConfigDB.QueryRow(`
		SELECT id, name, provider_id, model, sources, skills, outputs, mcps,
		       COALESCE(projects,'[]'), system_prompt, input_methods, output_methods,
		       user_prompt_prefix, temperature, max_tokens, top_p, privacy_enabled,
		       COALESCE(can_create_skills,0), COALESCE(stream_enabled,1)
		FROM agents WHERE id=?`, req.AgentID).
		Scan(&agent.ID, &agent.Name, &agent.ProviderID, &agent.Model,
			&agent.Sources, &agent.Skills, &agent.Outputs, &agent.MCPs,
			&agent.Projects, &agent.SystemPrompt, &agent.InputMethods,
			&agent.OutputMethods, &agent.UserPromptPrefix, &agent.Temperature,
			&agent.MaxTokens, &agent.TopP, &agent.PrivacyEnabled,
			&agent.CanCreateSkills, &agent.StreamEnabled)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to find the selected Agent.", "logs": []string{}})
		return
	}

	// Per-agent execute permission check for external API key requests.
	// The middleware already validated the key exists and has broad agents/execute scope;
	// here we enforce the specific agent ID against the key's permissions document.
	if permJSON, ok := r.Context().Value(contextKeyAPIKeyPerms{}).(string); ok && permJSON != "" {
		var perms map[string]interface{}
		agentAllowed := false
		if json.Unmarshal([]byte(permJSON), &perms) == nil {
			if agentsVal, hasAgents := perms["agents"]; hasAgents {
				if agentsMap, ok2 := agentsVal.(map[string]interface{}); ok2 {
					// Allow if wildcard "*" has execute:true.
					if wc, hasWC := agentsMap["*"].(map[string]interface{}); hasWC {
						if b, _ := wc["execute"].(bool); b {
							agentAllowed = true
						}
					}
					// Allow if the specific agent ID has execute:true.
					if !agentAllowed {
						if entry, hasEntry := agentsMap[agent.ID].(map[string]interface{}); hasEntry {
							if b, _ := entry["execute"].(bool); b {
								agentAllowed = true
							}
						}
					}
				}
			}
		}
		if !agentAllowed {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Forbidden: API key does not have execute permission for this agent"}`, http.StatusForbidden)
			return
		}
	}

	// Determine whether streaming is active for this request.
	// Caller may override via the `stream` field; otherwise use agent setting.
	useStream := agent.StreamEnabled
	if req.Stream != nil {
		useStream = *req.Stream
	}

	// --- Input Modality Validation ---
	if req.InputType == "" {
		req.InputType = "Text"
	}
	var acceptedInputs []string
	json.Unmarshal([]byte(agent.InputMethods), &acceptedInputs)

	isInputAllowed := false
	for _, m := range acceptedInputs {
		if m == req.InputType || (req.InputType == "File" && m == "Text") {
			isInputAllowed = true
			break
		}
	}
	if !isInputAllowed {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"reply": fmt.Sprintf("[SYSTEM KERNEL] Action Denied. Agent configuration strictly prohibits '%s' input. Supported inputs: %s", req.InputType, agent.InputMethods),
			"logs":  []string{"[Modality Shield]: Input rejected by agent config."},
		})
		return
	}

	// Apply prompt overrides injected by Agent-type skill sub-calls.
	if ov := r.Header.Get("X-Agent-System-Prompt-Override"); ov != "" {
		agent.SystemPrompt = ov
	}
	if ov := r.Header.Get("X-Agent-User-Prompt-Prefix-Override"); ov != "" {
		agent.UserPromptPrefix = ov
	}

	// --- Load Provider ---
	var prov Provider
	if err := a.ConfigDB.QueryRow(
		"SELECT COALESCE(type,'OpenAI'), endpoint, api_key, COALESCE(extra_config,'{}') FROM providers WHERE id=?",
		agent.ProviderID,
	).Scan(&prov.Type, &prov.Endpoint, &prov.APIKey, &prov.ExtraConfig); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to find the assigned model provider.", "logs": []string{}})
		return
	}

	var executionLogs []string
	processedUserMsg := req.Message

	// --- Presidio PII Masking ---
	if agent.PrivacyEnabled && req.PresidioEnabled {
		executionLogs = append(executionLogs, "[Privacy]: Calling Presidio Engine...")
		// SSRF guard: Presidio URLs come from the request body; validate scheme only
		// (Presidio is typically local, so private hosts are allowed, but non-HTTP schemes are not).
		for _, rawU := range []string{req.AnalyzerURL, req.AnonymizerURL} {
			if p, err := url.ParseRequestURI(rawU); err != nil || (p.Scheme != "http" && p.Scheme != "https") {
				executionLogs = append(executionLogs, "[Privacy Error]: Invalid Presidio URL scheme — skipping PII masking.")
				goto skipPresidio
			}
		}
		analyzerEndpoint := strings.TrimRight(req.AnalyzerURL, "/") + "/analyze"
		anonymizerEndpoint := strings.TrimRight(req.AnonymizerURL, "/") + "/anonymize"

		aBody, _ := json.Marshal(map[string]interface{}{"text": processedUserMsg, "language": "en"})
		aResp, errA := http.Post(analyzerEndpoint, "application/json", bytes.NewBuffer(aBody))
		if errA == nil {
			var aResults []interface{}
			json.NewDecoder(aResp.Body).Decode(&aResults)
			aResp.Body.Close()

			if len(aResults) > 0 {
				anBody, _ := json.Marshal(map[string]interface{}{"text": processedUserMsg, "analyzer_results": aResults})
				anResp, errAn := http.Post(anonymizerEndpoint, "application/json", bytes.NewBuffer(anBody))
				if errAn == nil {
					var anResult map[string]interface{}
					json.NewDecoder(anResp.Body).Decode(&anResult)
					anResp.Body.Close()
					if finalTxt, ok := anResult["text"].(string); ok {
						processedUserMsg = finalTxt
						executionLogs = append(executionLogs, "[Privacy]: PII Data Successfully Masked.")
					}
				}
			} else {
				executionLogs = append(executionLogs, "[Privacy]: No PII Detected in user prompt.")
			}
		} else {
			executionLogs = append(executionLogs, "[Privacy]: Presidio Container Unreachable. Skipping protection layer.")
		}
	}
skipPresidio:

	AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, session_id, role, content) VALUES (?, ?, ?, ?)", agent.ID, req.SessionID, "user", processedUserMsg)

	// --- Build help panel & allowlists ---
	helpPanel := "Commands:\n/getSourcesList\n/getSourceData <id>\n/nextPage\n/getRAGList\n/getRAGDataList <id>\n/addRAGData <id> <name> <data>\n/deleteRAGData <id> <name>\n/editRAGData <id> <name> <data>\n/getPrefList\n/getPrefDataList <id>\n/addPrefData <id> <name> <data>\n/deletePrefData <id> <name>\n/editPrefData <id> <name> <data>\n"
	if agent.CanCreateSkills {
		helpPanel += "/createSkill <name> <description> - Create native Go skill dynamically.\n"
	}

	allowedRAGs := make(map[string]bool)
	allowedPreferences := make(map[string]bool)
	allowedSources := make(map[string]bool)
	allowedDynamicTools := make(map[string]bool)
	activeInjections := ""

	var sourceMaps []map[string]interface{}
	json.Unmarshal([]byte(agent.Sources), &sourceMaps)
	for _, sm := range sourceMaps {
		srcID, _ := sm["id"].(string)
		var sName, sType, sFile string
		if a.ConfigDB.QueryRow("SELECT name, type, file_path FROM sources WHERE id=?", srcID).Scan(&sName, &sType, &sFile) == nil {
			allowedSources[sName] = true
			if sm["fetch_prompt"] == true {
				if sType == "Local File" {
					if b, err := os.ReadFile(sFile); err == nil {
						activeInjections += fmt.Sprintf("\n[Source Data %s]\n%s\n", sName, string(b))
					}
				} else {
					var d, ts string
					if a.ConfigDB.QueryRow("SELECT data, timestamp FROM source_logs WHERE source_id=? ORDER BY id DESC LIMIT 1", srcID).Scan(&d, &ts) == nil {
						activeInjections += fmt.Sprintf("\n[Source Data %s (At %s)]\n%s\n", sName, ts, d)
					}
				}
			}
		} else if a.ConfigDB.QueryRow("SELECT name FROM rags WHERE id=?", srcID).Scan(&sName) == nil {
			allowedRAGs[srcID] = true
		} else if a.ConfigDB.QueryRow("SELECT name FROM preferences WHERE id=?", srcID).Scan(&sName) == nil {
			allowedPreferences[srcID] = true
		}
	}

	var skillIDs []string
	json.Unmarshal([]byte(agent.Skills), &skillIDs)
	for _, sid := range skillIDs {
		var sName, sInst, sType, sCont, sUrl string
		if a.ConfigDB.QueryRow("SELECT name, instruction, type, content, api_url FROM skills WHERE id=?", sid).Scan(&sName, &sInst, &sType, &sCont, &sUrl) == nil {
			allowedDynamicTools[sName] = true
			argsStr := strings.Join(extractPlaceholders(sCont+sUrl), " ")
			helpPanel += fmt.Sprintf("/%s %s - %s [%s]\n", sName, argsStr, sInst, sType)
		}
	}

	var mcpIDs []string
	json.Unmarshal([]byte(agent.MCPs), &mcpIDs)
	for _, mid := range mcpIDs {
		var mName string
		if a.ConfigDB.QueryRow("SELECT name FROM mcp_servers WHERE id=?", mid).Scan(&mName) == nil {
			helpPanel += fmt.Sprintf("/mcp_%s <args> - Exec MCP: %s\n", mName, mName)
		}
	}

	var projIDs []string
	json.Unmarshal([]byte(agent.Projects), &projIDs)
	for _, pid := range projIDs {
		var pName, pFlow string
		if a.ConfigDB.QueryRow("SELECT name, flow_json FROM projects WHERE id=?", pid).Scan(&pName, &pFlow) == nil {
			var flow struct {
				Nodes []map[string]interface{} `json:"nodes"`
			}
			json.Unmarshal([]byte(pFlow), &flow)
			for _, n := range flow.Nodes {
				if n["type"] == "Start" && n["trigger"] == "Command" {
					if cmdName, ok := n["param"].(string); ok {
						helpPanel += fmt.Sprintf("%s - Execute Visual Project: %s\n", cmdName, pName)
					}
				}
			}
		}
	}

	// --- RAG auto-retrieval ---
	if len(allowedRAGs) > 0 && processedUserMsg != "" {
		var hybridEnabled string
		a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='rag_hybrid_search'").Scan(&hybridEnabled)
		userVector := getEmbedding(processedUserMsg, a.ConfigDB)
		keywords := strings.Fields(strings.ToLower(processedUserMsg))

		activeInjections += "\n[RAG Auto-Retrieved Memory]\n"
		for ragID := range allowedRAGs {
			var rawAutoTName string
			a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&rawAutoTName)
			if rawAutoTName == "" {
				continue
			}
			tName, tErr := sanitizeTableName(rawAutoTName)
			if tErr != nil {
				log.Printf("[Security] RAG auto-retrieval: table name failed sanitization for ragID %q: %v", ragID, tErr)
				continue
			}
			rows, err := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data, vector FROM %s", tName))
			if err != nil {
				continue
			}
			type ragCandidate struct {
				name  string
				data  string
				score float32
			}
			var candidates []ragCandidate
			for rows.Next() {
				var rName, rData, vJson string
				rows.Scan(&rName, &rData, &vJson)
				var score float32

				// Vector similarity score.
				if userVector != nil {
					var dbVector []float32
					if json.Unmarshal([]byte(vJson), &dbVector) == nil && len(dbVector) > 0 {
						score = cosineSimilarity(userVector, dbVector)
					}
				}

				// Hybrid: add keyword match bonus.
				if hybridEnabled == "true" {
					dataLower := strings.ToLower(rData)
					keywordHits := 0
					for _, kw := range keywords {
						if len(kw) > 2 && strings.Contains(dataLower, kw) {
							keywordHits++
						}
					}
					if len(keywords) > 0 {
						keywordScore := float32(keywordHits) / float32(len(keywords))
						score = score*0.7 + keywordScore*0.3 // 70% vector, 30% keyword
					}
				}

				if score > 0.3 {
					candidates = append(candidates, ragCandidate{name: rName, data: rData, score: score})
				}
			}
			rows.Close()

			// Sort by score descending, take top 3.
			sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
			limit := 3
			if len(candidates) < limit {
				limit = len(candidates)
			}
			for i := 0; i < limit; i++ {
				c := candidates[i]
				activeInjections += fmt.Sprintf("- From %s (Match: %.2f%%): %s\n", c.name, c.score*100, c.data)
			}
		}
	}

	// --- Build dynamic system prompt ---
	dynamicSysPrompt := fmt.Sprintf(
		"%s\n\n[ENV]\nTime: %s\nOutput Methods: %s\n\n[PROTOCOL]\nTo use a tool, reply MUST be ONE line starting with '/'. e.g. /help",
		agent.SystemPrompt,
		time.Now().Format("2006-01-02 15:04:05"),
		agent.OutputMethods,
	)
	if activeInjections != "" {
		dynamicSysPrompt += "\n\n[REAL-TIME DATA]\n" + activeInjections
	}

	// --- Load chat history (last 40 messages) ---
	// Base messages list — system prompt is added per-provider inside the loop
	// because Claude requires it as a top-level key, not a messages entry.
	var messages []map[string]interface{}
	histRows, err := a.ConfigDB.Query(
		"SELECT role, content FROM (SELECT role, content, id FROM chat_history WHERE agent_id=? ORDER BY id DESC LIMIT 40) ORDER BY id ASC",
		agent.ID,
	)
	if err == nil {
		for histRows.Next() {
			var role, content string
			histRows.Scan(&role, &content)
			if role == "user" && agent.UserPromptPrefix != "" {
				content = agent.UserPromptPrefix + "\n" + content
			}
			// Skip system-tagged history rows — they were internal tool results
			// injected as system messages. Re-inject them as assistant context.
			if role == "system" {
				role = "assistant"
			}
			messages = append(messages, map[string]interface{}{"role": role, "content": content})
		}
		histRows.Close()
	}

	// --- File payload extraction ---
	// Normalize: prefer FilePaths array; fall back to legacy single FilePath.
	totalTokensUsed := 0
	currentUserText := processedUserMsg
	if currentUserText == "" {
		currentUserText = "Please analyze the attached file(s)."
	}

	// Build canonical list of file paths to process.
	var allFilePaths []string
	if len(req.FilePaths) > 0 {
		allFilePaths = req.FilePaths
	} else if req.FilePath != "" {
		allFilePaths = []string{req.FilePath}
	}

	// Filter to non-text input types only (text files are injected as prompt text below).
	type fileEntry struct {
		mime string
		data string // base64
	}
	var binaryFiles []fileEntry // non-text images, pdfs, etc.

	for _, fp := range allFilePaths {
		if req.InputType == "Text" || fp == "" {
			continue
		}
		mime, data, ok := extractFilePayload(fp)
		if !ok {
			executionLogs = append(executionLogs, "[I/O Shield]: Failed to read file: "+filepath.Base(fp))
			continue
		}
		executionLogs = append(executionLogs, fmt.Sprintf("[Modality]: Loaded %s (%s).", filepath.Base(fp), mime))
		if mime == "text/plain" {
			// Plain text files are injected directly into the prompt.
			if rawBytes, err := base64.StdEncoding.DecodeString(data); err == nil {
				baseName := filepath.Base(fp)
				originalName := baseName
				if parts := strings.SplitN(baseName, "___", 3); len(parts) == 3 {
					originalName = parts[2]
				}
				currentUserText = fmt.Sprintf("Attached File (%s):\n\n```\n%s\n```\n\n%s", originalName, string(rawBytes), currentUserText)
				executionLogs = append(executionLogs, fmt.Sprintf("[Modality]: Injected %s into prompt.", originalName))
			}
		} else {
			binaryFiles = append(binaryFiles, fileEntry{mime, data})
		}
	}

	hasFile := len(binaryFiles) > 0

	// -------------------------------------------------------------------------
	// Streaming infrastructure — set SSE headers once, before any loop.
	// -------------------------------------------------------------------------
	var sseWriter http.Flusher
	var canSSEFlush bool
	if useStream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		sseWriter, canSSEFlush = w.(http.Flusher)
	}

	// sseEmit writes a single SSE data frame and flushes immediately.
	sseEmit := func(payload map[string]interface{}) {
		fmt.Fprintf(w, "data: %s\n\n", jsonStr(payload))
		if canSSEFlush {
			sseWriter.Flush()
		}
	}

	// -------------------------------------------------------------------------
	// Main agentic loop (up to 5 iterations to handle tool calls).
	// -------------------------------------------------------------------------
	for loop := 0; loop < 5; loop++ {
		apiURL := strings.TrimRight(prov.Endpoint, "/")
		pType := strings.ToLower(prov.Type)

		// Build currentMessages with the system prompt injected correctly per provider.
		// Claude: system prompt is a top-level reqBody key — do NOT include it in messages.
		// OpenAI / Ollama / compatible: prepend {"role":"system","content":"..."}.
		var currentMessages []map[string]interface{}
		if pType != "claude" {
			currentMessages = append(currentMessages, map[string]interface{}{
				"role":    "system",
				"content": dynamicSysPrompt,
			})
		}
		currentMessages = append(currentMessages, messages...)

		// Build final user message (with file if applicable).
		var finalUserMsg map[string]interface{}
		if hasFile {
			switch pType {
			case "claude":
				contentArray := []map[string]interface{}{}
				for _, bf := range binaryFiles {
					if strings.HasPrefix(bf.mime, "image/") {
						contentArray = append(contentArray, map[string]interface{}{
							"type":   "image",
							"source": map[string]string{"type": "base64", "media_type": bf.mime, "data": bf.data},
						})
					} else if bf.mime == "application/pdf" {
						contentArray = append(contentArray, map[string]interface{}{
							"type":   "document",
							"source": map[string]string{"type": "base64", "media_type": bf.mime, "data": bf.data},
						})
					} else {
						currentUserText = fmt.Sprintf("[File base64/%s]\n%s", bf.mime, currentUserText)
					}
				}
				contentArray = append(contentArray, map[string]interface{}{"type": "text", "text": currentUserText})
				finalUserMsg = map[string]interface{}{"role": "user", "content": contentArray}

			case "ollama":
				var imageDataList []string
				for _, bf := range binaryFiles {
					if strings.HasPrefix(bf.mime, "image/") {
						imageDataList = append(imageDataList, bf.data)
					} else {
						currentUserText = fmt.Sprintf("[System Note: Attached %s file, unsupported natively.]\n%s", bf.mime, currentUserText)
					}
				}
				if len(imageDataList) > 0 {
					finalUserMsg = map[string]interface{}{
						"role":    "user",
						"content": currentUserText,
						"images":  imageDataList,
					}
				} else {
					finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
				}

			default: // OpenAI-compatible
				contentArray := []map[string]interface{}{{"type": "text", "text": currentUserText}}
				for _, bf := range binaryFiles {
					if strings.HasPrefix(bf.mime, "image/") {
						contentArray = append(contentArray, map[string]interface{}{
							"type":      "image_url",
							"image_url": map[string]string{"url": fmt.Sprintf("data:%s;base64,%s", bf.mime, bf.data)},
						})
					} else {
						currentUserText = fmt.Sprintf("[System Note: User attached a %s file, not supported by standard chat completion.]\n%s", bf.mime, currentUserText)
					}
				}
				if len(contentArray) > 1 {
					finalUserMsg = map[string]interface{}{"role": "user", "content": contentArray}
				} else {
					finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
				}
			}
		} else {
			finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
		}
		currentMessages = append(currentMessages, finalUserMsg)

		// Build API endpoint URL.
		switch pType {
		case "claude":
			if !strings.HasSuffix(apiURL, "/v1/messages") {
				apiURL += "/v1/messages"
			}
		case "ollama":
			if !strings.HasSuffix(apiURL, "/api/chat") {
				apiURL += "/api/chat"
			}
		default:
			if !strings.HasSuffix(apiURL, "/v1/chat/completions") {
				apiURL += "/v1/chat/completions"
			}
		}

		// Build request body scoped per provider type.
		reqBody := map[string]interface{}{
			"model":    agent.Model,
			"messages": currentMessages,
		}

		switch pType {
		case "claude":
			// Anthropic messages API differences:
			//   • system prompt is a top-level "system" string, NOT in messages
			//   • max_tokens is REQUIRED (must be > 0)
			//   • stream is omitted when false (sending stream:false is fine but
			//     sending it is unnecessary; sending stream:true enables SSE)
			//   • temperature and top_p are mutually exclusive
			reqBody["system"] = dynamicSysPrompt
			claudeMaxTokens := agent.MaxTokens
			if claudeMaxTokens <= 0 {
				claudeMaxTokens = 4096 // safe default; required by Anthropic
			}
			reqBody["max_tokens"] = claudeMaxTokens
			if useStream {
				reqBody["stream"] = true
			}
			// Only set one of temperature/top_p (Anthropic rejects both together).
			if agent.TopP != 1.0 {
				reqBody["top_p"] = agent.TopP
			} else {
				reqBody["temperature"] = agent.Temperature
			}

		case "ollama":
			// Ollama /api/chat:
			//   • always send stream so we control NDJSON vs single-object mode
			//   • sampling params go inside "options"
			reqBody["stream"] = useStream
			options := map[string]interface{}{
				"temperature": agent.Temperature,
			}
			if agent.MaxTokens > 0 {
				options["num_predict"] = agent.MaxTokens
			}
			if agent.TopP != 1.0 {
				options["top_p"] = agent.TopP
			}
			reqBody["options"] = options

		default:
			// OpenAI-compatible providers (OpenAI, Azure, Together, Groq, Qwen,
			// DeepSeek, Mistral, etc.).
			reqBody["stream"] = useStream
			reqBody["temperature"] = agent.Temperature
			if agent.MaxTokens > 0 {
				reqBody["max_tokens"] = agent.MaxTokens
			}
			// Only send top_p when it deviates from the default; some providers
			// (Qwen, ERNIE, etc.) return error 2013 when both temperature and
			// top_p are present even at their default values.
			if agent.TopP != 1.0 {
				reqBody["top_p"] = agent.TopP
			}
		}

		// Merge extra provider params from the provider's ExtraConfig JSON.
		// These override the defaults above for advanced per-provider tuning.
		// Protected parameters cannot be overridden by ExtraConfig.
		protectedParams := map[string]bool{"model": true, "messages": true, "stream": true, "temperature": true, "max_tokens": true, "top_p": true, "system": true}
		var extras map[string]interface{}
		if json.Unmarshal([]byte(prov.ExtraConfig), &extras) == nil {
			for k, v := range extras {
				if !protectedParams[k] {
					reqBody[k] = v
				}
			}
		}

		payloadBytes, _ := json.Marshal(reqBody)

		// Cache lookup (only for non-streaming, non-loop-0+ tool calls).
		cacheHash := sha256.Sum256(payloadBytes)
		cacheKey := hex.EncodeToString(cacheHash[:])
		var replyContent string

		if !useStream && req.UseCache {
			var cachedReply string
			if a.ConfigDB.QueryRow("SELECT response FROM response_cache WHERE hash=?", cacheKey).Scan(&cachedReply) == nil && cachedReply != "" {
				replyContent = cachedReply
				executionLogs = append(executionLogs, "[Cache Hit]: Serving cached response.")
			}
		}

		if replyContent == "" {
			apiReq, errReq := http.NewRequest("POST", apiURL, bytes.NewBuffer(payloadBytes))
			if errReq != nil {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to generate API request.", "logs": append(executionLogs, errReq.Error())})
				return
			}

			apiReq.Header.Set("Content-Type", "application/json")
			switch pType {
			case "claude":
				apiReq.Header.Set("x-api-key", prov.APIKey)
				apiReq.Header.Set("anthropic-version", "2023-06-01")
			default:
				if pType != "ollama" {
					apiReq.Header.Set("Authorization", "Bearer "+prov.APIKey)
				}
			}

			// ---------- STREAMING PATH (all loops) ----------
			if useStream {
				client := &http.Client{Timeout: 120 * time.Second}
				resp, errDo := client.Do(apiReq)
				if errDo != nil {
					executionLogs = append(executionLogs, "[Network Error]: "+errDo.Error())
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					a.notifySystemError("Agent API Connection Failed", fmt.Sprintf("Agent %s could not connect to provider: %s", agent.Name, errDo.Error()))
					sseEmit(map[string]interface{}{"done": true, "error": "[SYSTEM] Failed to connect to the API provider.", "logs": executionLogs})
					return
				}

				// Non-2xx from provider means the body is a JSON error, not an SSE stream.
				if resp.StatusCode < 200 || resp.StatusCode >= 300 {
					errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
					resp.Body.Close()
					// Try to extract a human-readable error message from the provider JSON.
					var errObj map[string]interface{}
					errMsg := fmt.Sprintf("[SYSTEM] Provider returned HTTP %d.", resp.StatusCode)
					if json.Unmarshal(errBody, &errObj) == nil {
						if errDetail, ok := errObj["error"].(map[string]interface{}); ok {
							if msg, ok := errDetail["message"].(string); ok && msg != "" {
								errMsg = "[Provider Error] " + msg
							}
						} else if msg, ok := errObj["message"].(string); ok && msg != "" {
							errMsg = "[Provider Error] " + msg
						}
					}
					executionLogs = append(executionLogs, fmt.Sprintf("[HTTP %d]: %s", resp.StatusCode, string(errBody)))
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					sseEmit(map[string]interface{}{"done": true, "error": errMsg, "logs": executionLogs})
					return
				}

				var fullReply strings.Builder
				scanner := bufio.NewScanner(resp.Body)
				for scanner.Scan() {
					line := scanner.Text()
					if !strings.HasPrefix(line, "data: ") {
						// For Ollama which doesn't use SSE prefix, try parsing raw JSON.
						if pType == "ollama" && strings.HasPrefix(line, "{") {
							var chunk map[string]interface{}
							if json.Unmarshal([]byte(line), &chunk) == nil {
								if msg, ok := chunk["message"].(map[string]interface{}); ok {
									if delta, _ := msg["content"].(string); delta != "" {
										fullReply.WriteString(delta)
										sseEmit(map[string]interface{}{"delta": delta})
									}
								}
								if done, _ := chunk["done"].(bool); done {
									break
								}
							}
						}
						continue
					}
					data := strings.TrimPrefix(line, "data: ")
					if data == "[DONE]" {
						break
					}

					var chunk map[string]interface{}
					if json.Unmarshal([]byte(data), &chunk) != nil {
						continue
					}

					// Claude terminates with {"type":"message_stop"} — treat as done.
					if chunkType, _ := chunk["type"].(string); chunkType == "message_stop" {
						break
					}

					// Surface streaming-level error events from the provider.
					if errObj, ok := chunk["error"].(map[string]interface{}); ok {
						errMsg := "[Provider Error] Unknown streaming error."
						if msg, ok := errObj["message"].(string); ok && msg != "" {
							errMsg = "[Provider Error] " + msg
						}
						executionLogs = append(executionLogs, errMsg)
						resp.Body.Close()
						a.logAnalytics("agent", agent.ID, 0, false)
						a.logAnalytics("provider", agent.ProviderID, 0, false)
						sseEmit(map[string]interface{}{"done": true, "error": errMsg, "logs": executionLogs})
						return
					}

					var delta string
					switch pType {
					case "claude":
						// Claude SSE event types:
						//   content_block_delta → delta.type="text_delta", delta.text="..."
						//   message_delta       → contains stop_reason (no text)
						//   message_start/stop, content_block_start/stop → no text
						chunkType, _ := chunk["type"].(string)
						if chunkType == "content_block_delta" {
							if deltaObj, ok := chunk["delta"].(map[string]interface{}); ok {
								delta, _ = deltaObj["text"].(string)
							}
						}
					case "ollama":
						if msg, ok := chunk["message"].(map[string]interface{}); ok {
							delta, _ = msg["content"].(string)
						}
					default:
						if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
							if choice, ok := choices[0].(map[string]interface{}); ok {
								if deltaObj, ok := choice["delta"].(map[string]interface{}); ok {
									delta, _ = deltaObj["content"].(string)
								}
							}
						}
					}

					if delta != "" {
						fullReply.WriteString(delta)
						sseEmit(map[string]interface{}{"delta": delta})
					}
				}
				resp.Body.Close()

				// Collect complete reply for this loop iteration.
				replyContent = fullReply.String()
				if replyContent == "" {
					// Empty stream means the provider sent nothing useful — surface an error.
					errMsg := "[SYSTEM] Provider returned an empty streaming response. Check model name and API key."
					executionLogs = append(executionLogs, "[Empty Stream]")
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					sseEmit(map[string]interface{}{"done": true, "error": errMsg, "logs": executionLogs})
					return
				}
				totalTokensUsed += estimateTokens(string(payloadBytes)) + estimateTokens(replyContent)
			} else {
				// ---------- NON-STREAMING PATH ----------
				resp2, errDo2 := (&http.Client{Timeout: 120 * time.Second}).Do(apiReq)
				if errDo2 != nil {
					executionLogs = append(executionLogs, "[Network Error]: "+errDo2.Error())
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					a.notifySystemError("Agent API Connection Failed", fmt.Sprintf("Agent %s could not connect to provider: %s", agent.Name, errDo2.Error()))
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to connect to the API provider.", "logs": executionLogs})
					return
				}

				bodyBytes, _ := io.ReadAll(resp2.Body)
				resp2.Body.Close()

				// Non-2xx: parse and surface the provider error message before attempting to extract content.
				if resp2.StatusCode < 200 || resp2.StatusCode >= 300 {
					var errObj map[string]interface{}
					errMsg := fmt.Sprintf("[SYSTEM] Provider returned HTTP %d.", resp2.StatusCode)
					if json.Unmarshal(bodyBytes, &errObj) == nil {
						if errDetail, ok := errObj["error"].(map[string]interface{}); ok {
							if msg, ok := errDetail["message"].(string); ok && msg != "" {
								errMsg = "[Provider Error] " + msg
							}
						} else if msg, ok := errObj["message"].(string); ok && msg != "" {
							errMsg = "[Provider Error] " + msg
						}
					}
					executionLogs = append(executionLogs, fmt.Sprintf("[HTTP %d]: %s", resp2.StatusCode, string(bodyBytes)))
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": errMsg, "logs": executionLogs})
					return
				}

				var result2 map[string]interface{}
				if err := json.Unmarshal(bodyBytes, &result2); err != nil {
					executionLogs = append(executionLogs, "[Parse Error]: Provider returned non-JSON body.")
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Provider returned an invalid (non-JSON) response.", "logs": executionLogs})
					return
				}

				if choices, ok := result2["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if msg, ok := choice["message"].(map[string]interface{}); ok {
							replyContent, _ = msg["content"].(string)
							// content can be null for tool-call-only responses; fall through to empty check.
						}
					}
				} else if message, ok := result2["message"].(map[string]interface{}); ok {
					// Ollama non-streaming wraps the reply in "message".
					replyContent, _ = message["content"].(string)
				} else if contentArr, ok := result2["content"].([]interface{}); ok && len(contentArr) > 0 {
					// Claude returns an array of content blocks.
					var combinedText []string
					for _, b := range contentArr {
						if block, ok := b.(map[string]interface{}); ok {
							if txt, ok := block["text"].(string); ok && txt != "" {
								combinedText = append(combinedText, txt)
							}
							if block["type"] == "tool_use" {
								toolName2, _ := block["name"].(string)
								inputData2, _ := json.Marshal(block["input"])
								combinedText = append(combinedText, fmt.Sprintf("/native_tool_%s %s", toolName2, string(inputData2)))
							}
						}
					}
					replyContent = strings.Join(combinedText, "\n")
				}

				if replyContent == "" {
					// Try to surface an inline error field even on a 200 response (some providers do this).
					var errMsg string
					if errDetail, ok := result2["error"].(map[string]interface{}); ok {
						errMsg, _ = errDetail["message"].(string)
					}
					if errMsg == "" {
						errMsg = "[SYSTEM] Provider returned an empty response. Check your model name and API key."
					} else {
						errMsg = "[Provider Error] " + errMsg
					}
					executionLogs = append(executionLogs, "[Empty Response]: "+string(bodyBytes))
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": errMsg, "logs": executionLogs})
					return
				}

				if req.UseCache {
					AsyncDBExec(a.ConfigDB, "INSERT OR REPLACE INTO response_cache (hash, response) VALUES (?, ?)", cacheKey, replyContent)
				}
				totalTokensUsed += estimateTokens(string(payloadBytes)) + estimateTokens(replyContent)
			} // end else (non-streaming)
		} // end if replyContent == ""

		replyContent = strings.TrimSpace(replyContent)

		// --- Skill editor state check ---
		skillEditorMu.Lock()
		editorState, isEditing := skillEditorStore[agent.ID]
		if isEditing {
			delete(skillEditorStore, agent.ID)
		}
		skillEditorMu.Unlock()

		if isEditing {
			if replyContent == "/cancel" {
				messages = append(messages, map[string]interface{}{"role": "user", "content": "[SYSTEM] Skill creation cancelled."})
				continue
			}
			cleanCode := replyContent
			for _, prefix := range []string{"```go", "```"} {
				cleanCode = strings.TrimPrefix(cleanCode, prefix)
			}
			cleanCode = strings.TrimSuffix(strings.TrimSpace(cleanCode), "```")
			cleanCode = strings.TrimSpace(cleanCode)

			skillID := "sk_" + fmt.Sprint(time.Now().UnixNano())
			a.ConfigDB.Exec("INSERT INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
				skillID, editorState["name"], "Go", editorState["desc"], cleanCode, "GET", "", "", "")

			var currentSkills []string
			json.Unmarshal([]byte(agent.Skills), &currentSkills)
			currentSkills = append(currentSkills, skillID)
			newSkillsJson, _ := json.Marshal(currentSkills)
			a.ConfigDB.Exec("UPDATE agents SET skills=? WHERE id=?", string(newSkillsJson), agent.ID)
			agent.Skills = string(newSkillsJson)
			allowedDynamicTools[editorState["name"]] = true

			sysMsg := fmt.Sprintf("[SYSTEM SUCCESS] Native Go Skill '%s' compiled and mounted. You can now use /%s. Please inform the user.", editorState["name"], editorState["name"])
			messages = append(messages, map[string]interface{}{"role": "user", "content": sysMsg})
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, session_id, role, content) VALUES (?, ?, ?, ?)", agent.ID, req.SessionID, "system", sysMsg)
			continue
		}

		messages = append(messages, map[string]interface{}{"role": "assistant", "content": replyContent})
		executionLogs = append(executionLogs, "[Agent]: "+replyContent)
		AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, session_id, role, content) VALUES (?, ?, ?, ?)", agent.ID, req.SessionID, "assistant", replyContent)

		// --- Command parsing ---
		var cmdLines []string
		for _, line := range strings.Split(replyContent, "\n") {
			if line = strings.TrimSpace(line); strings.HasPrefix(line, "/") {
				cmdLines = append(cmdLines, line)
			}
		}

		if len(cmdLines) > 0 {
			var mu sync.Mutex
			var combinedSysMsg string

			// For streaming mode emit tool_start immediately, then execute in parallel,
			// then emit tool_end. Sequential emit ordering is best-effort via a channel.
			type toolResult struct {
				cmd    string
				result string
				ms     int64
			}
			results := make([]toolResult, len(cmdLines))

			var wg sync.WaitGroup
			for i, cmdLine := range cmdLines {
				wg.Add(1)
				// Emit tool_start immediately (before goroutine) so the UI sees it right away.
				if useStream {
					sseEmit(map[string]interface{}{
						"tool_start": cmdLine,
						"step":       i,
					})
				}
				go func(idx int, cLine string) {
					defer wg.Done()
					t0 := time.Now()
					sysMsg := a.executeCommand(cLine, agent, projIDs, allowedSources, allowedRAGs, allowedPreferences, allowedDynamicTools, helpPanel, &executionLogs, &mu)
					elapsed := time.Since(t0).Milliseconds()
					results[idx] = toolResult{cmd: cLine, result: sysMsg, ms: elapsed}
				}(i, cmdLine)
			}
			wg.Wait()

			// After all tools complete emit tool_end events and build system message.
			for i, tr := range results {
				if useStream {
					sseEmit(map[string]interface{}{
						"tool_end": tr.cmd,
						"step":     i,
						"ms":       tr.ms,
						"result":   tr.result,
					})
				}
				mu.Lock()
				combinedSysMsg += fmt.Sprintf("[Command Executed: %s]\n%s\n\n", tr.cmd, tr.result)
				mu.Unlock()
			}

			combinedSysMsg = strings.TrimSpace(combinedSysMsg)
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, session_id, role, content) VALUES (?, ?, ?, ?)", agent.ID, req.SessionID, "system", combinedSysMsg)

			// Check if any result should be sent directly to user (Agent Skill with return_to: "user").
			if strings.HasPrefix(combinedSysMsg, "[USER_REPLY]") {
				userReply := strings.TrimPrefix(combinedSysMsg, "[USER_REPLY]")
				AsyncDBExec(a.ConfigDB, "UPDATE agents SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ID)
				AsyncDBExec(a.ConfigDB, "UPDATE providers SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ProviderID)
				a.logAnalytics("agent", agent.ID, totalTokensUsed, true)
				a.logAnalytics("provider", agent.ProviderID, totalTokensUsed, true)
				if useStream {
					sseEmit(map[string]interface{}{"delta": userReply})
					sseEmit(map[string]interface{}{"done": true, "logs": executionLogs})
				} else {
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": userReply, "logs": executionLogs})
				}
				return
			}

			messages = append(messages, map[string]interface{}{"role": "user", "content": combinedSysMsg})
			continue
		}

		// No tool call — final response.
		AsyncDBExec(a.ConfigDB, "UPDATE agents SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ID)
		AsyncDBExec(a.ConfigDB, "UPDATE providers SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ProviderID)
		a.logAnalytics("agent", agent.ID, totalTokensUsed, true)
		a.logAnalytics("provider", agent.ProviderID, totalTokensUsed, true)

		// --- Auto-generation routing ---
		// If agent's output methods don't include Text, route reply to generations API.
		var outputMethods []string
		json.Unmarshal([]byte(agent.OutputMethods), &outputMethods)
		hasTextOutput := false
		for _, m := range outputMethods {
			if m == "Text" || m == "MD" {
				hasTextOutput = true
				break
			}
		}
		if !hasTextOutput && len(outputMethods) > 0 {
			// Determine which generation API to call based on output methods.
			var prov Provider
			if err := a.ConfigDB.QueryRow("SELECT api_key, endpoint FROM providers WHERE id=?", agent.ProviderID).Scan(&prov.APIKey, &prov.Endpoint); err == nil {
				for _, m := range outputMethods {
					switch m {
					case "Image":
						executionLogs = append(executionLogs, "[Image Generation]: Generating image...")
						imgURL, _, imgErr := generateImage(replyContent, "dall-e-3", prov.APIKey, prov.Endpoint)
						if imgErr == nil && imgURL != "" {
							replyContent = imgURL
							executionLogs = append(executionLogs, "[Image Generation]: Success")
						} else {
							executionLogs = append(executionLogs, "[Image Generation Error]: "+imgErr.Error())
						}
					case "Video", "Audio":
						executionLogs = append(executionLogs, fmt.Sprintf("[Generation]: %s generation not yet implemented.", m))
					}
					break // Use the first matching generation type.
				}
			}
		}

		// Dispatch webhook outputs asynchronously.
		go a.dispatchWebhookOutputs(agent, replyContent)

		if useStream {
			sseEmit(map[string]interface{}{"done": true, "logs": executionLogs})
		} else {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"reply": replyContent, "logs": executionLogs})
		}
		return
	}

	// Loop exhausted.
	exhausted := "[SYSTEM] Agent has continuously repeated a command 5 times without resolution. Stopping."
	if useStream {
		sseEmit(map[string]interface{}{"done": true, "reply": exhausted, "logs": executionLogs})
	} else {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"reply": exhausted,
			"logs":  executionLogs,
		})
	}
}

// jsonStr is a helper to marshal a map to a compact JSON string, returning "{}" on error.
func jsonStr(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// callAgentSkill invokes a target agent synchronously (non-streaming) as a sub-call
// and returns its reply string. skillContent is the JSON stored in the skill's content
// field with the shape:
//
//	{"agent_id":"...", "system_prompt":"...(override)...", "user_prompt_prefix":"...(override)..."}
//
// The callerInput is the text passed by the calling agent to this skill as arguments.
func (a *App) callAgentSkill(skillContent string, callerInput string) string {
	var cfg struct {
		AgentID          string `json:"agent_id"`
		SystemPrompt     string `json:"system_prompt"`
		UserPromptPrefix string `json:"user_prompt_prefix"`
	}
	if err := json.Unmarshal([]byte(skillContent), &cfg); err != nil || cfg.AgentID == "" {
		return "[AGENT SKILL ERROR] Invalid config: 'agent_id' is required in skill content JSON."
	}

	// Load target agent from DB.
	var target Agent
	err := a.ConfigDB.QueryRow(
		"SELECT id, name, provider_id, model, sources, skills, outputs, mcps, projects, system_prompt, token_usage, input_methods, output_methods, user_prompt_prefix, temperature, max_tokens, top_p, privacy_enabled, can_create_skills, stream_enabled FROM agents WHERE id=?",
		cfg.AgentID,
	).Scan(
		&target.ID, &target.Name, &target.ProviderID, &target.Model,
		&target.Sources, &target.Skills, &target.Outputs, &target.MCPs, &target.Projects,
		&target.SystemPrompt, &target.TokenUsage, &target.InputMethods, &target.OutputMethods,
		&target.UserPromptPrefix, &target.Temperature, &target.MaxTokens, &target.TopP,
		&target.PrivacyEnabled, &target.CanCreateSkills, &target.StreamEnabled,
	)
	if err != nil {
		return "[AGENT SKILL ERROR] Target agent not found: " + cfg.AgentID
	}

	// Apply overrides from the skill config.
	if cfg.SystemPrompt != "" {
		target.SystemPrompt = cfg.SystemPrompt
	}
	if cfg.UserPromptPrefix != "" {
		target.UserPromptPrefix = cfg.UserPromptPrefix
	}

	// Build a synthetic non-streaming chat request for the target agent.
	chatPayload, _ := json.Marshal(map[string]interface{}{
		"agent_id": target.ID,
		"message":  callerInput,
		"stream":   false,
	})
	synReq, _ := http.NewRequest("POST", "/api/v1/chat", bytes.NewReader(chatPayload))
	synReq.Header.Set("Content-Type", "application/json")
	// Stamp the overrides as headers so handleChat can pick them up before
	// the DB agent is loaded — we use X-Override headers and handle them below.
	if cfg.SystemPrompt != "" {
		synReq.Header.Set("X-Agent-System-Prompt-Override", cfg.SystemPrompt)
	}
	if cfg.UserPromptPrefix != "" {
		synReq.Header.Set("X-Agent-User-Prompt-Prefix-Override", cfg.UserPromptPrefix)
	}

	rec := &responseRecorder{header: make(http.Header), code: http.StatusOK}
	a.handleChat(rec, synReq)

	// Parse the reply from the recorded response.
	var result map[string]interface{}
	if err := json.Unmarshal(rec.body, &result); err != nil {
		return "[AGENT SKILL ERROR] Failed to parse sub-agent response."
	}
	if errMsg, ok := result["error"].(string); ok && errMsg != "" {
		return "[AGENT SKILL ERROR] " + errMsg
	}
	reply, _ := result["reply"].(string)
	if reply == "" {
		return "[AGENT SKILL] Agent returned no content."
	}
	return reply
}

// executeCommand handles a single slash-command issued by the agent within the agentic loop.
// It is safe to call concurrently.
func (a *App) executeCommand(
	cLine string,
	agent Agent,
	projIDs []string,
	allowedSources, allowedRAGs, allowedPreferences, allowedDynamicTools map[string]bool,
	helpPanel string,
	executionLogs *[]string,
	logMu *sync.Mutex,
) string {
	parts := strings.Fields(cLine)
	if len(parts) == 0 {
		return ""
	}
	cmd := parts[0]

	appendLog := func(msg string) {
		logMu.Lock()
		*executionLogs = append(*executionLogs, msg)
		logMu.Unlock()
	}

	// Check project pipeline commands first.
	for _, pid := range projIDs {
		var pFlow string
		a.ConfigDB.QueryRow("SELECT flow_json FROM projects WHERE id=?", pid).Scan(&pFlow)
		if strings.Contains(pFlow, cmd) {
			appendLog("[Pipeline]: Running Visual Router [" + pid + "]...")
			return a.runProjectPipeline(pid, agent.ID, cLine)
		}
	}

	switch cmd {
	case "/help":
		return "[SYSTEM]\n" + helpPanel

	case "/nextPage":
		paginationMu.Lock()
		defer paginationMu.Unlock()
		chunks, exists := paginationStore[agent.ID]
		if !exists {
			return "[SYSTEM ERROR] No pagination active."
		}
		idx := paginationStep[agent.ID] + 1
		if idx >= len(chunks) {
			return "[SYSTEM RESULT] End of data."
		}
		paginationStep[agent.ID] = idx
		return fmt.Sprintf("%s\n\n[Page %d/%d, use /nextPage for more]", chunks[idx], idx+1, len(chunks))

	case "/getSourcesList":
		res := "[AVAILABLE SOURCES]\n"
		for s := range allowedSources {
			res += "- " + s + "\n"
		}
		return res

	case "/getSourceData":
		if len(parts) < 2 {
			return "[SYSTEM ERROR] Usage: /getSourceData <name>"
		}
		sName := parts[1]
		if !allowedSources[sName] {
			return "[SYSTEM ERROR] Unauthorized."
		}
		var sID, sType, sFile string
		var sPageSize int
		a.ConfigDB.QueryRow("SELECT id, type, page_size, file_path FROM sources WHERE name=?", sName).Scan(&sID, &sType, &sPageSize, &sFile)
		var finalData string
		if sType == "Local File" {
			if b, err := os.ReadFile(sFile); err != nil {
				finalData = "Error: " + err.Error()
			} else {
				finalData = string(b)
			}
		} else {
			var d, ts string
			if a.ConfigDB.QueryRow("SELECT data, timestamp FROM source_logs WHERE source_id=? ORDER BY id DESC LIMIT 1", sID).Scan(&d, &ts) != nil {
				finalData = "No data available."
			} else {
				finalData = fmt.Sprintf("[DATA AT %s]\n%s", ts, d)
			}
		}
		return triggerPagination(agent.ID, finalData, sPageSize)

	case "/getPrefList":
		res := "[PREFERENCES DATABASES]\n"
		for prefID := range allowedPreferences {
			var nm, ds string
			if a.ConfigDB.QueryRow("SELECT name, description FROM preferences WHERE id=?", prefID).Scan(&nm, &ds) == nil {
				res += fmt.Sprintf("- ID: %s | Name: %s\n", prefID, nm)
			}
		}
		return res

	case "/getPrefDataList", "/addPrefData", "/deletePrefData", "/editPrefData":
		if len(parts) < 2 {
			return "[SYSTEM ERROR] Missing <PrefID>"
		}
		prefID := parts[1]
		if !allowedPreferences[prefID] {
			return "[SYSTEM ERROR] Unauthorized."
		}
		var rawTName string
		a.ConfigDB.QueryRow("SELECT table_name FROM preferences WHERE id=?", prefID).Scan(&rawTName)
		if rawTName == "" {
			return "[SYSTEM ERROR] Preference DB not found."
		}
		tName, tErr := sanitizeTableName(rawTName)
		if tErr != nil {
			log.Printf("[Security] Pref table name failed sanitization: %v", tErr)
			return "[SYSTEM ERROR] Invalid preference table."
		}
		switch cmd {
		case "/getPrefDataList":
			rows, err := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data FROM %s", tName))
			res := "[PREFERENCE RECORDS]\n"
			if err == nil {
				defer rows.Close()
				for rows.Next() {
					var rn, rd string
					rows.Scan(&rn, &rd)
					res += fmt.Sprintf("- %s: %s\n", rn, rd)
				}
			}
			return res
		case "/addPrefData":
			if len(parts) < 4 {
				return "[SYSTEM ERROR] Usage: /addPrefData <id> <name> <data>"
			}
			if _, err := a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data) VALUES (?, ?)", tName), parts[2], strings.Join(parts[3:], " ")); err != nil {
				return "[RESULT] Error: " + err.Error()
			}
			return "[RESULT] Record Added."
		case "/editPrefData":
			if len(parts) < 4 {
				return "[SYSTEM ERROR] Usage: /editPrefData <id> <name> <data>"
			}
			if _, err := a.MemoryDB.Exec(fmt.Sprintf("UPDATE %s SET data=? WHERE record_name=?", tName), strings.Join(parts[3:], " "), parts[2]); err != nil {
				return "[RESULT] Error: " + err.Error()
			}
			return "[RESULT] Record Updated."
		case "/deletePrefData":
			if len(parts) < 3 {
				return "[SYSTEM ERROR] Usage: /deletePrefData <id> <name>"
			}
			a.MemoryDB.Exec(fmt.Sprintf("DELETE FROM %s WHERE record_name=?", tName), parts[2])
			return "[RESULT] Record Deleted."
		}

	case "/getRAGList":
		res := "[RAG DATABASES]\n"
		for ragID := range allowedRAGs {
			var nm, ds string
			if a.ConfigDB.QueryRow("SELECT name, description FROM rags WHERE id=?", ragID).Scan(&nm, &ds) == nil {
				res += fmt.Sprintf("- ID: %s | Name: %s\n", ragID, nm)
			}
		}
		return res

	case "/getRAGDataList", "/addRAGData", "/deleteRAGData", "/editRAGData":
		if len(parts) < 2 {
			return "[SYSTEM ERROR] Missing <RAGID>"
		}
		ragID := parts[1]
		if !allowedRAGs[ragID] {
			return "[SYSTEM ERROR] Unauthorized."
		}
		var rawTNameRAG string
		a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&rawTNameRAG)
		if rawTNameRAG == "" {
			return "[SYSTEM ERROR] RAG not found."
		}
		tName, tErr := sanitizeTableName(rawTNameRAG)
		if tErr != nil {
			log.Printf("[Security] RAG table name failed sanitization: %v", tErr)
			return "[SYSTEM ERROR] Invalid RAG table."
		}
		switch cmd {
		case "/getRAGDataList":
			rows, err := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data FROM %s", tName))
			res := "[RAG RECORDS]\n"
			if err == nil {
				defer rows.Close()
				for rows.Next() {
					var rn, rd string
					rows.Scan(&rn, &rd)
					res += fmt.Sprintf("- %s: %s\n", rn, rd)
				}
			}
			return res
		case "/addRAGData":
			if len(parts) < 4 {
				return "[SYSTEM ERROR] Usage: /addRAGData <id> <name> <data>"
			}
			ragDataStr := strings.Join(parts[3:], " ")

			// Check if chunking is enabled.
			var chunkingEnabled string
			a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='rag_chunking'").Scan(&chunkingEnabled)
			if chunkingEnabled == "true" {
				var chunkSizeStr, chunkOverlapStr string
				a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='rag_chunk_size'").Scan(&chunkSizeStr)
				a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='rag_chunk_overlap'").Scan(&chunkOverlapStr)
				chunkSize := 500
				chunkOverlap := 50
				if v, err := strconv.Atoi(chunkSizeStr); err == nil && v > 0 {
					chunkSize = v
				}
				if v, err := strconv.Atoi(chunkOverlapStr); err == nil && v >= 0 {
					chunkOverlap = v
				}
				chunks := chunkText(ragDataStr, chunkSize, chunkOverlap)
				added := 0
				for i, chunk := range chunks {
					chunkName := fmt.Sprintf("%s_chunk_%d", parts[2], i)
					vector := getEmbedding(chunk, a.ConfigDB)
					vectorJSON := "[]"
					if vector != nil {
						if vBytes, err := json.Marshal(vector); err == nil {
							vectorJSON = string(vBytes)
						}
					}
					if _, err := a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), chunkName, chunk, vectorJSON); err != nil {
						if strings.Contains(err.Error(), "has no column named vector") {
							a.MemoryDB.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN vector TEXT DEFAULT '[]'", tName))
							a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), chunkName, chunk, vectorJSON)
						}
					}
					added++
				}
				return fmt.Sprintf("[RESULT] Added %d chunks from '%s'.", added, parts[2])
			}

			vector := getEmbedding(ragDataStr, a.ConfigDB)
			vectorJSON := "[]"
			if vector != nil {
				if vBytes, err := json.Marshal(vector); err == nil {
					vectorJSON = string(vBytes)
				}
			}
			_, err := a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), parts[2], ragDataStr, vectorJSON)
			if err != nil && strings.Contains(err.Error(), "has no column named vector") {
				a.MemoryDB.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN vector TEXT DEFAULT '[]'", tName))
				_, err = a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), parts[2], ragDataStr, vectorJSON)
			}
			if err != nil {
				return "[RESULT] Error: " + err.Error()
			}
			return "[RESULT] Record & Vector Embedding Added."
		case "/editRAGData":
			if len(parts) < 4 {
				return "[SYSTEM ERROR] Usage: /editRAGData <id> <name> <data>"
			}
			newData := strings.Join(parts[3:], " ")
			// Re-embed updated text so similarity search stays accurate.
			newVector := getEmbedding(newData, a.ConfigDB)
			newVectorJSON := "[]"
			if newVector != nil {
				if vb, e := json.Marshal(newVector); e == nil {
					newVectorJSON = string(vb)
				}
			}
			if _, err := a.MemoryDB.Exec(fmt.Sprintf("UPDATE %s SET data=?, vector=? WHERE record_name=?", tName), newData, newVectorJSON, parts[2]); err != nil {
				return "[RESULT] Error: " + err.Error()
			}
			return "[RESULT] Record & Vector Updated."
		case "/deleteRAGData":
			if len(parts) < 3 {
				return "[SYSTEM ERROR] Usage: /deleteRAGData <id> <name>"
			}
			a.MemoryDB.Exec(fmt.Sprintf("DELETE FROM %s WHERE record_name=?", tName), parts[2])
			return "[RESULT] Record Deleted."
		}

	case "/createSkill":
		if !agent.CanCreateSkills {
			return "[SYSTEM ERROR] Unauthorized. You do not have permission to create skills."
		}
		if len(parts) < 3 {
			return "[SYSTEM ERROR] Usage: /createSkill <name> <description>"
		}
		skillEditorMu.Lock()
		skillEditorStore[agent.ID] = map[string]string{
			"name": parts[1],
			"desc": strings.Join(parts[2:], " "),
		}
		skillEditorMu.Unlock()
		return "[SYSTEM] ENTERING NATIVE GO CODE EDITOR MODE.\nYour next reply MUST contain ONLY valid Go code. Reply /cancel to abort."

	case "/createTasks":
		p := strings.SplitN(cLine, " ", 4)
		if len(p) < 4 {
			return "[SYSTEM ERROR] Usage: /createTasks <regex> <command> <true/false>"
		}
		repeat := 0
		if p[3] == "true" {
			repeat = 1
		}
		AsyncDBExec(a.ConfigDB, "INSERT INTO tasks (id, agent_id, regex, command, repeat) VALUES (?, ?, ?, ?, ?)",
			"tsk_"+fmt.Sprint(time.Now().UnixNano()), agent.ID, p[1], p[2], repeat)
		return "[TASK SYSTEM] Background Job Deployed. Monitoring pattern: " + p[1]

	default:
		// MCP commands.
		if strings.HasPrefix(cmd, "/mcp_") {
			mcpName := strings.TrimPrefix(cmd, "/mcp_")
			var mURL, mCmd string
			if err := a.ConfigDB.QueryRow("SELECT url, command FROM mcp_servers WHERE name=?", mcpName).Scan(&mURL, &mCmd); err != nil {
				return "[MCP ERROR] Server not configured: " + mcpName
			}
			toolName := "default_tool"
			if len(parts) > 1 {
				toolName = parts[1]
			}
			argsData := map[string]interface{}{}
			if len(parts) > 2 {
				rawArgs := strings.Join(parts[2:], " ")
				if json.Unmarshal([]byte(rawArgs), &argsData) != nil {
					argsData["query"] = rawArgs
				}
			}
			requestPayload := map[string]interface{}{
				"jsonrpc": "2.0",
				"id":      time.Now().UnixNano(),
				"method":  "tools/call",
				"params":  map[string]interface{}{"name": toolName, "arguments": argsData},
			}
			reqBytes, _ := json.Marshal(requestPayload)
			if mURL != "" {
				apiR, _ := http.NewRequest("POST", mURL, bytes.NewBuffer(reqBytes))
				apiR.Header.Set("Content-Type", "application/json")
				resp, e := (&http.Client{Timeout: 30 * time.Second}).Do(apiR)
				if e != nil {
					return "[MCP HTTP ERROR] " + e.Error()
				}
				b, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				return fmt.Sprintf("[MCP %s Response]\n%s", mcpName, string(b))
			} else if mCmd != "" {
				// Parse command into executable + args without shell interpretation
				// to prevent command injection via poisoned DB values.
				parts := strings.Fields(mCmd)
				if len(parts) == 0 {
					return "[MCP ERROR] Empty command for " + mcpName
				}
				execCmd := exec.Command(parts[0], parts[1:]...)
				stdin, _ := execCmd.StdinPipe()
				stdout, _ := execCmd.StdoutPipe()
				execCmd.Start()
				stdin.Write(append(reqBytes, '\n'))
				stdin.Close()
				outBytes, e := io.ReadAll(stdout)
				execCmd.Wait()
				if e != nil {
					return "[MCP STDIO ERROR] " + e.Error()
				}
				return fmt.Sprintf("[MCP %s Response]\n%s", mcpName, string(outBytes))
			}
			return "[MCP ERROR] Both URL and Command are empty for " + mcpName
		}

		// Dynamic skill commands.
		toolName := strings.TrimPrefix(cmd, "/")
		if allowedDynamicTools[toolName] {
			agentArgs := parts[1:]
			var sType, sContent, sMethod, sUrl, sHeaders, sBody string
			var useDocker bool
			if err := a.ConfigDB.QueryRow("SELECT type, content, api_method, api_url, api_headers, api_body, COALESCE(use_docker,0) FROM skills WHERE name=?", toolName).
				Scan(&sType, &sContent, &sMethod, &sUrl, &sHeaders, &sBody, &useDocker); err != nil {
				return "[SKILL ERROR] Skill not found."
			}
			switch sType {
			case "Agent":
				// Call a sub-agent synchronously; agentArgs become the input text.
				input := strings.Join(agentArgs, " ")
				// Check if skill config has return_to: "user" to send result directly to user.
				var skillCfg struct {
					ReturnTo string `json:"return_to"`
				}
				json.Unmarshal([]byte(sContent), &skillCfg)
				result := a.callAgentSkill(sContent, input)
				if skillCfg.ReturnTo == "user" {
					// Prefix with marker so handleChat knows to send this directly to user.
					return "[USER_REPLY]" + result
				}
				return result
			case "MD":
				return "[INSTRUCTION]\n" + sContent
			case "Placeholder":
				// Placeholder skills don't execute anything — the command is returned as-is
				// so the agent treats it as a normal response to show to the user.
				return cLine
			case "Bash", "Go", "JavaScript":
				fCode := sContent
				ph := extractPlaceholders(sContent)
				for i, arg := range agentArgs {
					if i < len(ph) {
						fCode = strings.ReplaceAll(fCode, ph[i], arg)
					}
				}
				var result string
				if useDocker {
					dockerOut, dockerErr := executeInDocker(a.ConfigDB, fCode, sType, agentArgs)
					if dockerErr != nil {
						result = "[DOCKER ERROR] " + dockerErr.Error()
						if dockerOut != "" {
							result += "\nOutput: " + dockerOut
						}
					} else {
						result = "[RESULT (Docker)]\n" + dockerOut
					}
				} else {
					var execCmd *exec.Cmd
					var tmpExt string
					switch sType {
					case "Bash":
						tmpExt = ".sh"
					case "JavaScript":
						tmpExt = ".js"
					default:
						tmpExt = ".go"
					}
					tmp := filepath.Join(os.TempDir(), fmt.Sprintf("sk_%d%s", time.Now().UnixNano(), tmpExt))
					if err := os.WriteFile(tmp, []byte(fCode), 0700); err != nil {
						return "[SKILL ERROR] failed to write temp script"
					}
					defer os.Remove(tmp)
					switch sType {
					case "Bash":
						execCmd = exec.Command("sh", tmp)
					case "JavaScript":
						execCmd = exec.Command("node", tmp)
					default:
						execCmd = exec.Command("go", "run", tmp)
					}
					out, e := execCmd.CombinedOutput()
					result = "[RESULT]\n" + string(out)
					if e != nil {
						result += "\nErr: " + e.Error()
					}
				}
				return result
			case "API":
				fUrl, fHead, fBody := sUrl, sHeaders, sBody
				ph := extractPlaceholders(sUrl + sHeaders + sBody)
				for i, arg := range agentArgs {
					if i < len(ph) {
						fUrl = strings.ReplaceAll(fUrl, ph[i], arg)
						fHead = strings.ReplaceAll(fHead, ph[i], arg)
						fBody = strings.ReplaceAll(fBody, ph[i], arg)
					}
				}
				// SSRF guard: block requests to private/loopback addresses.
				if isPrivateHost(fUrl) {
					return "[API ERROR] URL targets a private/loopback address and is blocked."
				}
				apiR, _ := http.NewRequest(sMethod, fUrl, bytes.NewBuffer([]byte(fBody)))
				var hm map[string]string
				if json.Unmarshal([]byte(fHead), &hm) == nil {
					for k, v := range hm {
						apiR.Header.Set(k, v)
					}
				}
				extResp, e := (&http.Client{Timeout: 15 * time.Second}).Do(apiR)
				if e != nil {
					return "[API ERROR] " + e.Error()
				}
				b, _ := io.ReadAll(extResp.Body)
				extResp.Body.Close()
				return "[API RESULT]\n" + string(b)
			}
		}
		return "[SYSTEM] Command not found. Use /help"
	}
	return ""
}

// handleFileUpload processes multipart payload files securely onto disk storage.
func (a *App) handleFileUpload(w http.ResponseWriter, r *http.Request) {
	// Limit upload to 64 MB.
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, `{"error":"invalid request or file too large"}`, http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, `{"error":"invalid file"}`, http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Sanitize agentID: only allow alphanumerics, hyphens, and underscores.
	agentID := r.FormValue("agent_id")
	if agentID == "" {
		agentID = "global_temp"
	}
	if !regexp.MustCompile(`^[a-zA-Z0-9_\-]+$`).MatchString(agentID) {
		http.Error(w, `{"error":"invalid agent_id"}`, http.StatusBadRequest)
		return
	}

	// Extension allowlist — reject anything not in the approved set.
	safeBase := filepath.Base(filepath.Clean(header.Filename))
	ext := strings.ToLower(filepath.Ext(safeBase))
	if !allowedUploadExts[ext] {
		http.Error(w, `{"error":"file type not allowed"}`, http.StatusUnsupportedMediaType)
		return
	}

	// Read the first 512 bytes for magic-byte MIME detection.
	buf := make([]byte, 512)
	n, _ := file.Read(buf)
	detectedMIME := http.DetectContentType(buf[:n])

	// Map allowed MIME prefixes to their permitted extensions to catch disguised files
	// (e.g. an .jpg that is actually a ZIP). Only allow content that truly matches.
	allowedMIMEForExt := map[string]string{
		".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
		".webp": "image/", ".gif": "image/gif",
		".pdf": "application/pdf",
		".txt": "text/plain", ".md": "text/plain", ".csv": "text/plain",
		".json": "text/plain", ".yaml": "text/plain", ".yml": "text/plain",
		".html": "text/html",
		".mp3":  "audio/", ".wav": "audio/",
		".mp4": "video/",
	}
	if expectedMIMEPrefix, ok := allowedMIMEForExt[ext]; ok {
		if !strings.HasPrefix(detectedMIME, expectedMIMEPrefix) {
			// Plain text files often detect as text/plain regardless of extension — allow that.
			if expectedMIMEPrefix != "text/plain" || !strings.HasPrefix(detectedMIME, "text/") {
				http.Error(w, `{"error":"file content does not match declared extension"}`, http.StatusUnsupportedMediaType)
				return
			}
		}
	}

	// Rewind to start so the full file is written to disk.
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		// Seek may not be available on all multipart readers; fall back to prepending.
	}

	fileName := fmt.Sprintf("%s___%d___%s", agentID, time.Now().UnixNano(), safeBase)
	uploadsDir, _ := filepath.Abs("uploads")
	path := filepath.Join(uploadsDir, fileName)

	// Confirm the resolved path is still inside the uploads directory.
	if !strings.HasPrefix(path, uploadsDir+string(filepath.Separator)) {
		http.Error(w, `{"error":"invalid file path"}`, http.StatusBadRequest)
		return
	}

	dst, err := os.Create(path)
	if err != nil {
		http.Error(w, `{"error":"failed to save file"}`, http.StatusInternalServerError)
		return
	}
	defer dst.Close()
	// Write the already-read 512 bytes first, then stream the remainder.
	dst.Write(buf[:n])
	io.Copy(dst, file)

	// Return a relative path for portability.
	relPath := filepath.Join("uploads", fileName)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"path": relPath, "status": "ok"})
}

// autoMigrateColumn safely ensures column schema synchronization without truncating data.
func autoMigrateColumn(db *sql.DB, table string, column string, colDef string) {
	var name string
	err := db.QueryRow(fmt.Sprintf("SELECT name FROM pragma_table_info('%s') WHERE name='%s'", table, column)).Scan(&name)
	if err != nil {
		if _, execErr := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, colDef)); execErr == nil {
			fmt.Printf("[DB Migration]: Upgraded table '%s' -> Added column '%s'\n", table, column)
		} else {
			fmt.Printf("[DB Migration Error]: %v\n", execErr)
		}
	}
}

// handleGetChatHistory retrieves conversation history for a specific agent, optionally filtered by session.
func (a *App) handleGetChatHistory(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	sessionID := r.URL.Query().Get("session_id")
	var rows *sql.Rows
	var err error
	if sessionID != "" {
		rows, err = a.ConfigDB.Query("SELECT id, role, content, timestamp FROM chat_history WHERE agent_id=? AND session_id=? ORDER BY id ASC", agentID, sessionID)
	} else {
		rows, err = a.ConfigDB.Query("SELECT id, role, content, timestamp FROM chat_history WHERE agent_id=? AND (session_id='' OR session_id IS NULL) ORDER BY id ASC", agentID)
	}
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]ChatMessage{})
		return
	}
	defer rows.Close()
	var list []ChatMessage
	for rows.Next() {
		var m ChatMessage
		m.AgentID = agentID
		m.SessionID = sessionID
		rows.Scan(&m.ID, &m.Role, &m.Content, &m.Timestamp)
		list = append(list, m)
	}
	if list == nil {
		list = []ChatMessage{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

// handleClearChatHistory purges memory allocations assigned to a specific agent entity.
func (a *App) handleClearChatHistory(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	a.ConfigDB.Exec("DELETE FROM chat_history WHERE agent_id=?", agentID)

	paginationMu.Lock()
	delete(paginationStore, agentID)
	paginationMu.Unlock()

	if files, err := os.ReadDir("./uploads"); err == nil {
		prefix := agentID + "___"
		for _, f := range files {
			if !f.IsDir() && strings.HasPrefix(f.Name(), prefix) {
				os.Remove(filepath.Join("./uploads", f.Name()))
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleExportChatHistory exports conversation history as JSON or Markdown.
func (a *App) handleExportChatHistory(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}
	rows, err := a.ConfigDB.Query("SELECT role, content, timestamp FROM chat_history WHERE agent_id=? ORDER BY id ASC", agentID)
	if err != nil {
		http.Error(w, `{"error":"db error"}`, http.StatusInternalServerError)
		return
	}
	defer rows.Close()
	type msg struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Timestamp string `json:"timestamp"`
	}
	var messages []msg
	for rows.Next() {
		var m msg
		rows.Scan(&m.Role, &m.Content, &m.Timestamp)
		messages = append(messages, m)
	}
	if messages == nil {
		messages = []msg{}
	}

	if format == "markdown" {
		var md strings.Builder
		fmt.Fprintf(&md, "# Chat History — Agent: %s\n\n", agentID)
		for _, m := range messages {
			role := strings.Title(m.Role)
			fmt.Fprintf(&md, "### %s (%s)\n\n%s\n\n---\n\n", role, m.Timestamp, m.Content)
		}
		w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="chat_%s.md"`, agentID))
		w.Write([]byte(md.String()))
		return
	}

	out, _ := json.MarshalIndent(messages, "", "  ")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="chat_%s.json"`, agentID))
	w.Write(out)
}

func (a *App) handleGetSessions(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	rows, err := a.ConfigDB.Query("SELECT id, agent_id, name, created_at, updated_at FROM chat_sessions WHERE agent_id=? ORDER BY updated_at DESC", agentID)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]ChatSession{})
		return
	}
	defer rows.Close()
	var list []ChatSession
	for rows.Next() {
		var s ChatSession
		rows.Scan(&s.ID, &s.AgentID, &s.Name, &s.CreatedAt, &s.UpdatedAt)
		list = append(list, s)
	}
	if list == nil {
		list = []ChatSession{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	var req struct {
		Name string `json:"name"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Name == "" {
		req.Name = time.Now().Format("2006-01-02 15:04")
	}
	sessionID := fmt.Sprintf("sess_%d", time.Now().UnixNano())
	now := time.Now().Format(time.RFC3339)
	a.ConfigDB.Exec("INSERT INTO chat_sessions (id, agent_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", sessionID, agentID, req.Name, now, now)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": sessionID, "name": req.Name})
}

func (a *App) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("session_id")
	a.ConfigDB.Exec("DELETE FROM chat_history WHERE session_id=?", sessionID)
	a.ConfigDB.Exec("DELETE FROM chat_sessions WHERE id=?", sessionID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetTasks(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, agent_id, regex, command, COALESCE(repeat,0), COALESCE(active,1) FROM tasks ORDER BY id DESC")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Task{})
		return
	}
	defer rows.Close()
	var list []Task
	for rows.Next() {
		var t Task
		var repeat, active int
		rows.Scan(&t.ID, &t.AgentID, &t.Regex, &t.Command, &repeat, &active)
		t.Repeat = repeat == 1
		t.Active = active == 1
		list = append(list, t)
	}
	if list == nil {
		list = []Task{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateTask(w http.ResponseWriter, r *http.Request) {
	var t Task
	json.NewDecoder(r.Body).Decode(&t)
	if t.ID == "" {
		t.ID = fmt.Sprintf("tsk_%d", time.Now().UnixNano())
	}
	repeat := 0
	if t.Repeat {
		repeat = 1
	}
	active := 1
	if !t.Active {
		active = 0
	}
	a.ConfigDB.Exec("INSERT INTO tasks (id, agent_id, regex, command, repeat, active) VALUES (?, ?, ?, ?, ?, ?)", t.ID, t.AgentID, t.Regex, t.Command, repeat, active)
	a.appendSecurityEvent("task_create", r.URL.Path, "admin", fmt.Sprintf("created task %s for agent %s", t.ID, t.AgentID))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": t.ID})
}

func (a *App) handleUpdateTask(w http.ResponseWriter, r *http.Request) {
	var t Task
	json.NewDecoder(r.Body).Decode(&t)
	repeat := 0
	if t.Repeat {
		repeat = 1
	}
	active := 1
	if !t.Active {
		active = 0
	}
	a.ConfigDB.Exec("UPDATE tasks SET agent_id=?, regex=?, command=?, repeat=?, active=? WHERE id=?", t.AgentID, t.Regex, t.Command, repeat, active, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteTask(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM tasks WHERE id=?", r.PathValue("id"))
	a.appendSecurityEvent("task_delete", r.URL.Path, "admin", fmt.Sprintf("deleted task %s", r.PathValue("id")))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetStats(w http.ResponseWriter, r *http.Request) {
	var ag, sk, rg, tk int
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM agents").Scan(&ag)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM skills").Scan(&sk)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM rags").Scan(&rg)
	a.ConfigDB.QueryRow("SELECT COALESCE(SUM(token_usage), 0) FROM providers").Scan(&tk)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"agents": ag, "skills": sk, "rags": rg, "tokens": tk})
}

func maskSecret(secret string) string {
	if strings.TrimSpace(secret) == "" {
		return ""
	}
	// Mask short secrets completely to avoid leaking length information.
	if len(secret) <= 12 {
		return strings.Repeat("•", 8)
	}
	return secret[:4] + strings.Repeat("•", len(secret)-8) + secret[len(secret)-4:]
}

// buildBackupData collects all database tables into a map for backup/restore.
func buildBackupData(app *App) map[string]interface{} {
	backup := map[string]interface{}{}
	tables := []string{"agents", "skills", "sources", "projects", "rags", "outputs", "mcp_servers", "tasks", "api_keys", "settings", "providers"}
	for _, t := range tables {
		rows, err := app.ConfigDB.Query("SELECT * FROM " + t)
		if err != nil {
			continue
		}
		cols, _ := rows.Columns()
		var rowsData []map[string]interface{}
		for rows.Next() {
			values := make([]interface{}, len(cols))
			ptrs := make([]interface{}, len(cols))
			for i := range values {
				ptrs[i] = &values[i]
			}
			rows.Scan(ptrs...)
			row := make(map[string]interface{})
			for i, col := range cols {
				v := values[i]
				if b, ok := v.([]byte); ok {
					row[col] = string(b)
				} else {
					row[col] = v
				}
			}
			rowsData = append(rowsData, row)
		}
		rows.Close()
		backup[t] = rowsData
	}
	backup["_meta"] = map[string]interface{}{
		"version":    CurrentVersion,
		"exported_at": time.Now().UTC().Format(time.RFC3339),
	}
	return backup
}

// encryptBackup encrypts JSON data with a password using AES-256-GCM + PBKDF2.
func encryptBackup(data []byte, password string) ([]byte, error) {
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return nil, err
	}
	key := deriveKey(password, salt)
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, aesGCM.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	ciphertext := aesGCM.Seal(nil, nonce, data, nil)
	// Format: salt(16) + nonce(12) + ciphertext
	result := make([]byte, 0, 16+12+len(ciphertext))
	result = append(result, salt...)
	result = append(result, nonce...)
	result = append(result, ciphertext...)
	return result, nil
}

// decryptBackup decrypts AES-256-GCM encrypted data with a password.
func decryptBackup(data []byte, password string) ([]byte, error) {
	if len(data) < 28 { // 16 (salt) + 12 (nonce) minimum
		return nil, fmt.Errorf("invalid encrypted data")
	}
	salt := data[:16]
	nonce := data[16:28]
	ciphertext := data[28:]
	key := deriveKey(password, salt)
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	plaintext, err := aesGCM.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed (wrong password?)")
	}
	return plaintext, nil
}

// deriveKey derives a 32-byte AES key from a password using PBKDF2-SHA256.
func deriveKey(password string, salt []byte) []byte {
	return pbkdf2.Key([]byte(password), salt, 100000, 32, sha256.New)
}

// Master key for encrypting API keys at rest.
// Loaded from ZUVER_MASTER_KEY env var, or auto-generated and stored in .zuver_master_key.
var masterKey []byte

func loadOrCreateMasterKey() {
	// Try environment variable first.
	if envKey := os.Getenv("ZUVER_MASTER_KEY"); envKey != "" {
		// Derive a 32-byte key from the env var.
		masterKey = deriveKey(envKey, []byte("zuver-master-key-salt"))
		log.Printf("[Security] Master key loaded from ZUVER_MASTER_KEY environment variable.")
		return
	}
	// Try loading from file.
	keyFile := ".zuver_master_key"
	if data, err := os.ReadFile(keyFile); err == nil && len(data) == 64 {
		// File contains hex-encoded 32-byte key.
		key, err := hex.DecodeString(string(data))
		if err == nil && len(key) == 32 {
			masterKey = key
			log.Printf("[Security] Master key loaded from %s.", keyFile)
			return
		}
	}
	// Generate new master key and save to file.
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		log.Printf("[Security] Failed to generate master key: %v. API keys will not be encrypted.", err)
		return
	}
	masterKey = b
	if err := os.WriteFile(keyFile, []byte(hex.EncodeToString(b)), 0600); err != nil {
		log.Printf("[Security] Failed to save master key to %s: %v", keyFile, err)
	} else {
		log.Printf("[Security] New master key generated and saved to %s. KEEP THIS FILE SAFE!", keyFile)
	}
}

// encryptSecret encrypts a string with the master key using AES-256-GCM.
// Returns base64-encoded ciphertext. Returns plaintext if master key is not available.
func encryptSecret(plaintext string) string {
	if len(masterKey) == 0 || plaintext == "" {
		return plaintext
	}
	salt := make([]byte, 16)
	rand.Read(salt)
	block, err := aes.NewCipher(masterKey)
	if err != nil {
		return plaintext
	}
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return plaintext
	}
	nonce := make([]byte, aesGCM.NonceSize())
	rand.Read(nonce)
	ciphertext := aesGCM.Seal(nil, nonce, []byte(plaintext), nil)
	// Format: base64(salt + nonce + ciphertext)
	result := make([]byte, 0, 16+12+len(ciphertext))
	result = append(result, salt...)
	result = append(result, nonce...)
	result = append(result, ciphertext...)
	return base64.StdEncoding.EncodeToString(result)
}

// decryptSecret decrypts a base64-encoded ciphertext with the master key.
// Returns plaintext. Returns the input unchanged if master key is not available or decryption fails.
func decryptSecret(ciphertext string) string {
	if len(masterKey) == 0 || ciphertext == "" {
		return ciphertext
	}
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil || len(data) < 28 {
		return ciphertext // Not encrypted or invalid, return as-is.
	}
	salt := data[:16]
	nonce := data[16:28]
	encrypted := data[28:]
	block, err := aes.NewCipher(masterKey)
	if err != nil {
		return ciphertext
	}
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return ciphertext
	}
	plaintext, err := aesGCM.Open(nil, nonce, encrypted, salt)
	if err != nil {
		return ciphertext // Decryption failed, return as-is (legacy plaintext).
	}
	return string(plaintext)
}

// isEncrypted checks if a string looks like base64-encoded encrypted data.
func isEncrypted(s string) bool {
	if len(s) < 50 {
		return false
	}
	_, err := base64.StdEncoding.DecodeString(s)
	return err == nil && len(s) > 40
}

// TOTP functions (RFC 6238)
const totpPeriod = 30 // seconds
const totpDigits = 6

// generateTOTPSecret generates a random base32-encoded TOTP secret.
func generateTOTPSecret() string {
	b := make([]byte, 20) // 160 bits
	rand.Read(b)
	return base32Encode(b)
}

// base32Encode encodes bytes to RFC 4648 base32 without padding.
func base32Encode(data []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
	var result []byte
	for i := 0; i < len(data); i += 5 {
		// Read up to 5 bytes into a 40-bit buffer.
		var buf [5]byte
		n := copy(buf[:], data[i:])
		var val uint64
		for j := 0; j < n; j++ {
			val = (val << 8) | uint64(buf[j])
		}
		// Pad remaining bytes with zeros if less than 5.
		for j := n; j < 5; j++ {
			val <<= 8
		}
		// Extract 8 base32 characters (5 bits each).
		for j := 0; j < 8; j++ {
			idx := val >> (35 - j*5) & 31
			result = append(result, alphabet[idx])
		}
	}
	return string(result)
}

// verifyTOTP verifies a TOTP code against a secret. Returns true if valid.
func verifyTOTP(secret, code string) bool {
	if len(code) != totpDigits {
		return false
	}
	secretBytes, err := base32Decode(secret)
	if err != nil {
		return false
	}
	// Check current time window and ±1 window for clock skew.
	now := time.Now().Unix()
	for _, offset := range []int64{-1, 0, 1} {
		counter := uint64((now / totpPeriod) + offset)
		if generateTOTPCode(secretBytes, counter) == code {
			return true
		}
	}
	return false
}

// generateTOTPCode generates a 6-digit TOTP code for a given counter.
func generateTOTPCode(secret []byte, counter uint64) string {
	buf := make([]byte, 8)
	buf[0] = byte(counter >> 56)
	buf[1] = byte(counter >> 48)
	buf[2] = byte(counter >> 40)
	buf[3] = byte(counter >> 32)
	buf[4] = byte(counter >> 24)
	buf[5] = byte(counter >> 16)
	buf[6] = byte(counter >> 8)
	buf[7] = byte(counter)

	// TOTP RFC 6238 uses HMAC-SHA1.
	mac := hmac.New(sha1.New, secret)
	mac.Write(buf)
	hash := mac.Sum(nil)

	offset := hash[len(hash)-1] & 0x0f
	code := uint32(hash[offset])&0x7f<<24 | uint32(hash[offset+1])&0xff<<16 | uint32(hash[offset+2])&0xff<<8 | uint32(hash[offset+3])&0xff
	code = code % 1000000

	return fmt.Sprintf("%06d", code)
}

// base32Decode decodes an RFC 4648 base32 string to bytes.
func base32Decode(s string) ([]byte, error) {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
	s = strings.ToUpper(strings.TrimSpace(s))
	s = strings.TrimRight(s, "=")
	if len(s) == 0 {
		return nil, fmt.Errorf("empty base32 string")
	}
	// Process in groups of 8 characters (= 5 bytes).
	var result []byte
	for i := 0; i < len(s); i += 8 {
		chunk := s[i:]
		if len(chunk) > 8 {
			chunk = chunk[:8]
		}
		var val uint64
		for j, c := range chunk {
			idx := strings.IndexByte(alphabet, byte(c))
			if idx < 0 {
				return nil, fmt.Errorf("invalid base32 character: %c", c)
			}
			val = val<<5 | uint64(idx)
			_ = j
		}
		// Pad short chunks: shift left to align.
		if len(chunk) < 8 {
			val <<= uint((8 - len(chunk)) * 5)
		}
		// Extract up to 5 bytes.
		for j := 0; j < 5; j++ {
			b := byte(val >> (32 - j*8))
			// Only append bytes that were actually encoded.
			if j < len(chunk)*5/8 || (len(chunk)*5)%8 != 0 && j == len(chunk)*5/8 {
				result = append(result, b)
			} else if len(chunk) == 8 {
				result = append(result, b)
			}
		}
	}
	return result, nil
}

// getTOTPQRCodeURL generates an otpauth:// URI for QR code generation.
func getTOTPQRCodeURL(secret, account string) string {
	return fmt.Sprintf("otpauth://totp/Zuver:%s?secret=%s&issuer=Zuver&digits=%d&period=%d",
		account, secret, totpDigits, totpPeriod)
}

func (a *App) handleGetProviders(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, COALESCE(type,'OpenAI'), endpoint, api_key, COALESCE(extra_config,'{}'), COALESCE(models,'[]'), token_usage FROM providers")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Provider{})
		return
	}
	defer rows.Close()
	var list []Provider
	for rows.Next() {
		var i Provider
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.Endpoint, &i.APIKey, &i.ExtraConfig, &i.Models, &i.TokenUsage)
		i.APIKey = maskSecret(decryptSecret(i.APIKey))
		list = append(list, i)
	}
	if list == nil {
		list = []Provider{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateProvider(w http.ResponseWriter, r *http.Request) {
	var i Provider
	json.NewDecoder(r.Body).Decode(&i)
	if i.Models == "" {
		i.Models = "[]"
	}
	SyncDBExec(a.ConfigDB, "INSERT INTO providers (id, name, type, endpoint, api_key, extra_config, models) VALUES (?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Endpoint, encryptSecret(i.APIKey), i.ExtraConfig, i.Models)
	a.appendSecurityEvent("provider_create", r.URL.Path, "admin", fmt.Sprintf("created provider %s (%s)", i.ID, i.Name))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateProvider(w http.ResponseWriter, r *http.Request) {
	var i Provider
	json.NewDecoder(r.Body).Decode(&i)
	if i.Models == "" {
		i.Models = "[]"
	}
	if strings.TrimSpace(i.APIKey) == "" {
		a.ConfigDB.Exec("UPDATE providers SET name=?, type=?, endpoint=?, extra_config=?, models=? WHERE id=?", i.Name, i.Type, i.Endpoint, i.ExtraConfig, i.Models, r.PathValue("id"))
	} else {
		a.ConfigDB.Exec("UPDATE providers SET name=?, type=?, endpoint=?, api_key=?, extra_config=?, models=? WHERE id=?", i.Name, i.Type, i.Endpoint, encryptSecret(i.APIKey), i.ExtraConfig, i.Models, r.PathValue("id"))
	}
	a.appendSecurityEvent("provider_update", r.URL.Path, "admin", fmt.Sprintf("updated provider %s", r.PathValue("id")))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteProvider(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM providers WHERE id=?", r.PathValue("id"))
	a.appendSecurityEvent("provider_delete", r.URL.Path, "admin", fmt.Sprintf("deleted provider %s", r.PathValue("id")))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSources(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, type, COALESCE(api_key,''), COALESCE(file_path,'') FROM sources")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Source{})
		return
	}
	defer rows.Close()
	var list []Source
	for rows.Next() {
		var i Source
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.APIKey, &i.FilePath)
		i.APIKey = maskSecret(i.APIKey)
		list = append(list, i)
	}
	if list == nil {
		list = []Source{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateSource(w http.ResponseWriter, r *http.Request) {
	var i Source
	json.NewDecoder(r.Body).Decode(&i)
	if i.ID == "" {
		i.ID = fmt.Sprintf("src_%d", time.Now().UnixNano())
	}
	if i.Type != "Local File" && i.APIKey == "" {
		i.APIKey = fmt.Sprintf("sk_src_%d", time.Now().UnixNano())
	}
	a.ConfigDB.Exec("INSERT INTO sources (id, name, type, api_key, file_path) VALUES (?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, encryptSecret(i.APIKey), i.FilePath)
	a.appendSecurityEvent("source_create", r.URL.Path, "admin", fmt.Sprintf("created source %s (%s)", i.ID, i.Name))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSource(w http.ResponseWriter, r *http.Request) {
	var i Source
	json.NewDecoder(r.Body).Decode(&i)
	if i.Type != "Local File" && strings.TrimSpace(i.APIKey) == "" {
		a.ConfigDB.Exec("UPDATE sources SET name=?, type=?, file_path=? WHERE id=?", i.Name, i.Type, i.FilePath, r.PathValue("id"))
	} else {
		a.ConfigDB.Exec("UPDATE sources SET name=?, type=?, api_key=?, file_path=? WHERE id=?", i.Name, i.Type, encryptSecret(i.APIKey), i.FilePath, r.PathValue("id"))
	}
	a.appendSecurityEvent("source_update", r.URL.Path, "admin", fmt.Sprintf("updated source %s", r.PathValue("id")))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteSource(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM sources WHERE id=?", r.PathValue("id"))
	a.appendSecurityEvent("source_delete", r.URL.Path, "admin", fmt.Sprintf("deleted source %s", r.PathValue("id")))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleSourceUpdate allows remote clients to push structured payloads directly into Agent memory.
func (a *App) handleSourceUpdate(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	reqKey := r.URL.Query().Get("key")
	if reqKey == "" {
		reqKey = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}

	var id, dbKey string
	if err := a.ConfigDB.QueryRow("SELECT id, api_key FROM sources WHERE name=?", name).Scan(&id, &dbKey); err != nil {
		http.Error(w, `{"error": "Unauthorized or Invalid API Key"}`, http.StatusUnauthorized)
		return
	}
	dbKey = decryptSecret(dbKey)
	if dbKey != "" && dbKey != reqKey {
		http.Error(w, `{"error": "Unauthorized or Invalid API Key"}`, http.StatusUnauthorized)
		return
	}

	body, _ := io.ReadAll(r.Body)
	ts := time.Now().Format("2006-01-02 15:04:05")
	AsyncDBExec(a.ConfigDB, "INSERT INTO source_logs (source_id, data, timestamp) VALUES (?, ?, ?)", id, string(body), ts)

	rows, _ := a.ConfigDB.Query("SELECT id, sources FROM agents")
	if rows != nil {
		defer rows.Close()
		for rows.Next() {
			var aId, srcJson string
			rows.Scan(&aId, &srcJson)
			var mounts []map[string]interface{}
			json.Unmarshal([]byte(srcJson), &mounts)
			for _, m := range mounts {
				if m["id"] == id && m["push_update"] == true {
					msg := fmt.Sprintf("[Source Update %s]\n%s", name, string(body))
					AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", aId, "system", msg)
				}
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSkills(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, type, instruction, content, api_method, api_url, api_headers, api_body, COALESCE(use_docker,0) FROM skills")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Skill{})
		return
	}
	defer rows.Close()
	var list []Skill
	for rows.Next() {
		var i Skill
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.Instruction, &i.Content, &i.APIMethod, &i.APIURL, &i.APIHeaders, &i.APIBody, &i.UseDocker)
		list = append(list, i)
	}
	if list == nil {
		list = []Skill{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateSkill(w http.ResponseWriter, r *http.Request) {
	var i Skill
	json.NewDecoder(r.Body).Decode(&i)
	if i.Type != "Bash" && i.Type != "JavaScript" && i.Type != "Go" {
		i.UseDocker = false
	}
	a.ConfigDB.Exec("INSERT INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body, use_docker) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody, i.UseDocker)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSkill(w http.ResponseWriter, r *http.Request) {
	var i Skill
	json.NewDecoder(r.Body).Decode(&i)
	if i.Type != "Bash" && i.Type != "JavaScript" && i.Type != "Go" {
		i.UseDocker = false
	}
	a.ConfigDB.Exec("UPDATE skills SET name=?, type=?, instruction=?, content=?, api_method=?, api_url=?, api_headers=?, api_body=?, use_docker=? WHERE id=?", i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody, i.UseDocker, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteSkill(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM skills WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetRAGs(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, table_name FROM rags")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]RAG{})
		return
	}
	defer rows.Close()
	var list []RAG
	for rows.Next() {
		var i RAG
		rows.Scan(&i.ID, &i.Name, &i.Description, &i.TableName)
		list = append(list, i)
	}
	if list == nil {
		list = []RAG{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateRAG(w http.ResponseWriter, r *http.Request) {
	var i RAG
	json.NewDecoder(r.Body).Decode(&i)
	// Always generate ID and table name server-side — never trust client-supplied values.
	i.ID = fmt.Sprintf("rag_%d", time.Now().UnixNano())
	i.TableName = fmt.Sprintf("rag_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO rags (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	// sanitizeTableName is called here as a belt-and-suspenders check on our own generated name.
	if safeTbl, err := sanitizeTableName(i.TableName); err == nil {
		a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT, vector TEXT DEFAULT '[]')", safeTbl))
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateRAG(w http.ResponseWriter, r *http.Request) {
	var i RAG
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE rags SET name=?, description=? WHERE id=?", i.Name, i.Description, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteRAG(w http.ResponseWriter, r *http.Request) {
	var t string
	a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", r.PathValue("id")).Scan(&t)
	if safeTbl, err := sanitizeTableName(t); err == nil {
		a.MemoryDB.Exec("DROP TABLE IF EXISTS " + safeTbl)
	}
	a.ConfigDB.Exec("DELETE FROM rags WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleQueryRAG tests a query against a RAG database and returns matching records.
func (a *App) handleQueryRAG(w http.ResponseWriter, r *http.Request) {
	ragID := r.PathValue("id")
	var req struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Query == "" {
		http.Error(w, `{"error":"query is required"}`, http.StatusBadRequest)
		return
	}
	if req.Limit <= 0 || req.Limit > 20 {
		req.Limit = 5
	}

	var rawTName string
	a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&rawTName)
	if rawTName == "" {
		http.Error(w, `{"error":"RAG not found"}`, http.StatusNotFound)
		return
	}
	tName, tErr := sanitizeTableName(rawTName)
	if tErr != nil {
		http.Error(w, `{"error":"invalid RAG table"}`, http.StatusBadRequest)
		return
	}

	userVector := getEmbedding(req.Query, a.ConfigDB)
	var hybridEnabled string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='rag_hybrid_search'").Scan(&hybridEnabled)
	keywords := strings.Fields(strings.ToLower(req.Query))

	rows, err := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data, vector FROM %s", tName))
	if err != nil {
		http.Error(w, `{"error":"query failed"}`, http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	type result struct {
		Name  string  `json:"name"`
		Data  string  `json:"data"`
		Score float64 `json:"score"`
	}
	var results []result
	for rows.Next() {
		var rName, rData, vJson string
		rows.Scan(&rName, &rData, &vJson)
		var score float32
		if userVector != nil {
			var dbVector []float32
			if json.Unmarshal([]byte(vJson), &dbVector) == nil && len(dbVector) > 0 {
				score = cosineSimilarity(userVector, dbVector)
			}
		}
		if hybridEnabled == "true" {
			dataLower := strings.ToLower(rData)
			keywordHits := 0
			for _, kw := range keywords {
				if len(kw) > 2 && strings.Contains(dataLower, kw) {
					keywordHits++
				}
			}
			if len(keywords) > 0 {
				keywordScore := float32(keywordHits) / float32(len(keywords))
				score = score*0.7 + keywordScore*0.3
			}
		}
		if score > 0.2 {
			results = append(results, result{Name: rName, Data: rData, Score: float64(score)})
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > req.Limit {
		results = results[:req.Limit]
	}
	if results == nil {
		results = []result{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"results": results, "total": len(results)})
}

// notifySystemError sends an error notification to the configured webhook Output.
// Called asynchronously — failures are silently logged.
func (a *App) notifySystemError(title, detail string) {
	var outputID string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='error_output_id'").Scan(&outputID)
	if outputID == "" {
		return
	}
	var webhookURL string
	a.ConfigDB.QueryRow("SELECT webhook_url FROM outputs WHERE id=? AND type='Webhook'", outputID).Scan(&webhookURL)
	if webhookURL == "" {
		return
	}
	if isPrivateHost(webhookURL) {
		return
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"event":     "system_error",
		"title":     title,
		"detail":    detail,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   CurrentVersion,
	})
	req, err := http.NewRequest("POST", webhookURL, bytes.NewReader(payload))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Zuver-ErrorNotify/1.0")
	go func() {
		resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
		if err == nil {
			resp.Body.Close()
		}
	}()
}

// dispatchWebhookOutputs fires any Webhook-type outputs assigned to the agent, sending the
// reply as a JSON POST payload to each configured URL. Runs in a background goroutine.
func (a *App) dispatchWebhookOutputs(agent Agent, reply string) {
	var assignedOutputs []string
	if err := json.Unmarshal([]byte(agent.Outputs), &assignedOutputs); err != nil || len(assignedOutputs) == 0 {
		return
	}
	if len(assignedOutputs) == 0 {
		return
	}
	placeholders := make([]string, len(assignedOutputs))
	args := make([]interface{}, len(assignedOutputs))
	for i, id := range assignedOutputs {
		placeholders[i] = "?"
		args[i] = id
	}
	query := fmt.Sprintf("SELECT id, name, type, webhook_url FROM outputs WHERE type='Webhook' AND id IN (%s)", strings.Join(placeholders, ","))
	rows, err := a.ConfigDB.Query(query, args...)
	if err != nil {
		return
	}
	defer rows.Close()
	type whOut struct{ id, name, typ, url string }
	var hooks []whOut
	for rows.Next() {
		var h whOut
		rows.Scan(&h.id, &h.name, &h.typ, &h.url)
		if h.url != "" {
			hooks = append(hooks, h)
		}
	}
	rows.Close()
	for _, h := range hooks {
		// SSRF guard: refuse to POST to private/loopback addresses.
		if isPrivateHost(h.url) {
			log.Printf("[Security] Webhook output %q blocked: URL %q resolves to a private host", h.id, h.url)
			continue
		}
		payload, _ := json.Marshal(map[string]interface{}{
			"agent_id":   agent.ID,
			"agent_name": agent.Name,
			"reply":      reply,
			"output_id":  h.id,
			"timestamp":  time.Now().UTC().Format(time.RFC3339),
		})
		req, err := http.NewRequest("POST", h.url, bytes.NewReader(payload))
		if err != nil {
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "Zuver-Webhook/1.0")
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err == nil {
			resp.Body.Close()
		}
	}
}

func (a *App) handleGetOutputs(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, instruction, COALESCE(type,'Command'), COALESCE(command_tpl,''), COALESCE(webhook_url,'') FROM outputs")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Output{})
		return
	}
	defer rows.Close()
	var list []Output
	for rows.Next() {
		var i Output
		rows.Scan(&i.ID, &i.Name, &i.Instruction, &i.Type, &i.CommandTpl, &i.WebhookURL)
		if i.Type == "" {
			i.Type = "Command"
		}
		list = append(list, i)
	}
	if list == nil {
		list = []Output{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateOutput(w http.ResponseWriter, r *http.Request) {
	var i Output
	json.NewDecoder(r.Body).Decode(&i)
	if i.ID == "" {
		i.ID = fmt.Sprintf("out_%d", time.Now().UnixNano())
	}
	if i.Type == "" {
		i.Type = "Command"
	}
	a.ConfigDB.Exec("INSERT INTO outputs (id, name, instruction, type, command_tpl, webhook_url) VALUES (?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Instruction, i.Type, i.CommandTpl, i.WebhookURL)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateOutput(w http.ResponseWriter, r *http.Request) {
	var i Output
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE outputs SET name=?, instruction=?, type=?, command_tpl=?, webhook_url=? WHERE id=?", i.Name, i.Instruction, i.Type, i.CommandTpl, i.WebhookURL, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteOutput(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM outputs WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetAgents(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query(`
		SELECT id, name, provider_id, model, sources, skills, outputs, mcps,
		       COALESCE(mcp_tools,'[]'),
		       COALESCE(projects,'[]'), system_prompt, token_usage,
		       COALESCE(input_methods,'["Text"]'), COALESCE(output_methods,'["Text"]'),
		       COALESCE(user_prompt_prefix,''), COALESCE(temperature,0.7),
		       COALESCE(max_tokens,4096), COALESCE(top_p,1.0),
		       COALESCE(privacy_enabled,0), COALESCE(can_create_skills,0),
		       COALESCE(stream_enabled,1)
		FROM agents`)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Agent{})
		return
	}
	defer rows.Close()
	var list []Agent
	for rows.Next() {
		var i Agent
		rows.Scan(&i.ID, &i.Name, &i.ProviderID, &i.Model, &i.Sources, &i.Skills,
			&i.Outputs, &i.MCPs, &i.MCPTools, &i.Projects, &i.SystemPrompt, &i.TokenUsage,
			&i.InputMethods, &i.OutputMethods, &i.UserPromptPrefix, &i.Temperature,
			&i.MaxTokens, &i.TopP, &i.PrivacyEnabled, &i.CanCreateSkills, &i.StreamEnabled)
		list = append(list, i)
	}
	if list == nil {
		list = []Agent{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateAgent(w http.ResponseWriter, r *http.Request) {
	var i Agent
	json.NewDecoder(r.Body).Decode(&i)
	if i.MCPTools == "" {
		i.MCPTools = "[]"
	}
	a.ConfigDB.Exec(`INSERT INTO agents
		(id, name, provider_id, model, sources, skills, outputs, mcps, mcp_tools, projects,
		 system_prompt, input_methods, output_methods, user_prompt_prefix,
		 temperature, max_tokens, top_p, privacy_enabled, can_create_skills, stream_enabled)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		i.ID, i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs,
		i.MCPs, i.MCPTools, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods,
		i.UserPromptPrefix, i.Temperature, i.MaxTokens, i.TopP,
		i.PrivacyEnabled, i.CanCreateSkills, i.StreamEnabled)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateAgent(w http.ResponseWriter, r *http.Request) {
	var i Agent
	json.NewDecoder(r.Body).Decode(&i)
	if i.MCPTools == "" {
		i.MCPTools = "[]"
	}
	a.ConfigDB.Exec(`UPDATE agents SET
		name=?, provider_id=?, model=?, sources=?, skills=?, outputs=?, mcps=?, mcp_tools=?,
		projects=?, system_prompt=?, input_methods=?, output_methods=?,
		user_prompt_prefix=?, temperature=?, max_tokens=?, top_p=?,
		privacy_enabled=?, can_create_skills=?, stream_enabled=?
		WHERE id=?`,
		i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs,
		i.MCPs, i.MCPTools, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods,
		i.UserPromptPrefix, i.Temperature, i.MaxTokens, i.TopP,
		i.PrivacyEnabled, i.CanCreateSkills, i.StreamEnabled, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteAgent(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM agents WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetProjects(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, flow_json, COALESCE(tags,''), COALESCE(is_active,1) FROM projects")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Project{})
		return
	}
	defer rows.Close()
	var list []Project
	for rows.Next() {
		var i Project
		var isActiveInt int
		rows.Scan(&i.ID, &i.Name, &i.Description, &i.FlowJSON, &i.Tags, &isActiveInt)
		i.IsActive = isActiveInt == 1
		list = append(list, i)
	}
	if list == nil {
		list = []Project{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateProject(w http.ResponseWriter, r *http.Request) {
	var i Project
	json.NewDecoder(r.Body).Decode(&i)
	if i.ID == "" {
		i.ID = fmt.Sprintf("proj_%d", time.Now().UnixNano())
	}
	if i.FlowJSON == "" {
		i.FlowJSON = `{"nodes":[{"id":"start","type":"Start","x":80,"y":180,"trigger":"Manual"},{"id":"end","type":"End","x":520,"y":180}],"edges":[]}`
	}
	isActive := 1
	if !i.IsActive {
		isActive = 0
	}
	SyncDBExec(a.ConfigDB, "INSERT INTO projects (id, name, description, flow_json, tags, is_active) VALUES (?, ?, ?, ?, ?, ?)",
		i.ID, i.Name, i.Description, i.FlowJSON, i.Tags, isActive)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": i.ID})
}

func (a *App) handleUpdateProject(w http.ResponseWriter, r *http.Request) {
	var i Project
	json.NewDecoder(r.Body).Decode(&i)
	isActive := 1
	if !i.IsActive {
		isActive = 0
	}
	a.ConfigDB.Exec("UPDATE projects SET name=?, description=?, flow_json=?, tags=?, is_active=? WHERE id=?",
		i.Name, i.Description, i.FlowJSON, i.Tags, isActive, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteProject(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	a.ConfigDB.Exec("DELETE FROM projects WHERE id=?", id)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleRunProject executes a saved project pipeline via HTTP POST, returning the final output.
func (a *App) handleRunProject(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		Input   string `json:"input"`
		AgentID string `json:"agent_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Input == "" {
		req.Input = "[Pipeline Manual Trigger]"
	}
	result, logs := a.runProjectPipelineVerbose(id, req.AgentID, req.Input)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"result": result, "logs": logs})
}

// handleTriggerProject is a webhook-friendly endpoint for external systems to trigger pipelines.
// Accepts optional JSON body with "input" field. Returns the pipeline result.
func (a *App) handleTriggerProject(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	// Verify project exists.
	var name string
	if err := a.ConfigDB.QueryRow("SELECT name FROM projects WHERE id=?", id).Scan(&name); err != nil {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"project not found"}`, http.StatusNotFound)
		return
	}
	var req struct {
		Input   string `json:"input"`
		AgentID string `json:"agent_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Input == "" {
		req.Input = "[Webhook Trigger]"
	}
	a.appendSecurityEvent("webhook_trigger", r.URL.Path, "api_key", fmt.Sprintf("triggered project %s (%s)", id, name))
	result, logs := a.runProjectPipelineVerbose(id, req.AgentID, req.Input)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"result": result, "logs": logs, "project_id": id, "project_name": name})
}

func (a *App) handleGetMCP(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, url, command, COALESCE(tools,'[]') FROM mcp_servers")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]MCPServer{})
		return
	}
	defer rows.Close()
	var list []MCPServer
	for rows.Next() {
		var i MCPServer
		rows.Scan(&i.ID, &i.Name, &i.URL, &i.Command, &i.Tools)
		list = append(list, i)
	}
	if list == nil {
		list = []MCPServer{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateMCP(w http.ResponseWriter, r *http.Request) {
	var i MCPServer
	json.NewDecoder(r.Body).Decode(&i)
	i.ID = fmt.Sprintf("mcp_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO mcp_servers (id, name, url, command, tools) VALUES (?, ?, ?, ?, '[]')", i.ID, i.Name, i.URL, i.Command)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": i.ID})
}

// mcpResolveHTTP calls tools/list via the HTTP/SSE JSON-RPC transport and returns tool names.
func mcpResolveHTTP(mcpURL string) ([]string, error) {
	parsed, err := url.ParseRequestURI(mcpURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("invalid MCP URL scheme")
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0", "id": 1,
		"method": "tools/list", "params": map[string]interface{}{},
	})
	resp, err := (&http.Client{Timeout: 8 * time.Second}).Post(mcpURL, "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var rpcResp struct {
		Result struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return nil, fmt.Errorf("invalid JSON-RPC response")
	}
	var tools []string
	for _, t := range rpcResp.Result.Tools {
		if t.Name != "" {
			tools = append(tools, t.Name)
		}
	}
	return tools, nil
}

// mcpResolveStdio launches a stdio MCP process, performs initialize + tools/list, and returns tool names.
func mcpResolveStdio(mcpCmd string) ([]string, error) {
	parts := strings.Fields(mcpCmd)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty MCP command")
	}
	cmd := exec.Command(parts[0], parts[1:]...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start MCP process: %w", err)
	}
	defer cmd.Process.Kill()

	writeRPC := func(method string, params interface{}, msgID int) {
		msg, _ := json.Marshal(map[string]interface{}{
			"jsonrpc": "2.0", "id": msgID, "method": method, "params": params,
		})
		fmt.Fprintf(stdin, "Content-Length: %d\r\n\r\n%s", len(msg), msg)
	}
	writeRPC("initialize", map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo":      map[string]interface{}{"name": "zuver", "version": "1.0"},
	}, 1)
	writeRPC("notifications/initialized", nil, 2)
	writeRPC("tools/list", map[string]interface{}{}, 3)
	stdin.Close()

	type readResult struct {
		data []byte
		err  error
	}
	ch := make(chan readResult, 1)
	go func() {
		buf := make([]byte, 65536)
		n, err := io.ReadFull(stdout, buf)
		if err != nil && n == 0 {
			// Fall back to partial read
			n2, _ := stdout.Read(buf)
			ch <- readResult{buf[:n2], nil}
			return
		}
		ch <- readResult{buf[:n], nil}
	}()

	var rawOut []byte
	select {
	case res := <-ch:
		rawOut = res.data
	case <-time.After(8 * time.Second):
		return nil, fmt.Errorf("stdio process timed out")
	}

	// Parse LSP-style framed messages: "Content-Length: N\r\n\r\n{...}"
	var rpcResp struct {
		ID     int `json:"id"`
		Result struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"result"`
	}
	remaining := string(rawOut)
	for len(remaining) > 0 {
		idx := strings.Index(remaining, "\r\n\r\n")
		if idx < 0 {
			break
		}
		body := remaining[idx+4:]
		remaining = ""
		// Find the end of this JSON object
		depth, end := 0, -1
		for i, c := range body {
			if c == '{' {
				depth++
			} else if c == '}' {
				depth--
				if depth == 0 {
					end = i + 1
					break
				}
			}
		}
		if end < 0 {
			break
		}
		chunk := body[:end]
		remaining = body[end:]
		if err := json.Unmarshal([]byte(chunk), &rpcResp); err == nil && rpcResp.ID == 3 {
			break
		}
	}
	var tools []string
	for _, t := range rpcResp.Result.Tools {
		if t.Name != "" {
			tools = append(tools, t.Name)
		}
	}
	return tools, nil
}

// handleResolveMCP contacts the MCP server using ALL configured transports concurrently
// (HTTP/SSE and stdio), merges the tool lists, deduplicates, and persists to DB.
func (a *App) handleResolveMCP(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var mcpURL, mcpCmd string
	if err := a.ConfigDB.QueryRow("SELECT url, command FROM mcp_servers WHERE id=?", id).Scan(&mcpURL, &mcpCmd); err != nil {
		http.Error(w, `{"error":"MCP server not found"}`, http.StatusNotFound)
		return
	}
	if mcpURL == "" && mcpCmd == "" {
		http.Error(w, `{"error":"MCP server has neither URL nor command configured"}`, http.StatusBadRequest)
		return
	}

	type result struct {
		tools []string
		err   error
	}
	ch := make(chan result, 2)

	if mcpURL != "" {
		go func() {
			t, err := mcpResolveHTTP(mcpURL)
			ch <- result{t, err}
		}()
	}
	if mcpCmd != "" {
		go func() {
			t, err := mcpResolveStdio(mcpCmd)
			ch <- result{t, err}
		}()
	}

	// Collect results from however many goroutines were launched
	expected := 0
	if mcpURL != "" {
		expected++
	}
	if mcpCmd != "" {
		expected++
	}

	seen := map[string]bool{}
	var tools []string
	var errs []string
	for i := 0; i < expected; i++ {
		res := <-ch
		if res.err != nil {
			errs = append(errs, res.err.Error())
			continue
		}
		for _, t := range res.tools {
			if !seen[t] {
				seen[t] = true
				tools = append(tools, t)
			}
		}
	}

	// Only fail hard if every transport failed
	if len(tools) == 0 && len(errs) == expected {
		http.Error(w, `{"error":"all transports failed: `+strings.Join(errs, "; ")+`"}`, http.StatusBadGateway)
		return
	}

	toolsJSON, _ := json.Marshal(tools)
	a.ConfigDB.Exec("UPDATE mcp_servers SET tools=? WHERE id=?", string(toolsJSON), id)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tools":    tools,
		"status":   "ok",
		"warnings": errs,
	})
}

func (a *App) handleUpdateMCP(w http.ResponseWriter, r *http.Request) {
	var i MCPServer
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE mcp_servers SET name=?, url=?, command=? WHERE id=?",
		i.Name, i.URL, i.Command, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteMCP(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM mcp_servers WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetPreferences(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, table_name FROM preferences")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Preference{})
		return
	}
	defer rows.Close()
	var list []Preference
	for rows.Next() {
		var i Preference
		rows.Scan(&i.ID, &i.Name, &i.Description, &i.TableName)
		list = append(list, i)
	}
	if list == nil {
		list = []Preference{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreatePreference(w http.ResponseWriter, r *http.Request) {
	var i Preference
	json.NewDecoder(r.Body).Decode(&i)
	// Always generate ID and table name server-side — never trust client-supplied values.
	i.ID = fmt.Sprintf("pref_%d", time.Now().UnixNano())
	i.TableName = fmt.Sprintf("pref_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO preferences (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	if safeTbl, err := sanitizeTableName(i.TableName); err == nil {
		a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT)", safeTbl))
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdatePreference(w http.ResponseWriter, r *http.Request) {
	var i Preference
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE preferences SET name=?, description=? WHERE id=?", i.Name, i.Description, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeletePreference(w http.ResponseWriter, r *http.Request) {
	var t string
	a.ConfigDB.QueryRow("SELECT table_name FROM preferences WHERE id=?", r.PathValue("id")).Scan(&t)
	if safeTbl, err := sanitizeTableName(t); err == nil {
		a.MemoryDB.Exec("DROP TABLE IF EXISTS " + safeTbl)
	}
	a.ConfigDB.Exec("DELETE FROM preferences WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSettings(w http.ResponseWriter, r *http.Request) {
	rows, _ := a.ConfigDB.Query("SELECT key, value FROM settings")
	defer rows.Close()
	res := make(map[string]string)
	for rows.Next() {
		var k, v string
		rows.Scan(&k, &v)
		res[k] = v
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

func (a *App) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	var data map[string]string
	json.NewDecoder(r.Body).Decode(&data)
	for k, v := range data {
		a.ConfigDB.Exec("UPDATE settings SET value=? WHERE key=?", v, k)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleChangePassword(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req struct {
		CurrentPassword string `json:"current_password"`
		NewPassword     string `json:"new_password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "invalid request"}`, http.StatusBadRequest)
		return
	}
	if len(req.NewPassword) < 8 {
		http.Error(w, `{"error": "New password must be at least 8 characters"}`, http.StatusBadRequest)
		return
	}

	var dbHash string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='admin_password'").Scan(&dbHash)
	if dbHash == "" {
		http.Error(w, `{"error": "Admin password not initialized"}`, http.StatusUnauthorized)
		return
	}

	// Verify current password.
	if !bcryptCompare(dbHash, req.CurrentPassword) {
		a.appendSecurityEvent("password_change_failure", r.URL.Path, getClientIP(r), "invalid current password")
		http.Error(w, `{"error": "Current password is incorrect"}`, http.StatusUnauthorized)
		return
	}

	// Hash and store new password.
	newHash, err := bcryptHash(req.NewPassword)
	if err != nil {
		http.Error(w, `{"error": "Server error"}`, http.StatusInternalServerError)
		return
	}
	SyncDBExec(a.ConfigDB, "UPDATE settings SET value=? WHERE key='admin_password'", newHash)
	a.appendSecurityEvent("password_changed", r.URL.Path, getClientIP(r), "admin password updated")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// modelPricing represents per-model token pricing (per 1M tokens).
type modelPricing struct {
	ModelID    string  `json:"model_id"`
	Alias      string  `json:"alias"`
	InputPrice float64 `json:"input_price"`
	OutputPrice float64 `json:"output_price"`
}

// lookupModelCost calculates the cost for a given token count and provider model pricing.
// Returns 0 if no pricing info is found.
func (a *App) lookupModelCost(providerID string, tokens int) float64 {
	var modelsJSON string
	a.ConfigDB.QueryRow("SELECT COALESCE(models,'[]') FROM providers WHERE id=?", providerID).Scan(&modelsJSON)
	var models []modelPricing
	if json.Unmarshal([]byte(modelsJSON), &models) != nil || len(models) == 0 {
		return 0
	}
	// Use the first model's pricing as the default for this provider.
	// Assume roughly 50/50 input/output split when we don't have exact counts.
	p := models[0]
	avgPrice := (p.InputPrice + p.OutputPrice) / 2
	return float64(tokens) * avgPrice / 1_000_000
}

func (a *App) logAnalytics(entityType, entityID string, tokens int, success bool) {
	go func() {
		var cost float64
		if entityType == "agent" {
			// Look up provider pricing via agent -> provider mapping.
			var providerID string
			a.ConfigDB.QueryRow("SELECT provider_id FROM agents WHERE id=?", entityID).Scan(&providerID)
			if providerID != "" {
				cost = a.lookupModelCost(providerID, tokens)
			}
		} else if entityType == "provider" {
			cost = a.lookupModelCost(entityID, tokens)
		}
		a.ConfigDB.Exec("INSERT INTO analytics_logs (entity_type, entity_id, tokens, is_success, cost) VALUES (?, ?, ?, ?, ?)", entityType, entityID, tokens, success, cost)
	}()
}

// runProjectPipeline is the legacy single-return wrapper used by the agentic command handler.
func (a *App) runProjectPipeline(projectID string, callerAgentID string, input string) string {
	result, _ := a.runProjectPipelineVerbose(projectID, callerAgentID, input)
	return result
}

// runProjectPipelineVerbose executes a full project pipeline with real agent LLM calls,
// skill invocations, conditional branching, HTTP requests, and transform nodes.
// It returns the final output text and an execution log slice.
func (a *App) runProjectPipelineVerbose(projectID string, callerAgentID string, input string) (string, []string) {
	var flowJson string
	a.ConfigDB.QueryRow("SELECT flow_json FROM projects WHERE id=?", projectID).Scan(&flowJson)
	if flowJson == "" {
		return "[PIPELINE ERROR] Project not found.", []string{}
	}

	var flow struct {
		Nodes []map[string]interface{} `json:"nodes"`
		Edges []map[string]interface{} `json:"edges"`
	}
	if err := json.Unmarshal([]byte(flowJson), &flow); err != nil {
		return "[PIPELINE ERROR] Invalid flow JSON.", []string{}
	}

	// Build adjacency map for fast edge lookup (from → []to).
	adjMap := make(map[string][]string)
	for _, e := range flow.Edges {
		from, _ := e["from"].(string)
		to, _ := e["to"].(string)
		if from != "" && to != "" {
			adjMap[from] = append(adjMap[from], to)
		}
	}

	// Build node map for fast lookup.
	nodeMap := make(map[string]map[string]interface{})
	for _, n := range flow.Nodes {
		if id, ok := n["id"].(string); ok {
			nodeMap[id] = n
		}
	}

	// Find the Start node.
	var startID string
	for _, n := range flow.Nodes {
		if n["type"] == "Start" {
			startID, _ = n["id"].(string)
			break
		}
	}
	if startID == "" {
		return "[PIPELINE ERROR] No Start node found.", []string{}
	}

	var execLogs []string
	logf := func(format string, args ...interface{}) {
		execLogs = append(execLogs, fmt.Sprintf("[Pipeline] "+format, args...))
	}

	currentNodeID := startID
	lastResult := input
	visited := make(map[string]int)

	// Pipeline variables bag for data passing between nodes.
	vars := map[string]string{"input": input}

	for step := 0; step < 30; step++ {
		nexts := adjMap[currentNodeID]
		if len(nexts) == 0 {
			logf("No outbound edges from '%s'. Terminating.", currentNodeID)
			break
		}

		// For conditional nodes we may choose among multiple outbound edges.
		// Default: follow the first outbound edge.
		nextID := nexts[0]

		node := nodeMap[currentNodeID]
		nodeType, _ := node["type"].(string)

		// Guard against infinite loops.
		visited[currentNodeID]++
		if visited[currentNodeID] > 5 {
			logf("Cycle detected at node '%s'. Terminating.", currentNodeID)
			break
		}

		logf("Executing node '%s' (type=%s)", currentNodeID, nodeType)

		// Resolve {variable} references in a string.
		interpolate := func(s string) string {
			for k, v := range vars {
				s = strings.ReplaceAll(s, "{"+k+"}", v)
			}
			s = strings.ReplaceAll(s, "{input}", lastResult)
			return s
		}

		switch nodeType {
		case "Start":
			// Nothing to execute — move forward.

		case "End":
			logf("Reached End node.")
			return lastResult, execLogs

		case "Agent":
			aID, _ := node["agentId"].(string)
			customSys, _ := node["sysPrompt"].(string)
			customPrefix, _ := node["userPrefix"].(string)
			if aID == "" {
				logf("Agent node has no agentId configured.")
				lastResult = "[PIPELINE ERROR] Agent node not configured."
				break
			}
			var agentModel, agentProviderID, defaultSys, defaultPrefix, agentExtraConfig string
			var agentTemp float64
			var agentMaxTokens int
			a.ConfigDB.QueryRow(
				"SELECT model, provider_id, system_prompt, COALESCE(user_prompt_prefix,''), temperature, max_tokens FROM agents WHERE id=?", aID,
			).Scan(&agentModel, &agentProviderID, &defaultSys, &defaultPrefix, &agentTemp, &agentMaxTokens)

			sysPrompt := defaultSys
			if customSys != "" {
				sysPrompt = customSys
			}
			userPrefix := defaultPrefix
			if customPrefix != "" {
				userPrefix = customPrefix
			}

			var provType, provEndpoint, provKey string
			a.ConfigDB.QueryRow(
				"SELECT COALESCE(type,'OpenAI'), endpoint, api_key, COALESCE(extra_config,'{}') FROM providers WHERE id=?", agentProviderID,
			).Scan(&provType, &provEndpoint, &provKey, &agentExtraConfig)

			userMsg := interpolate(lastResult)
			if userPrefix != "" {
				userMsg = userPrefix + "\n" + userMsg
			}

			msgs := []map[string]interface{}{
				{"role": "system", "content": sysPrompt},
				{"role": "user", "content": userMsg},
			}

			apiURL := strings.TrimRight(provEndpoint, "/")
			switch strings.ToLower(provType) {
			case "claude":
				if !strings.HasSuffix(apiURL, "/v1/messages") {
					apiURL += "/v1/messages"
				}
			case "ollama":
				if !strings.HasSuffix(apiURL, "/api/chat") {
					apiURL += "/api/chat"
				}
			default:
				if !strings.HasSuffix(apiURL, "/v1/chat/completions") {
					apiURL += "/v1/chat/completions"
				}
			}

			if agentTemp == 0 {
				agentTemp = 0.7
			}
			if agentMaxTokens == 0 {
				agentMaxTokens = 4096
			}

			body := map[string]interface{}{
				"model":    agentModel,
				"messages": msgs,
				"stream":   false,
			}
			pTypeProj := strings.ToLower(provType)
			switch pTypeProj {
			case "claude":
				body["max_tokens"] = agentMaxTokens
				if agentTemp != 0.7 {
					body["temperature"] = agentTemp
				}
			case "ollama":
				body["options"] = map[string]interface{}{
					"temperature": agentTemp,
					"num_predict": agentMaxTokens,
				}
			default:
				body["temperature"] = agentTemp
				if agentMaxTokens > 0 {
					body["max_tokens"] = agentMaxTokens
				}
			}
			protectedBodyParams := map[string]bool{"model": true, "messages": true, "stream": true, "temperature": true, "max_tokens": true, "top_p": true, "system": true, "options": true}
			var extras2 map[string]interface{}
			if json.Unmarshal([]byte(agentExtraConfig), &extras2) == nil {
				for k, v := range extras2 {
					if !protectedBodyParams[k] {
						body[k] = v
					}
				}
			}

			payloadBytes, _ := json.Marshal(body)
			apiReq, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(payloadBytes))
			apiReq.Header.Set("Content-Type", "application/json")
			switch strings.ToLower(provType) {
			case "claude":
				apiReq.Header.Set("x-api-key", provKey)
				apiReq.Header.Set("anthropic-version", "2023-06-01")
			default:
				if strings.ToLower(provType) != "ollama" {
					apiReq.Header.Set("Authorization", "Bearer "+provKey)
				}
			}

			resp, errDo := (&http.Client{Timeout: 60 * time.Second}).Do(apiReq)
			if errDo != nil {
				lastResult = "[PIPELINE AGENT ERROR] " + errDo.Error()
				logf("Agent %s HTTP error: %v", aID, errDo)
				break
			}
			resBytes, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			var result map[string]interface{}
			json.Unmarshal(resBytes, &result)
			agentReply := ""
			if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
				if ch, ok := choices[0].(map[string]interface{}); ok {
					if msg, ok := ch["message"].(map[string]interface{}); ok {
						agentReply, _ = msg["content"].(string)
					}
				}
			} else if message, ok := result["message"].(map[string]interface{}); ok {
				agentReply, _ = message["content"].(string)
			} else if contentArr, ok := result["content"].([]interface{}); ok && len(contentArr) > 0 {
				if block, ok := contentArr[0].(map[string]interface{}); ok {
					agentReply, _ = block["text"].(string)
				}
			}
			if agentReply == "" {
				logf("Agent %s returned empty reply. Raw: %s", aID, string(resBytes))
				agentReply = "[PIPELINE] Agent returned no content."
			}

			lastResult = agentReply
			vars["agent_"+aID] = agentReply
			tokensUsed := estimateTokens(string(payloadBytes)) + estimateTokens(agentReply)
			AsyncDBExec(a.ConfigDB, "UPDATE agents SET token_usage = token_usage + ? WHERE id=?", tokensUsed, aID)
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", aID, "user", userMsg)
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", aID, "assistant", agentReply)
			a.logAnalytics("agent", aID, tokensUsed, true)
			logf("Agent %s replied (%d chars).", aID, len(agentReply))

		case "Skill":
			sID, _ := node["skillId"].(string)
			if sID == "" {
				logf("Skill node has no skillId.")
				break
			}
			var sType, sContent, sMethod, sUrl, sHeaders, sBody string
			var useDocker bool
			a.ConfigDB.QueryRow("SELECT type, content, api_method, api_url, api_headers, api_body, COALESCE(use_docker,0) FROM skills WHERE id=?", sID).
				Scan(&sType, &sContent, &sMethod, &sUrl, &sHeaders, &sBody, &useDocker)

			// Build a full substitution function that resolves:
			// 1. Named <paramName> placeholders bound via node["paramBindings"] map.
			// 2. The pipeline variable bag.
			// 3. The legacy {input} shorthand.
			paramBindings := map[string]string{}
			if pbRaw, ok := node["paramBindings"].(map[string]interface{}); ok {
				for k, v := range pbRaw {
					if sv, ok := v.(string); ok {
						// The binding value is a variable name; resolve from vars or use raw value.
						if resolved, exists := vars[sv]; exists {
							paramBindings[k] = resolved
						} else {
							paramBindings[k] = sv
						}
					}
				}
			}

			substituteSkill := func(s string) string {
				// Replace named <param> placeholders first.
				for param, val := range paramBindings {
					s = strings.ReplaceAll(s, "<"+param+">", val)
				}
				// Then pipeline variable interpolation.
				s = interpolate(s)
				return s
			}

			switch sType {
			case "Agent":
				// Invoke the target agent as a sub-call; pass the current pipeline
				// input (last result or the substituted content) as the message.
				input := substituteSkill(sContent)
				// If sContent is a JSON config (agent skill format), pass lastResult
				// as the user message; otherwise fall back to substituted content.
				var agentSkillCfg map[string]interface{}
				agentInput := lastResult
				if json.Unmarshal([]byte(sContent), &agentSkillCfg) != nil {
					// sContent is not JSON — treat as plain input text.
					agentInput = input
				}
				lastResult = a.callAgentSkill(sContent, agentInput)
			case "MD":
				lastResult = substituteSkill(sContent)
			case "Bash", "Go", "JavaScript":
				code := substituteSkill(sContent)
				var tmpExt string
				switch sType {
				case "Bash":
					tmpExt = ".sh"
				case "JavaScript":
					tmpExt = ".js"
				default:
					tmpExt = ".go"
				}
				if useDocker {
					dockerOut, dockerErr := executeInDocker(a.ConfigDB, code, sType, nil)
					if dockerErr != nil {
						lastResult = "[DOCKER ERROR] " + dockerErr.Error()
						if dockerOut != "" {
							lastResult += "\nOutput: " + dockerOut
						}
					} else {
						lastResult = strings.TrimSpace(dockerOut)
					}
				} else {
					tmp := filepath.Join(os.TempDir(), fmt.Sprintf("pipe_sk_%d%s", time.Now().UnixNano(), tmpExt))
					if err := os.WriteFile(tmp, []byte(code), 0700); err != nil {
						lastResult = "[SKILL ERROR] failed to write temp script"
						break
					}
					defer os.Remove(tmp)
					var cmd *exec.Cmd
					switch sType {
					case "Bash":
						cmd = exec.Command("sh", tmp)
					case "JavaScript":
						cmd = exec.Command("node", tmp)
					default:
						cmd = exec.Command("go", "run", tmp)
					}
					out, err := cmd.CombinedOutput()
					if err != nil {
						lastResult = fmt.Sprintf("[SKILL ERROR] %s\n%s", err.Error(), string(out))
					} else {
						lastResult = strings.TrimSpace(string(out))
					}
				}
			case "API":
				fUrl := substituteSkill(sUrl)
				fBody := substituteSkill(sBody)
				// SSRF guard: reject private/loopback destinations in API skills.
				if isPrivateHost(fUrl) {
					lastResult = "[SKILL API ERROR] URL targets a private/loopback address and is blocked."
				} else {
					apiR, _ := http.NewRequest(sMethod, fUrl, bytes.NewBuffer([]byte(fBody)))
					var hm map[string]string
					if json.Unmarshal([]byte(sHeaders), &hm) == nil {
						for k, v := range hm {
							apiR.Header.Set(k, v)
						}
					}
			extResp, e := safeHTTPClient(15 * time.Second).Do(apiR)
					if e != nil {
						lastResult = "[SKILL API ERROR] " + e.Error()
					} else {
						b, _ := io.ReadAll(extResp.Body)
						extResp.Body.Close()
						lastResult = string(b)
					}
				}
			}
			vars["skill_"+sID] = lastResult
			logf("Skill %s (%s) executed.", sID, sType)

		case "Code":
			code, _ := node["code"].(string)
			code = interpolate(code)
			tmp := filepath.Join(os.TempDir(), fmt.Sprintf("pipe_code_%d.go", time.Now().UnixNano()))
			os.WriteFile(tmp, []byte(code), 0644)
			cmd := exec.Command("go", "run", tmp)
			out, err := cmd.CombinedOutput()
			os.Remove(tmp)
			if err != nil {
				lastResult = fmt.Sprintf("[CODE ERROR] %s\n%s", err.Error(), string(out))
			} else {
				lastResult = strings.TrimSpace(string(out))
			}
			logf("Code node executed (%d chars output).", len(lastResult))

		case "HTTP":
			// HTTP Request node: performs an arbitrary outbound HTTP call.
			method, _ := node["httpMethod"].(string)
			url, _ := node["httpUrl"].(string)
			body, _ := node["httpBody"].(string)
			headersRaw, _ := node["httpHeaders"].(string)

			if method == "" {
				method = "GET"
			}
			url = interpolate(url)
			body = interpolate(body)

			// SSRF guard: reject private/loopback destinations in pipeline HTTP nodes.
			if isPrivateHost(url) {
				lastResult = "[HTTP ERROR] URL targets a private/loopback address and is blocked."
			} else {
				req, _ := http.NewRequest(method, url, bytes.NewBuffer([]byte(body)))
				req.Header.Set("Content-Type", "application/json")
				var hm map[string]string
				if json.Unmarshal([]byte(headersRaw), &hm) == nil {
					for k, v := range hm {
						req.Header.Set(k, v)
					}
				}
				resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
				if err != nil {
					lastResult = "[HTTP ERROR] " + err.Error()
				} else {
					b, _ := io.ReadAll(resp.Body)
					resp.Body.Close()
					lastResult = string(b)
				}
			}
			vars["http_result"] = lastResult
			logf("HTTP %s %s → %d chars", method, url, len(lastResult))

		case "If":
			// Conditional branching: evaluates a simple condition on lastResult.
			// Node config: condition (contains|equals|startsWith|regex), value, trueEdge, falseEdge
			condition, _ := node["condition"].(string)
			value, _ := node["condValue"].(string)
			trueEdge, _ := node["trueEdge"].(string)
			falseEdge, _ := node["falseEdge"].(string)
			value = interpolate(value)

			condMet := false
			switch condition {
			case "contains":
				condMet = strings.Contains(lastResult, value)
			case "equals":
				condMet = strings.TrimSpace(lastResult) == strings.TrimSpace(value)
			case "startsWith":
				condMet = strings.HasPrefix(strings.TrimSpace(lastResult), value)
			case "notContains":
				condMet = !strings.Contains(lastResult, value)
			case "regex":
				if len(value) > 100 {
					condMet = false
					break
				}
				re, err := regexp.Compile(value)
				if err != nil {
					condMet = false
					break
				}
				// Execute regex with 100ms timeout to prevent ReDoS.
				type result struct { val bool }
				ch := make(chan result, 1)
				go func() {
					ch <- result{re.MatchString(lastResult)}
				}()
				select {
				case r := <-ch:
					condMet = r.val
				case <-time.After(100 * time.Millisecond):
					condMet = false
				}
			default:
				condMet = lastResult != ""
			}

			logf("If node: condition=%s value=%q → %v", condition, value, condMet)

			if condMet && trueEdge != "" {
				nextID = trueEdge
			} else if !condMet && falseEdge != "" {
				nextID = falseEdge
			}

		case "Transform":
			// Text transformation: trim, upper, lower, extract JSON field, template.
			op, _ := node["operation"].(string)
			param, _ := node["param"].(string)
			param = interpolate(param)

			switch op {
			case "trim":
				lastResult = strings.TrimSpace(lastResult)
			case "upper":
				lastResult = strings.ToUpper(lastResult)
			case "lower":
				lastResult = strings.ToLower(lastResult)
			case "prefix":
				lastResult = param + lastResult
			case "suffix":
				lastResult = lastResult + param
			case "replace":
				// param format: "find|replace"
				parts := strings.SplitN(param, "|", 2)
				if len(parts) == 2 {
					lastResult = strings.ReplaceAll(lastResult, parts[0], parts[1])
				}
			case "jsonField":
				// Extract a top-level JSON field by name.
				var parsed map[string]interface{}
				if json.Unmarshal([]byte(lastResult), &parsed) == nil {
					if val, ok := parsed[param]; ok {
						switch v := val.(type) {
						case string:
							lastResult = v
						default:
							b, _ := json.Marshal(v)
							lastResult = string(b)
						}
					}
				}
			case "template":
				// param is a Go-style template where {input} is replaced.
				lastResult = strings.ReplaceAll(param, "{input}", lastResult)
			}
			vars["transformed"] = lastResult
			logf("Transform '%s' applied.", op)

		case "Wait":
			// Pause execution for a specified number of seconds (max 30).
			secRaw, _ := node["seconds"].(float64)
			if secRaw <= 0 {
				secRaw = 1
			}
			if secRaw > 30 {
				secRaw = 30
			}
			logf("Wait node: sleeping %vs", secRaw)
			time.Sleep(time.Duration(secRaw) * time.Second)

		case "Project":
			// Sub-project (nested pipeline) execution.
			subID, _ := node["projectId"].(string)
			if subID == "" || subID == projectID {
				logf("Sub-project node: invalid or self-referencing projectId.")
				break
			}
			subResult, subLogs := a.runProjectPipelineVerbose(subID, callerAgentID, lastResult)
			execLogs = append(execLogs, subLogs...)
			lastResult = subResult
			logf("Sub-project %s executed.", subID)

		case "SetVariable":
			// Stores the current output in a named variable.
			varName, _ := node["varName"].(string)
			if varName != "" {
				vars[varName] = lastResult
				logf("SetVariable: %s = %q", varName, lastResult)
			}

		case "GetVariable":
			// Retrieves a named variable into the pipeline stream.
			varName, _ := node["varName"].(string)
			if v, ok := vars[varName]; ok {
				lastResult = v
				logf("GetVariable: %s → %q", varName, lastResult)
			}
		}

		currentNodeID = nextID
	}

	return lastResult, execLogs
}

// extractFilePayload analyzes and extracts base64 representations and MIME headers from local binaries.
func extractFilePayload(path string) (mime string, pureB64 string, ok bool) {
	// Confine reads to the uploads directory to prevent path traversal.
	uploadsAbs, _ := filepath.Abs("uploads")
	absPath, err := filepath.Abs(path)
	if err != nil || !strings.HasPrefix(absPath, uploadsAbs+string(filepath.Separator)) {
		log.Printf("[Security] extractFilePayload rejected path outside uploads: %q", path)
		return "", "", false
	}

	b, err := os.ReadFile(absPath)
	if err != nil {
		log.Printf("[I/O Error] Failed to read file: %v", err)
		return "", "", false
	}

	ext := strings.ToLower(filepath.Ext(absPath))
	mime = "application/octet-stream"
	switch ext {
	case ".png":
		mime = "image/png"
	case ".jpg", ".jpeg":
		mime = "image/jpeg"
	case ".webp":
		mime = "image/webp"
	case ".gif":
		mime = "image/gif"
	case ".pdf":
		mime = "application/pdf"
	case ".txt", ".md", ".csv", ".go", ".py", ".html", ".json", ".yaml", ".yml":
		mime = "text/plain"
	case ".mp3", ".wav":
		mime = "audio/mpeg"
	case ".mp4":
		mime = "video/mp4"
	}
	return mime, base64.StdEncoding.EncodeToString(b), true
}

// fileToBase64 provides a streamlined conversion method for visual engines.
func fileToBase64(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("[System Warning] Failed to read resource: %v\n", err)
		return ""
	}
	mimeType := "image/jpeg"
	lowerPath := strings.ToLower(path)
	if strings.HasSuffix(lowerPath, ".png") {
		mimeType = "image/png"
	} else if strings.HasSuffix(lowerPath, ".webp") {
		mimeType = "image/webp"
	}
	return fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
}

// executeInDocker runs skill code inside a Docker container and returns the output.
func executeInDocker(db *sql.DB, code string, language string, args []string) (string, error) {
	var dockerEnabled, dockerHost, dockerImage string
	db.QueryRow("SELECT value FROM settings WHERE key='docker_enabled'").Scan(&dockerEnabled)
	if dockerEnabled != "true" {
		return "", fmt.Errorf("docker execution is disabled")
	}
	db.QueryRow("SELECT value FROM settings WHERE key='docker_host'").Scan(&dockerHost)
	db.QueryRow("SELECT value FROM settings WHERE key='docker_image'").Scan(&dockerImage)
	if dockerHost == "" || dockerImage == "" {
		return "", fmt.Errorf("docker host or image not configured")
	}

	var ext, entryCmd string
	switch language {
	case "Bash":
		ext = ".sh"
		entryCmd = "sh /code/skill" + ext
	case "Go":
		ext = ".go"
		entryCmd = "go run /code/skill" + ext
	case "JavaScript":
		ext = ".js"
		entryCmd = "node /code/skill" + ext
	default:
		return "", fmt.Errorf("unsupported language for docker: %s", language)
	}

	tmpDir, err := os.MkdirTemp("", "zuver_docker_")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	codeFile := filepath.Join(tmpDir, "skill"+ext)
	if err := os.WriteFile(codeFile, []byte(code), 0700); err != nil {
		return "", fmt.Errorf("failed to write skill file: %w", err)
	}

	argsSlice := []string{
		"-H", dockerHost,
		"run", "--rm",
		"-v", tmpDir + ":/code:ro",
		dockerImage,
		"sh", "-c", entryCmd,
	}

	cmd := exec.Command("docker", argsSlice...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("docker execution failed: %w", err)
	}
	return string(out), nil
}
