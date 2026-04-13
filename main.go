package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

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
	CommandTpl  string `json:"command_tpl"`
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
}

// ChatMessage represents a single message exchange within an agent's history.
type ChatMessage struct {
	ID        int    `json:"id"`
	AgentID   string `json:"agent_id"`
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
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
}

// DBTask encapsulates a database operation to be processed sequentially.
type DBTask struct {
	DB    *sql.DB
	Query string
	Args  []interface{}
}

// Thread-safe channels and mutexes for resolving concurrent SQLite locks and requests.
var (
	dbWriteQueue  = make(chan DBTask, 10000)
	adminToken    string
	loginAttempts = make(map[string]int)
	lockoutTime   = make(map[string]time.Time)
	authMu        sync.Mutex
	dbWriteMu     sync.Mutex
)

// SyncDBExec executes a synchronous database write protected by a global mutex.
func SyncDBExec(db *sql.DB, query string, args ...interface{}) (sql.Result, error) {
	dbWriteMu.Lock()
	defer dbWriteMu.Unlock()
	return db.Exec(query, args...)
}

// permissionScopeForPath maps an HTTP route to (resource, action) used in API key JSON.
// Actions: "read" (GET), "write" (POST/PUT), "delete" (DELETE), "execute" (POST on run/chat).
// Returns ("", "") if the route is allowed for any valid key.
// Returns ("admin_only", "") to block all external API keys unconditionally.
func permissionScopeForPath(method, path string) (resource string, action string) {
	// Strip /api/v1/ prefix and take first two path segments.
	trimmed := strings.TrimPrefix(path, "/api/v1/")
	parts := strings.SplitN(trimmed, "/", 3)
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
		// agents/{id}/run is an execute action.
		if len(parts) >= 3 && parts[2] == "run" {
			return "agents", "execute"
		}
		return "agents", toAction(method)
	case "chat":
		// Chat invocation is a special "execute" action on agents.
		return "agents", "execute"
	case "history":
		return "agents", "read"
	case "skills":
		return "skills", toAction(method)
	case "sources":
		// sources/update/{name} is a write-like push update.
		if len(parts) >= 2 && parts[1] == "update" {
			return "sources", "write"
		}
		return "sources", toAction(method)
	case "outputs":
		return "outputs", toAction(method)
	case "mcp":
		return "mcps", toAction(method)
	case "projects":
		// projects/{id}/run is an execute action.
		if len(parts) >= 3 && parts[2] == "run" {
			return "projects", "execute"
		}
		return "projects", toAction(method)
	case "preferences":
		return "preferences", toAction(method)
	case "rags":
		return "rags", toAction(method)
	case "providers", "apikeys", "settings", "wipe-database", "import":
		// Admin-only: never accessible via external API keys.
		return "admin_only", ""
	default:
		// stats, sysinfo, analytics, upload — permitted for any valid key.
		return "", ""
	}
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
		if adminToken != "" && token == adminToken {
			next.ServeHTTP(w, r)
			return
		}

		// Validate external API key and load its permission document.
		var permJSON string
		err := a.ConfigDB.QueryRow("SELECT permissions FROM api_keys WHERE token=?", token).Scan(&permJSON)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Unauthorized"}`, http.StatusUnauthorized)
			return
		}

		// Determine required scope for this endpoint.
		resource, action := permissionScopeForPath(r.Method, r.URL.Path)

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

		// Parse permissions and enforce resource + action-level check.
		var perms map[string]interface{}
		if json.Unmarshal([]byte(permJSON), &perms) != nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error": "Forbidden: malformed permissions"}`, http.StatusForbidden)
			return
		}

		allowed := false
		if val, ok := perms[resource]; ok {
			switch v := val.(type) {
			case map[string]interface{}:
				if resource == "preferences" {
					// Preferences use a nested map {pref_id: ["read","write"]}.
					// Any entry with the matching action grants access.
					for _, actionsRaw := range v {
						if actions, ok2 := actionsRaw.([]interface{}); ok2 {
							for _, a := range actions {
								if fmt.Sprintf("%v", a) == action {
									allowed = true
								}
							}
						}
					}
				} else {
					// Generic resource action map: {"read":true,"write":true,...}
					if boolVal, ok2 := v[action]; ok2 {
						if b, ok3 := boolVal.(bool); ok3 {
							allowed = b
						}
					}
				}
			case []interface{}:
				// Legacy list format: non-empty list grants read; write/delete require explicit flag.
				if action == "read" && len(v) > 0 {
					allowed = true
				} else if (action == "write" || action == "delete" || action == "execute") && len(v) > 0 {
					// For legacy keys without action maps, permit write if list is non-empty.
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

// handleLogin manages system authentication and rate-limiting against brute force attacks.
func (a *App) handleLogin(w http.ResponseWriter, r *http.Request) {
	// Security: limit body size to prevent DoS.
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req struct {
		Password string `json:"password"`
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

	// Strip port from RemoteAddr so the lockout key is IP-only.
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}

	authMu.Lock()
	defer authMu.Unlock()

	if time.Now().Before(lockoutTime[ip]) {
		http.Error(w, `{"error": "Too many failed attempts. Try again in 1 minute."}`, http.StatusTooManyRequests)
		return
	}

	var dbHash string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='admin_password'").Scan(&dbHash)

	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Password)))

	// First boot: initialise the admin password.
	if dbHash == "" {
		SyncDBExec(a.ConfigDB, "INSERT INTO settings (key, value) VALUES ('admin_password', ?)", hash)
		dbHash = hash
	}

	if hash == dbHash {
		loginAttempts[ip] = 0
		adminToken = fmt.Sprintf("tok_%d", time.Now().UnixNano())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"token": adminToken})
	} else {
		loginAttempts[ip]++
		if loginAttempts[ip] >= 5 {
			lockoutTime[ip] = time.Now().Add(1 * time.Minute)
		}
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
	var apiKey, endpoint string
	err := db.QueryRow("SELECT api_key, endpoint FROM providers WHERE type='OpenAI' LIMIT 1").Scan(&apiKey, &endpoint)
	if err != nil || apiKey == "" {
		return nil
	}

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

	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
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
	autoMigrateColumn(db, "agents", "can_create_skills", "INTEGER DEFAULT 0")
	autoMigrateColumn(db, "agents", "stream_enabled", "INTEGER DEFAULT 1")
	autoMigrateColumn(db, "projects", "tags", "TEXT DEFAULT ''")
	autoMigrateColumn(db, "projects", "is_active", "INTEGER DEFAULT 1")

	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('presidio_enabled', 'false'), ('presidio_analyzer', 'http://localhost:3000'), ('presidio_anonymizer', 'http://localhost:3001')")

	var pCount int
	db.QueryRow("SELECT COUNT(*) FROM projects").Scan(&pCount)
	if pCount == 0 {
		defaultFlow := `{"nodes":[{"id":"start","type":"Start","x":50,"y":150},{"id":"end","type":"End","x":450,"y":150}],"edges":[]}`
		db.Exec("INSERT INTO projects (id, name, description, flow_json) VALUES (?, ?, ?, ?)", "proj_default", "Default Router", "System default routing pipeline.", defaultFlow)
	}

	go startDBWorker()

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
	mux.HandleFunc("GET /api/v1/preferences", app.handleGetPreferences)
	mux.HandleFunc("POST /api/v1/preferences", app.handleCreatePreference)
	mux.HandleFunc("PUT /api/v1/preferences/{id}", app.handleUpdatePreference)
	mux.HandleFunc("DELETE /api/v1/preferences/{id}", app.handleDeletePreference)
	mux.HandleFunc("GET /api/v1/agents", app.handleGetAgents)
	mux.HandleFunc("POST /api/v1/agents", app.handleCreateAgent)
	mux.HandleFunc("PUT /api/v1/agents/{id}", app.handleUpdateAgent)
	mux.HandleFunc("DELETE /api/v1/agents/{id}", app.handleDeleteAgent)

	// ------------------------------------------------------------------
	// POST /api/v1/agents/{id}/run — standalone execution endpoint.
	// Accepts {"input":"..."} (or "message"), runs the agent pipeline
	// synchronously without streaming, returns {"reply":"...","logs":[]}.
	// Requires agents/execute scope for API key callers.
	// ------------------------------------------------------------------
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
	mux.HandleFunc("GET /api/v1/mcp", app.handleGetMCP)
	mux.HandleFunc("POST /api/v1/mcp", app.handleCreateMCP)
	mux.HandleFunc("DELETE /api/v1/mcp/{id}", app.handleDeleteMCP)
	mux.HandleFunc("GET /api/v1/settings", app.handleGetSettings)
	mux.HandleFunc("POST /api/v1/settings", app.handleUpdateSettings)
	mux.HandleFunc("POST /api/v1/chat", app.handleChat)
	mux.HandleFunc("GET /api/v1/history/{agent_id}", app.handleGetChatHistory)
	mux.HandleFunc("DELETE /api/v1/history/{agent_id}", app.handleClearChatHistory)

	mux.HandleFunc("GET /api/v1/apikeys", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query("SELECT id, name, description, token, permissions FROM api_keys")
		if err != nil {
			json.NewEncoder(w).Encode([]APIKey{})
			return
		}
		defer rows.Close()
		var list []APIKey
		for rows.Next() {
			var i APIKey
			rows.Scan(&i.ID, &i.Name, &i.Description, &i.Token, &i.Permissions)
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
		app.ConfigDB.Exec("INSERT INTO api_keys (id, name, description, token, permissions) VALUES (?, ?, ?, ?, ?)", i.ID, i.Name, i.Description, i.Token, i.Permissions)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("PUT /api/v1/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		var i APIKey
		json.NewDecoder(r.Body).Decode(&i)
		app.ConfigDB.Exec("UPDATE api_keys SET name=?, description=?, permissions=? WHERE id=?", i.Name, i.Description, i.Permissions, r.PathValue("id"))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /api/v1/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		app.ConfigDB.Exec("DELETE FROM api_keys WHERE id=?", r.PathValue("id"))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/v1/login", app.handleLogin)

	mux.HandleFunc("GET /api/v1/sysinfo", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"name":        "Zuver",
			"version":     "v1.1.2",
			"description": "Next-gen Generative AI Framework, built for secure.",
		})
	})

	mux.HandleFunc("GET /api/v1/analytics/summary", func(w http.ResponseWriter, r *http.Request) {
		rows, err := app.ConfigDB.Query(`
			SELECT entity_id, entity_type,
			       COALESCE(SUM(tokens), 0) as tokens_24h,
			       COUNT(*) as calls_24h,
			       MAX(timestamp) as last_call,
			       COALESCE(SUM(CASE WHEN is_success THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 100) as success_rate
			FROM analytics_logs
			WHERE timestamp >= datetime('now', '-1 day')
			GROUP BY entity_id, entity_type
		`)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"entities": map[string]interface{}{}, "global_tokens_24h": 0})
			return
		}
		defer rows.Close()

		type Stat struct {
			Tokens24h   int     `json:"tokens_24h"`
			Calls24h    int     `json:"calls_24h"`
			LastCall    string  `json:"last_call"`
			SuccessRate float64 `json:"success_rate"`
		}
		stats := make(map[string]Stat)
		for rows.Next() {
			var id, eType, lastCall string
			var tokens, calls int
			var rate float64
			rows.Scan(&id, &eType, &tokens, &calls, &lastCall, &rate)
			stats[id] = Stat{Tokens24h: tokens, Calls24h: calls, LastCall: lastCall, SuccessRate: rate}
		}

		var globalTokens int
		app.ConfigDB.QueryRow(`SELECT COALESCE(SUM(tokens), 0) FROM analytics_logs WHERE timestamp >= datetime('now', '-1 day')`).Scan(&globalTokens)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"entities": stats, "global_tokens_24h": globalTokens})
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
		tableNames := []string{"agents", "skills", "sources", "source_logs", "projects", "rags", "outputs", "mcp_servers", "chat_history", "tasks", "response_cache"}
		for _, t := range tableNames {
			// Security: table names are hardcoded — safe from injection.
			app.ConfigDB.Exec("DELETE FROM " + t)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// ------------------------------------------------------------------
	// /api/add — public confirmation page for one-click config import.
	// Usage: /api/add?configURL=https://...&type=skill|agent
	// The page fetches the remote JSON, shows a preview, and requires
	// the admin password before committing the import.
	// ------------------------------------------------------------------
	mux.HandleFunc("GET /api/add", func(w http.ResponseWriter, r *http.Request) {
		configURL := r.URL.Query().Get("configURL")
		itemType := r.URL.Query().Get("type")
		if configURL == "" || (itemType != "skill" && itemType != "agent") {
			http.Error(w, "Missing or invalid configURL / type parameter.", http.StatusBadRequest)
			return
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
		escapedType := itemType
		escapedConfigURL := configURL

		html := `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Import ` + escapedType + ` — Zuver</title>
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
    <span class="badge ` + escapedType + `">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
      ` + escapedType + `
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
        <button type="submit" class="btn btn-primary" id="confirm-btn">Import ` + escapedType + `</button>
        <button type="button" class="btn btn-danger" onclick="window.close()">Cancel</button>
      </div>
      <div id="msg" class="msg"></div>
    </form>
  </div>
  <div class="footer">Zuver Framework &mdash; Importing will add this ` + escapedType + ` to your instance permanently.</div>
</div>
<script>
const RAW_JSON = ` + "`" + safeJSONStr + "`" + `;
const ITEM_TYPE = "` + escapedType + `";
const CONFIG_URL = "` + escapedConfigURL + `";
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
    // Step 2: import via the API.
    const importResp = await fetch('/api/v1/import', {
      method: 'POST',
      headers: {'Content-Type':'application/json','Authorization':'Bearer '+token},
      body: JSON.stringify({type: ITEM_TYPE, data: JSON.parse(RAW_JSON)})
    });
    if (!importResp.ok) {
      const e = await importResp.json().catch(()=>({error:'Import failed'}));
      throw new Error(e.error || 'Import failed');
    }
    msg.textContent = '` + escapedType + ` imported successfully. You can close this page.';
    msg.className = 'msg success';
    btn.textContent = 'Imported';
  } catch(err) {
    msg.textContent = err.message;
    msg.className = 'msg error';
    btn.disabled = false;
    btn.textContent = 'Import ` + escapedType + `';
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
	log.Printf("Starting Zuver OS Framework on port %s", port)
	http.ListenAndServe(":"+port, loggingMiddleware(app.authMiddleware(mux)))
}

// --------------------------------------------------------------------------
// validateImportPayload enforces strict schema checks on agent/skill JSON
// before it is committed to the database, guarding against injection,
// oversized blobs, and structurally invalid configs.
// --------------------------------------------------------------------------

// allowedSkillTypes is the closed set of values the "type" field may hold.
var allowedSkillTypes = map[string]bool{
	"API": true, "Go": true, "Bash": true, "Python": true, "Text": true, "Prompt": true,
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
			return fmt.Errorf("skill type '%s' is not allowed; must be one of: API, Go, Bash, Python, Text, Prompt", skillType)
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

	AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "user", processedUserMsg)

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
		if userVector := getEmbedding(processedUserMsg, a.ConfigDB); userVector != nil {
			activeInjections += "\n[RAG Auto-Retrieved Memory]\n"
			for ragID := range allowedRAGs {
				var tName string
				a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&tName)
				if tName == "" {
					continue
				}
				rows, err := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data, vector FROM %s", tName))
				if err != nil {
					continue
				}
				var bestRecord, bestData string
				var highestScore float32 = -1.0
				for rows.Next() {
					var rName, rData, vJson string
					rows.Scan(&rName, &rData, &vJson)
					var dbVector []float32
					if json.Unmarshal([]byte(vJson), &dbVector) == nil && len(dbVector) > 0 {
						if score := cosineSimilarity(userVector, dbVector); score > highestScore {
							highestScore = score
							bestRecord = rName
							bestData = rData
						}
					}
				}
				rows.Close()
				if highestScore > 0.4 {
					activeInjections += fmt.Sprintf("- From %s (Match: %.2f%%): %s\n", bestRecord, highestScore*100, bestData)
				}
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
	messages := []map[string]interface{}{{"role": "system", "content": dynamicSysPrompt}}
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
			apiRole := role
			if role == "system" {
				apiRole = "user"
			}
			messages = append(messages, map[string]interface{}{"role": apiRole, "content": content})
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

		currentMessages := make([]map[string]interface{}, len(messages))
		copy(currentMessages, messages)

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

		// Build request body.
		reqBody := map[string]interface{}{
			"model":       agent.Model,
			"messages":    currentMessages,
			"temperature": agent.Temperature,
			"max_tokens":  agent.MaxTokens,
			"top_p":       agent.TopP,
			"stream":      useStream,
		}

		// Merge extra provider params (e.g., top_k, stop sequences).
		var extras map[string]interface{}
		if json.Unmarshal([]byte(prov.ExtraConfig), &extras) == nil {
			for k, v := range extras {
				reqBody[k] = v
			}
		}

		payloadBytes, _ := json.Marshal(reqBody)

		// Cache lookup (only for non-streaming, non-loop-0+ tool calls).
		cacheKey := hex.EncodeToString(sha256.New().Sum(payloadBytes))
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
					sseEmit(map[string]interface{}{"done": true, "error": "[SYSTEM] Failed to connect to the API provider.", "logs": executionLogs})
					return
				}
				defer resp.Body.Close()

				var fullReply strings.Builder
				scanner := bufio.NewScanner(resp.Body)
				for scanner.Scan() {
					line := scanner.Text()
					if !strings.HasPrefix(line, "data: ") {
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

					var delta string
					switch pType {
					case "claude":
						if deltaObj, ok := chunk["delta"].(map[string]interface{}); ok {
							delta, _ = deltaObj["text"].(string)
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
				// Collect complete reply for this loop iteration.
				replyContent = fullReply.String()
				totalTokensUsed += estimateTokens(string(payloadBytes)) + estimateTokens(replyContent)
			} else {
				// ---------- NON-STREAMING PATH ----------
				resp2, errDo2 := (&http.Client{Timeout: 120 * time.Second}).Do(apiReq)
				if errDo2 != nil {
					executionLogs = append(executionLogs, "[Network Error]: "+errDo2.Error())
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to connect to the API provider.", "logs": executionLogs})
					return
				}

				bodyBytes, _ := io.ReadAll(resp2.Body)
				resp2.Body.Close()

				var result2 map[string]interface{}
				json.Unmarshal(bodyBytes, &result2)

				if choices, ok := result2["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if msg, ok := choice["message"].(map[string]interface{}); ok {
							replyContent, _ = msg["content"].(string)
						}
					}
				} else if message, ok := result2["message"].(map[string]interface{}); ok {
					replyContent, _ = message["content"].(string)
				} else if contentArr, ok := result2["content"].([]interface{}); ok && len(contentArr) > 0 {
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
					executionLogs = append(executionLogs, "[API Refusal]: "+string(bodyBytes))
					a.logAnalytics("agent", agent.ID, 0, false)
					a.logAnalytics("provider", agent.ProviderID, 0, false)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[Error] API Provider returned no content.", "logs": executionLogs})
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
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "system", sysMsg)
			continue
		}

		messages = append(messages, map[string]interface{}{"role": "assistant", "content": replyContent})
		executionLogs = append(executionLogs, "[Agent]: "+replyContent)
		AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "assistant", replyContent)

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
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "system", combinedSysMsg)
			messages = append(messages, map[string]interface{}{"role": "user", "content": combinedSysMsg})
			continue
		}

		// No tool call — final response.
		AsyncDBExec(a.ConfigDB, "UPDATE agents SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ID)
		AsyncDBExec(a.ConfigDB, "UPDATE providers SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ProviderID)
		a.logAnalytics("agent", agent.ID, totalTokensUsed, true)
		a.logAnalytics("provider", agent.ProviderID, totalTokensUsed, true)

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
		var tName string
		a.ConfigDB.QueryRow("SELECT table_name FROM preferences WHERE id=?", prefID).Scan(&tName)
		if tName == "" {
			return "[SYSTEM ERROR] Preference DB not found."
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
		var tName string
		a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&tName)
		if tName == "" {
			return "[SYSTEM ERROR] RAG not found."
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
				execCmd := exec.Command("sh", "-c", mCmd)
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
			if err := a.ConfigDB.QueryRow("SELECT type, content, api_method, api_url, api_headers, api_body FROM skills WHERE name=?", toolName).
				Scan(&sType, &sContent, &sMethod, &sUrl, &sHeaders, &sBody); err != nil {
				return "[SKILL ERROR] Skill not found."
			}
			switch sType {
			case "MD":
				return "[INSTRUCTION]\n" + sContent
			case "Bash", "Go":
				fCode := sContent
				ph := extractPlaceholders(sContent)
				for i, arg := range agentArgs {
					if i < len(ph) {
						fCode = strings.ReplaceAll(fCode, ph[i], arg)
					}
				}
				var execCmd *exec.Cmd
				if sType == "Bash" {
					execCmd = exec.Command("sh", "-c", fCode)
				} else {
					tmp := filepath.Join(os.TempDir(), fmt.Sprintf("sk_%d.go", time.Now().UnixNano()))
					os.WriteFile(tmp, []byte(fCode), 0644)
					execCmd = exec.Command("go", "run", tmp)
					defer os.Remove(tmp)
				}
				out, e := execCmd.CombinedOutput()
				result := "[RESULT]\n" + string(out)
				if e != nil {
					result += "\nErr: " + e.Error()
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

	agentID := r.FormValue("agent_id")
	if agentID == "" {
		agentID = "global_temp"
	}

	// Security: sanitize original filename.
	safeBase := filepath.Base(filepath.Clean(header.Filename))
	fileName := fmt.Sprintf("%s___%d___%s", agentID, time.Now().UnixNano(), safeBase)
	path := filepath.Join("uploads", fileName)

	dst, err := os.Create(path)
	if err != nil {
		http.Error(w, `{"error":"failed to save file"}`, http.StatusInternalServerError)
		return
	}
	defer dst.Close()
	io.Copy(dst, file)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"path": path, "status": "ok"})
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

// handleGetChatHistory retrieves conversation history for a specific agent.
func (a *App) handleGetChatHistory(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, role, content, timestamp FROM chat_history WHERE agent_id=? ORDER BY id ASC", r.PathValue("agent_id"))
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]ChatMessage{})
		return
	}
	defer rows.Close()
	var list []ChatMessage
	for rows.Next() {
		var m ChatMessage
		m.AgentID = r.PathValue("agent_id")
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

func (a *App) handleGetStats(w http.ResponseWriter, r *http.Request) {
	var ag, sk, rg, tk int
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM agents").Scan(&ag)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM skills").Scan(&sk)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM rags").Scan(&rg)
	a.ConfigDB.QueryRow("SELECT COALESCE(SUM(token_usage), 0) FROM providers").Scan(&tk)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"agents": ag, "skills": sk, "rags": rg, "tokens": tk})
}

func (a *App) handleGetProviders(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, COALESCE(type,'OpenAI'), endpoint, api_key, COALESCE(extra_config,'{}'), token_usage FROM providers")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Provider{})
		return
	}
	defer rows.Close()
	var list []Provider
	for rows.Next() {
		var i Provider
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.Endpoint, &i.APIKey, &i.ExtraConfig, &i.TokenUsage)
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
	SyncDBExec(a.ConfigDB, "INSERT INTO providers (id, name, type, endpoint, api_key, extra_config) VALUES (?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Endpoint, i.APIKey, i.ExtraConfig)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateProvider(w http.ResponseWriter, r *http.Request) {
	var i Provider
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE providers SET name=?, type=?, endpoint=?, api_key=?, extra_config=? WHERE id=?", i.Name, i.Type, i.Endpoint, i.APIKey, i.ExtraConfig, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteProvider(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM providers WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSources(w http.ResponseWriter, r *http.Request) {
	rows, _ := a.ConfigDB.Query("SELECT id, name, type, COALESCE(api_key,''), COALESCE(file_path,'') FROM sources")
	defer rows.Close()
	var list []Source
	for rows.Next() {
		var i Source
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.APIKey, &i.FilePath)
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
	a.ConfigDB.Exec("INSERT INTO sources (id, name, type, api_key, file_path) VALUES (?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.APIKey, i.FilePath)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSource(w http.ResponseWriter, r *http.Request) {
	var i Source
	json.NewDecoder(r.Body).Decode(&i)
	if i.Type != "Local File" && i.APIKey == "" {
		i.APIKey = fmt.Sprintf("sk_src_%d", time.Now().UnixNano())
	}
	a.ConfigDB.Exec("UPDATE sources SET name=?, type=?, api_key=?, file_path=? WHERE id=?", i.Name, i.Type, i.APIKey, i.FilePath, r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteSource(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM sources WHERE id=?", r.PathValue("id"))
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
	if err := a.ConfigDB.QueryRow("SELECT id, api_key FROM sources WHERE name=?", name).Scan(&id, &dbKey); err != nil || (dbKey != "" && dbKey != reqKey) {
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
	rows, err := a.ConfigDB.Query("SELECT id, name, type, instruction, content, api_method, api_url, api_headers, api_body FROM skills")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Skill{})
		return
	}
	defer rows.Close()
	var list []Skill
	for rows.Next() {
		var i Skill
		rows.Scan(&i.ID, &i.Name, &i.Type, &i.Instruction, &i.Content, &i.APIMethod, &i.APIURL, &i.APIHeaders, &i.APIBody)
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
	a.ConfigDB.Exec("INSERT INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSkill(w http.ResponseWriter, r *http.Request) {
	var i Skill
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE skills SET name=?, type=?, instruction=?, content=?, api_method=?, api_url=?, api_headers=?, api_body=? WHERE id=?", i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody, r.PathValue("id"))
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
	i.TableName = fmt.Sprintf("rag_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO rags (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT, vector TEXT DEFAULT '[]')", i.TableName))
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
	a.MemoryDB.Exec("DROP TABLE IF EXISTS " + t)
	a.ConfigDB.Exec("DELETE FROM rags WHERE id=?", r.PathValue("id"))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetOutputs(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, instruction, command_tpl FROM outputs")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]Output{})
		return
	}
	defer rows.Close()
	var list []Output
	for rows.Next() {
		var i Output
		rows.Scan(&i.ID, &i.Name, &i.Instruction, &i.CommandTpl)
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
	a.ConfigDB.Exec("INSERT INTO outputs (id, name, instruction, command_tpl) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Instruction, i.CommandTpl)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateOutput(w http.ResponseWriter, r *http.Request) {
	var i Output
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE outputs SET name=?, instruction=?, command_tpl=? WHERE id=?", i.Name, i.Instruction, i.CommandTpl, r.PathValue("id"))
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
			&i.Outputs, &i.MCPs, &i.Projects, &i.SystemPrompt, &i.TokenUsage,
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
	a.ConfigDB.Exec(`INSERT INTO agents
		(id, name, provider_id, model, sources, skills, outputs, mcps, projects,
		 system_prompt, input_methods, output_methods, user_prompt_prefix,
		 temperature, max_tokens, top_p, privacy_enabled, can_create_skills, stream_enabled)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		i.ID, i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs,
		i.MCPs, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods,
		i.UserPromptPrefix, i.Temperature, i.MaxTokens, i.TopP,
		i.PrivacyEnabled, i.CanCreateSkills, i.StreamEnabled)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateAgent(w http.ResponseWriter, r *http.Request) {
	var i Agent
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec(`UPDATE agents SET
		name=?, provider_id=?, model=?, sources=?, skills=?, outputs=?, mcps=?,
		projects=?, system_prompt=?, input_methods=?, output_methods=?,
		user_prompt_prefix=?, temperature=?, max_tokens=?, top_p=?,
		privacy_enabled=?, can_create_skills=?, stream_enabled=?
		WHERE id=?`,
		i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs,
		i.MCPs, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods,
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

func (a *App) handleGetMCP(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, url, command FROM mcp_servers")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]MCPServer{})
		return
	}
	defer rows.Close()
	var list []MCPServer
	for rows.Next() {
		var i MCPServer
		rows.Scan(&i.ID, &i.Name, &i.URL, &i.Command)
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
	a.ConfigDB.Exec("INSERT INTO mcp_servers (id, name, url, command) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.URL, i.Command)
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
	i.TableName = fmt.Sprintf("pref_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO preferences (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT)", i.TableName))
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
	a.MemoryDB.Exec("DROP TABLE IF EXISTS " + t)
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

func (a *App) logAnalytics(entityType, entityID string, tokens int, success bool) {
	go func() {
		a.ConfigDB.Exec("INSERT INTO analytics_logs (entity_type, entity_id, tokens, is_success) VALUES (?, ?, ?, ?)", entityType, entityID, tokens, success)
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
				"model": agentModel, "messages": msgs,
				"temperature": agentTemp, "max_tokens": agentMaxTokens, "stream": false,
			}
			var extras map[string]interface{}
			if json.Unmarshal([]byte(agentExtraConfig), &extras) == nil {
				for k, v := range extras {
					body[k] = v
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
			a.ConfigDB.QueryRow("SELECT type, content, api_method, api_url, api_headers, api_body FROM skills WHERE id=?", sID).
				Scan(&sType, &sContent, &sMethod, &sUrl, &sHeaders, &sBody)

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
			case "MD":
				lastResult = substituteSkill(sContent)
			case "Bash", "Go":
				code := substituteSkill(sContent)
				var cmd *exec.Cmd
				if sType == "Bash" {
					cmd = exec.Command("sh", "-c", code)
				} else {
					tmp := filepath.Join(os.TempDir(), fmt.Sprintf("pipe_sk_%d.go", time.Now().UnixNano()))
					os.WriteFile(tmp, []byte(code), 0644)
					cmd = exec.Command("go", "run", tmp)
					defer os.Remove(tmp)
				}
				out, err := cmd.CombinedOutput()
				if err != nil {
					lastResult = fmt.Sprintf("[SKILL ERROR] %s\n%s", err.Error(), string(out))
				} else {
					lastResult = strings.TrimSpace(string(out))
				}
			case "API":
				fUrl := substituteSkill(sUrl)
				fBody := substituteSkill(sBody)
				apiR, _ := http.NewRequest(sMethod, fUrl, bytes.NewBuffer([]byte(fBody)))
				var hm map[string]string
				if json.Unmarshal([]byte(sHeaders), &hm) == nil {
					for k, v := range hm {
						apiR.Header.Set(k, v)
					}
				}
				extResp, e := (&http.Client{Timeout: 15 * time.Second}).Do(apiR)
				if e != nil {
					lastResult = "[SKILL API ERROR] " + e.Error()
				} else {
					b, _ := io.ReadAll(extResp.Body)
					extResp.Body.Close()
					lastResult = string(b)
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
				if re, err := regexp.Compile(value); err == nil {
					condMet = re.MatchString(lastResult)
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
	b, err := os.ReadFile(path)
	if err != nil {
		log.Printf("[I/O Error] Failed to read file: %v", err)
		return "", "", false
	}

	ext := strings.ToLower(filepath.Ext(path))
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
