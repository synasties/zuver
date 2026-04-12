package main

import (
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

// authMiddleware enforces API security and authenticates administrative or API requests.
func (a *App) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/") || r.URL.Path == "/api/login" {
			next.ServeHTTP(w, r)
			return
		}

		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if adminToken != "" && token == adminToken {
			next.ServeHTTP(w, r)
			return
		}

		var count int
		a.ConfigDB.QueryRow("SELECT COUNT(*) FROM api_keys WHERE token=?", token).Scan(&count)
		if count > 0 {
			next.ServeHTTP(w, r)
			return
		}

		http.Error(w, `{"error": "Unauthorized"}`, http.StatusUnauthorized)
	})
}

// handleLogin manages system authentication and rate-limiting against brute force attacks.
func (a *App) handleLogin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Password string `json:"password"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	ip := r.RemoteAddr

	authMu.Lock()
	defer authMu.Unlock()

	if time.Now().Before(lockoutTime[ip]) {
		http.Error(w, `{"error": "Too many failed attempts. Try again in 1 minute."}`, http.StatusTooManyRequests)
		return
	}

	var dbHash string
	a.ConfigDB.QueryRow("SELECT value FROM settings WHERE key='admin_password'").Scan(&dbHash)

	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Password)))

	if dbHash == "" {
		a.ConfigDB.Exec("INSERT INTO settings (key, value) VALUES ('admin_password', ?)", hash)
		dbHash = hash
	}

	if hash == dbHash {
		loginAttempts[ip] = 0
		adminToken = fmt.Sprintf("tok_%d", time.Now().UnixNano())
		json.NewEncoder(w).Encode(map[string]string{"token": adminToken})
	} else {
		loginAttempts[ip]++
		if loginAttempts[ip] >= 5 {
			lockoutTime[ip] = time.Now().Add(1 * time.Minute)
		}
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
	dbWriteQueue <- DBTask{DB: db, Query: query, Args: args}
}

var placeholderRe = regexp.MustCompile(`<([^>]+)>`)

// cosineSimilarity calculates the mathematical proximity between two vector embeddings.
func cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
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

	req, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
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
	json.NewDecoder(resp.Body).Decode(&res)
	if len(res.Data) > 0 {
		return res.Data[0].Embedding
	}
	return nil
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
	return fmt.Sprintf("%s\n\n[Page 1 of %d, response \"/nextPage\" to view next page.]", chunks[0], len(chunks))
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
		`CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, description TEXT, flow_json TEXT)`,
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
		db.Exec(q)
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

	db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('presidio_enabled', 'false'), ('presidio_analyzer', 'http://localhost:3000'), ('presidio_anonymizer', 'http://localhost:3001')")

	var pCount int
	db.QueryRow("SELECT COUNT(*) FROM projects").Scan(&pCount)
	if pCount == 0 {
		defaultFlow := `{"nodes":[{"id":"start","type":"Start","x":50,"y":150},{"id":"end","type":"End","x":450,"y":150}],"edges":[]}`
		db.Exec("INSERT INTO projects (id, name, description, flow_json) VALUES (?, ?, ?, ?)", "proj_default", "Default Router", "System default routing pipeline.", defaultFlow)
	}

	go startDBWorker()

	app := &App{ConfigDB: db, MemoryDB: ragDb}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.ServeFile(w, r, "index.html")
		} else {
			http.NotFound(w, r)
		}
	})
	mux.HandleFunc("GET /api/stats", app.handleGetStats)
	mux.HandleFunc("GET /api/providers", app.handleGetProviders)
	mux.HandleFunc("POST /api/providers", app.handleCreateProvider)
	mux.HandleFunc("PUT /api/providers/{id}", app.handleUpdateProvider)
	mux.HandleFunc("DELETE /api/providers/{id}", app.handleDeleteProvider)
	mux.HandleFunc("GET /api/sources", app.handleGetSources)
	mux.HandleFunc("POST /api/sources", app.handleCreateSource)
	mux.HandleFunc("PUT /api/sources/{id}", app.handleUpdateSource)
	mux.HandleFunc("GET /api/outputs", app.handleGetOutputs)
	mux.HandleFunc("POST /api/outputs", app.handleCreateOutput)
	mux.HandleFunc("PUT /api/outputs/{id}", app.handleUpdateOutput)
	mux.HandleFunc("DELETE /api/outputs/{id}", app.handleDeleteOutput)
	mux.HandleFunc("DELETE /api/sources/{id}", app.handleDeleteSource)
	mux.HandleFunc("POST /api/sources/update/{name}", app.handleSourceUpdate)
	mux.HandleFunc("GET /api/skills", app.handleGetSkills)
	mux.HandleFunc("POST /api/skills", app.handleCreateSkill)
	mux.HandleFunc("PUT /api/skills/{id}", app.handleUpdateSkill)
	mux.HandleFunc("DELETE /api/skills/{id}", app.handleDeleteSkill)
	mux.HandleFunc("GET /api/rags", app.handleGetRAGs)
	mux.HandleFunc("POST /api/rags", app.handleCreateRAG)
	mux.HandleFunc("PUT /api/rags/{id}", app.handleUpdateRAG)
	mux.HandleFunc("DELETE /api/rags/{id}", app.handleDeleteRAG)
	mux.HandleFunc("GET /api/preferences", app.handleGetPreferences)
	mux.HandleFunc("POST /api/preferences", app.handleCreatePreference)
	mux.HandleFunc("PUT /api/preferences/{id}", app.handleUpdatePreference)
	mux.HandleFunc("DELETE /api/preferences/{id}", app.handleDeletePreference)
	mux.HandleFunc("GET /api/agents", app.handleGetAgents)
	mux.HandleFunc("POST /api/agents", app.handleCreateAgent)
	mux.HandleFunc("PUT /api/agents/{id}", app.handleUpdateAgent)
	mux.HandleFunc("DELETE /api/agents/{id}", app.handleDeleteAgent)
	mux.HandleFunc("GET /api/projects", app.handleGetProjects)
	mux.HandleFunc("PUT /api/projects/{id}", app.handleUpdateProject)
	mux.HandleFunc("GET /api/mcp", app.handleGetMCP)
	mux.HandleFunc("POST /api/mcp", app.handleCreateMCP)
	mux.HandleFunc("DELETE /api/mcp/{id}", app.handleDeleteMCP)
	mux.HandleFunc("GET /api/settings", app.handleGetSettings)
	mux.HandleFunc("POST /api/settings", app.handleUpdateSettings)
	mux.HandleFunc("POST /api/chat", app.handleChat)
	mux.HandleFunc("GET /api/history/{agent_id}", app.handleGetChatHistory)
	mux.HandleFunc("DELETE /api/history/{agent_id}", app.handleClearChatHistory)

	mux.HandleFunc("GET /api/apikeys", func(w http.ResponseWriter, r *http.Request) {
		rows, _ := app.ConfigDB.Query("SELECT id, name, description, token, permissions FROM api_keys")
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
		json.NewEncoder(w).Encode(list)
	})

	mux.HandleFunc("POST /api/apikeys", func(w http.ResponseWriter, r *http.Request) {
		var i APIKey
		json.NewDecoder(r.Body).Decode(&i)
		i.ID = fmt.Sprintf("ak_%d", time.Now().UnixNano())
		i.Token = "zuv-" + hex.EncodeToString([]byte(i.ID))
		app.ConfigDB.Exec("INSERT INTO api_keys (id, name, description, token, permissions) VALUES (?, ?, ?, ?, ?)", i.ID, i.Name, i.Description, i.Token, i.Permissions)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("PUT /api/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		var i APIKey
		json.NewDecoder(r.Body).Decode(&i)
		app.ConfigDB.Exec("UPDATE api_keys SET name=?, description=?, permissions=? WHERE id=?", i.Name, i.Description, i.Permissions, r.PathValue("id"))
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /api/apikeys/{id}", func(w http.ResponseWriter, r *http.Request) {
		app.ConfigDB.Exec("DELETE FROM api_keys WHERE id=?", r.PathValue("id"))
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/login", app.handleLogin)
	mux.HandleFunc("GET /api/sysinfo", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{
			"name":        "Zuver",
			"version":     "v1.0.4",
			"description": "Next-gen Generative AI Framework, built for secure.",
		})
	})

	mux.HandleFunc("GET /api/analytics/summary", func(w http.ResponseWriter, r *http.Request) {
		rows, _ := app.ConfigDB.Query(`
			SELECT entity_id, entity_type,
			       COALESCE(SUM(tokens), 0) as tokens_24h,
			       COUNT(*) as calls_24h,
			       MAX(timestamp) as last_call,
			       COALESCE(SUM(CASE WHEN is_success THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 100) as success_rate
			FROM analytics_logs
			WHERE timestamp >= datetime('now', '-1 day')
			GROUP BY entity_id, entity_type
		`)
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

		json.NewEncoder(w).Encode(map[string]interface{}{"entities": stats, "global_tokens_24h": globalTokens})
	})

	mux.HandleFunc("POST /api/upload", app.handleFileUpload)
	os.MkdirAll("./uploads", 0755)

	mux.HandleFunc("GET /api/analytics/chart", func(w http.ResponseWriter, r *http.Request) {
		targetID := r.URL.Query().Get("id")

		query := `
		WITH RECURSIVE dates(d) AS (
			SELECT date('now', '-6 days')
			UNION ALL
			SELECT date(d, '+1 day') FROM dates WHERE d < date('now')
		)
		SELECT dates.d, COALESCE(SUM(a.tokens), 0), COUNT(a.id)
		FROM dates
		LEFT JOIN analytics_logs a ON date(a.timestamp) = dates.d 
		`
		var rows *sql.Rows
		var err error
		if targetID != "" && targetID != "global" {
			query += " AND a.entity_id = ? GROUP BY dates.d ORDER BY dates.d ASC"
			rows, err = app.ConfigDB.Query(query, targetID)
		} else {
			query += " GROUP BY dates.d ORDER BY dates.d ASC"
			rows, err = app.ConfigDB.Query(query)
		}

		if err != nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"labels": []string{}, "tokens": []int{}, "calls": []int{}})
			return
		}
		defer rows.Close()

		var labels []string
		var tokens, calls []int
		for rows.Next() {
			var d string
			var t, c int
			rows.Scan(&d, &t, &c)
			labels = append(labels, d)
			tokens = append(tokens, t)
			calls = append(calls, c)
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"labels": labels, "tokens": tokens, "calls": calls})
	})

	mux.HandleFunc("POST /api/wipe-database", func(w http.ResponseWriter, r *http.Request) {
		tables := []string{"agents", "skills", "sources", "source_logs", "projects", "rags", "outputs", "mcp_servers", "chat_history", "tasks", "response_cache"}
		for _, t := range tables {
			app.ConfigDB.Exec("DELETE FROM " + t)
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "18806"
	}
	log.Printf("Starting Zuver OS Framework on port %s", port)

	http.ListenAndServe(":"+port, loggingMiddleware(app.authMiddleware(mux)))

	go func() {
		for {
			time.Sleep(10 * time.Second)
			rows, _ := db.Query("SELECT id, agent_id, regex, command, repeat FROM tasks WHERE active=1")
			for rows.Next() {
				var tID, aID, regx, cmd string
				var repeat int
				rows.Scan(&tID, &aID, &regx, &cmd, &repeat)
				var lastMsg string
				db.QueryRow("SELECT content FROM chat_history ORDER BY id DESC LIMIT 1").Scan(&lastMsg)

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
}

// handleChat processes primary conversational payloads, routing the instructions appropriately.
func (a *App) handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		AgentID         string `json:"agent_id"`
		Message         string `json:"message"`
		FilePath        string `json:"file_path"`
		UseCache        bool   `json:"use_cache"`
		PresidioEnabled bool   `json:"presidio_enabled"`
		AnalyzerURL     string `json:"presidio_analyzer"`
		AnonymizerURL   string `json:"presidio_anonymizer"`
		InputType       string `json:"input_type"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	var agent Agent
	err := a.ConfigDB.QueryRow("SELECT id, name, provider_id, model, sources, skills, outputs, mcps, system_prompt, input_methods, output_methods, user_prompt_prefix, temperature, max_tokens, top_p, privacy_enabled FROM agents WHERE id=?", req.AgentID).
		Scan(&agent.ID, &agent.Name, &agent.ProviderID, &agent.Model, &agent.Sources, &agent.Skills, &agent.Outputs, &agent.MCPs, &agent.SystemPrompt, &agent.InputMethods, &agent.OutputMethods, &agent.UserPromptPrefix, &agent.Temperature, &agent.MaxTokens, &agent.TopP, &agent.PrivacyEnabled)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to find the selected Agent.", "logs": []string{}})
		return
	}

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

	var prov Provider
	errProv := a.ConfigDB.QueryRow("SELECT COALESCE(type, 'OpenAI'), endpoint, api_key, COALESCE(extra_config, '{}') FROM providers WHERE id=?", agent.ProviderID).Scan(&prov.Type, &prov.Endpoint, &prov.APIKey, &prov.ExtraConfig)
	if errProv != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to find the assigned model provider.", "logs": []string{}})
		return
	}

	var executionLogs []string
	processedUserMsg := req.Message

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

	allowedRAGs := make(map[string]bool)
	allowedPreferences := make(map[string]bool)
	allowedSources := make(map[string]bool)
	allowedDynamicTools := make(map[string]bool)
	helpPanel := "Commands:\n/getSourcesList - Check what you can access.\n/getSourceData <id> - Check the data of the Source you defined.\n/nextPage - Scroll current content to next page when supported.\n/getRAGList - Check all the RAG you can access.\n/getRAGDataList <id> - Check what's in the selected RAG.\n/addRAGData <id> <name> <data> - Add an record to the RAG, so you can check later.\n/deleteRAGData <id> <name> - Delete an record from RAG.\n/editRAGData <id> <name> <data> - Edit an record of the RAG.\n/getPrefList - Check all Preferences you can access.\n/getPrefDataList <id> - Check what's in the selected Preference.\n/addPrefData <id> <name> <data> - Add an record to Preference.\n/deletePrefData <id> <name> - Delete an record from Preference.\n/editPrefData <id> <name> <data> - Edit an record of the Preference.\n"

	if agent.CanCreateSkills {
		helpPanel += "/createSkill <name> <description> - Create native Go skill dynamically.\n"
	}

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
					b, err := os.ReadFile(sFile)
					if err == nil {
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
					cmdName, _ := n["param"].(string)
					helpPanel += fmt.Sprintf("%s - Execute Visual Project: %s\n", cmdName, pName)
				}
			}
		}
	}

	if len(allowedRAGs) > 0 && processedUserMsg != "" {
		userVector := getEmbedding(processedUserMsg, a.ConfigDB)
		if userVector != nil {
			activeInjections += "\n[RAG Auto-Retrieved Memory]\n"
			for ragID := range allowedRAGs {
				var tName string
				a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&tName)
				if tName != "" {
					rows, _ := a.MemoryDB.Query(fmt.Sprintf("SELECT record_name, data, vector FROM %s", tName))

					var bestRecord, bestData string
					var highestScore float32 = -1.0

					for rows.Next() {
						var rName, rData, vJson string
						rows.Scan(&rName, &rData, &vJson)
						var dbVector []float32
						if json.Unmarshal([]byte(vJson), &dbVector) == nil && len(dbVector) > 0 {
							score := cosineSimilarity(userVector, dbVector)
							if score > highestScore {
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
	}

	dynamicSysPrompt := fmt.Sprintf("%s\n\n[ENV]\nTime: %s\nOutput Methods: %s\n\n[PROTOCOL]\nTo use a tool, reply MUST be ONE line starting with '/'. e.g. /help", agent.SystemPrompt, time.Now().Format("2006-01-02 15:04:05"), agent.OutputMethods)
	if activeInjections != "" {
		dynamicSysPrompt += "\n\n[REAL-TIME DATA]\n" + activeInjections
	}

	messages := []map[string]interface{}{{"role": "system", "content": dynamicSysPrompt}}
	histRows, _ := a.ConfigDB.Query("SELECT role, content FROM (SELECT role, content, id FROM chat_history WHERE agent_id=? ORDER BY id DESC LIMIT 40) ORDER BY id ASC", agent.ID)
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

	totalTokensUsed := 0
	currentUserText := processedUserMsg
	if currentUserText == "" {
		currentUserText = "Please analyze the attached file."
	}

	var fileMime, fileData string
	var hasFile bool
	if req.FilePath != "" && req.InputType != "Text" {
		fileMime, fileData, hasFile = extractFilePayload(req.FilePath)
		if !hasFile {
			executionLogs = append(executionLogs, "[I/O Shield]: Failed to read file, falling back to pure text mode.")
		} else {
			executionLogs = append(executionLogs, fmt.Sprintf("[Modality]: Loaded %s payload.", fileMime))

			if fileMime == "text/plain" {
				rawBytes, err := base64.StdEncoding.DecodeString(fileData)
				if err == nil {
					baseName := filepath.Base(req.FilePath)
					originalName := baseName
					parts := strings.SplitN(baseName, "___", 3)
					if len(parts) == 3 {
						originalName = parts[2]
					}

					currentUserText = fmt.Sprintf("Attached File (%s):\n\n```\n%s\n```\n\n%s", originalName, string(rawBytes), currentUserText)
					hasFile = false
					executionLogs = append(executionLogs, fmt.Sprintf("[Modality]: Injected %s directly into prompt.", originalName))
				}
			}
		}
	}

	for loop := 0; loop < 5; loop++ {
		apiURL := strings.TrimRight(prov.Endpoint, "/")
		pType := strings.ToLower(prov.Type)

		var currentMessages []map[string]interface{}
		currentMessages = append(currentMessages, messages...)

		var finalUserMsg map[string]interface{}

		if hasFile {
			switch pType {
			case "claude":
				contentArray := []map[string]interface{}{}
				if strings.HasPrefix(fileMime, "image/") {
					contentArray = append(contentArray, map[string]interface{}{
						"type":   "image",
						"source": map[string]string{"type": "base64", "media_type": fileMime, "data": fileData},
					})
				} else if fileMime == "application/pdf" {
					contentArray = append(contentArray, map[string]interface{}{
						"type":   "document",
						"source": map[string]string{"type": "base64", "media_type": fileMime, "data": fileData},
					})
				} else {
					currentUserText = fmt.Sprintf("[File %s]\n", fileData, fileMime, currentUserText)
				}
				contentArray = append(contentArray, map[string]interface{}{"type": "text", "text": currentUserText})
				finalUserMsg = map[string]interface{}{"role": "user", "content": contentArray}

			case "ollama":
				if strings.HasPrefix(fileMime, "image/") {
					finalUserMsg = map[string]interface{}{
						"role":    "user",
						"content": currentUserText,
						"images":  []string{fileData},
					}
				} else {
					currentUserText = fmt.Sprintf("[System Note: Attached %s file, unsupported natively.]\n%s", fileMime, currentUserText)
					finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
				}

			default:
				if strings.HasPrefix(fileMime, "image/") {
					contentArray := []map[string]interface{}{
						{"type": "text", "text": currentUserText},
						{"type": "image_url", "image_url": map[string]string{"url": fmt.Sprintf("data:%s;base64,%s", fileMime, fileData)}},
					}
					finalUserMsg = map[string]interface{}{"role": "user", "content": contentArray}
				} else {
					currentUserText = fmt.Sprintf("[System Note: User attached a %s file, but standard chat completion does not support it natively.]\n%s", fileMime, currentUserText)
					finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
				}
			}
		} else {
			finalUserMsg = map[string]interface{}{"role": "user", "content": currentUserText}
		}

		currentMessages = append(currentMessages, finalUserMsg)

		reqBody := map[string]interface{}{
			"model":       agent.Model,
			"messages":    currentMessages,
			"temperature": agent.Temperature,
			"max_tokens":  agent.MaxTokens,
			"top_p":       agent.TopP,
			"stream":      false,
		}

		var extras map[string]interface{}
		if json.Unmarshal([]byte(prov.ExtraConfig), &extras) == nil {
			for k, v := range extras {
				reqBody[k] = v
			}
		}

		if pType == "claude" {
			if !strings.HasSuffix(apiURL, "/v1/messages") {
				apiURL += "/v1/messages"
			}
		} else if pType == "ollama" {
			if !strings.HasSuffix(apiURL, "/api/chat") {
				apiURL += "/api/chat"
			}
		} else {
			if !strings.HasSuffix(apiURL, "/v1/chat/completions") {
				apiURL += "/v1/chat/completions"
			}
		}

		if pType == "claude" && req.InputType == "Image" && req.FilePath != "" {
			mimeType := "image/jpeg"
			if strings.HasSuffix(strings.ToLower(req.FilePath), ".png") {
				mimeType = "image/png"
			} else if strings.HasSuffix(strings.ToLower(req.FilePath), ".webp") {
				mimeType = "image/webp"
			}

			lastMsgIndex := len(messages) - 1
			messages[lastMsgIndex]["content"] = []map[string]interface{}{
				{"type": "text", "text": processedUserMsg},
				{"type": "image", "source": map[string]string{
					"type":       "base64",
					"media_type": mimeType,
					"data":       fileData,
				}},
			}
			reqBody["messages"] = messages
		}
		payloadBytes, _ := json.Marshal(reqBody)

		hash := sha256.Sum256(payloadBytes)
		cacheKey := hex.EncodeToString(hash[:])

		var replyContent string
		var cachedReply string
		errCache := a.ConfigDB.QueryRow("SELECT response FROM response_cache WHERE hash=?", cacheKey).Scan(&cachedReply)

		if req.UseCache && errCache == nil && cachedReply != "" {
			replyContent = cachedReply
			executionLogs = append(executionLogs, "[Cache Hit]: Hit cached response.")
		} else {
			apiReq, errReq := http.NewRequest("POST", apiURL, bytes.NewBuffer(payloadBytes))

			if errReq != nil {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to generate API request.", "logs": append(executionLogs, errReq.Error())})
				return
			}

			apiReq.Header.Set("Content-Type", "application/json")
			if pType == "claude" {
				apiReq.Header.Set("x-api-key", prov.APIKey)
				apiReq.Header.Set("anthropic-version", "2023-06-01")
			} else if pType != "ollama" {
				apiReq.Header.Set("Authorization", "Bearer "+prov.APIKey)
			}

			resp, errDo := (&http.Client{Timeout: 60 * time.Second}).Do(apiReq)

			if errDo != nil {
				executionLogs = append(executionLogs, "[Network Error]: "+errDo.Error())
				a.logAnalytics("agent", agent.ID, 0, false)
				a.logAnalytics("provider", agent.ProviderID, 0, false)
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Failed to connect to the API provider.", "logs": executionLogs})
				return
			}

			bodyBytes, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			var result map[string]interface{}
			json.Unmarshal(bodyBytes, &result)

			if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if msg, ok := choice["message"].(map[string]interface{}); ok {
						replyContent, _ = msg["content"].(string)
					}
				}
			} else if message, ok := result["message"].(map[string]interface{}); ok {
				replyContent, _ = message["content"].(string)
			} else if contentArr, ok := result["content"].([]interface{}); ok && len(contentArr) > 0 {
				var combinedText []string
				for _, b := range contentArr {
					if block, ok := b.(map[string]interface{}); ok {
						if txt, ok := block["text"].(string); ok && txt != "" {
							combinedText = append(combinedText, txt)
						}
						if block["type"] == "tool_use" {
							toolName, _ := block["name"].(string)
							inputData, _ := json.Marshal(block["input"])
							combinedText = append(combinedText, fmt.Sprintf("/native_tool_%s %s", toolName, string(inputData)))
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
				json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[Error] API Provider returned errors.", "logs": executionLogs})
				return
			}

			if req.UseCache {
				AsyncDBExec(a.ConfigDB, "INSERT OR REPLACE INTO response_cache (hash, response) VALUES (?, ?)", cacheKey, replyContent)
			}
			totalTokensUsed += estimateTokens(string(payloadBytes)) + estimateTokens(replyContent)
		}

		replyContent = strings.TrimSpace(replyContent)
		totalTokensUsed += estimateTokens(string(payloadBytes)) + estimateTokens(replyContent)

		skillEditorMu.Lock()
		editorState, isEditing := skillEditorStore[agent.ID]
		if isEditing {
			delete(skillEditorStore, agent.ID)
		}
		skillEditorMu.Unlock()

		if isEditing {
			if replyContent == "/cancel" {
				sysMsg := "[SYSTEM] Skill creation cancelled."
				messages = append(messages, map[string]interface{}{"role": "user", "content": sysMsg})
				continue
			}

			cleanCode := replyContent
			if strings.HasPrefix(cleanCode, "```go") {
				cleanCode = strings.TrimPrefix(cleanCode, "```go")
			}
			if strings.HasPrefix(cleanCode, "```") {
				cleanCode = strings.TrimPrefix(cleanCode, "```")
			}
			if strings.HasSuffix(cleanCode, "```") {
				cleanCode = strings.TrimSuffix(cleanCode, "```")
			}
			cleanCode = strings.TrimSpace(cleanCode)

			skillID := "sk_" + fmt.Sprint(time.Now().UnixNano())
			a.ConfigDB.Exec("INSERT INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", skillID, editorState["name"], "Go", editorState["desc"], cleanCode, "GET", "", "", "")

			var currentSkills []string
			json.Unmarshal([]byte(agent.Skills), &currentSkills)
			currentSkills = append(currentSkills, skillID)
			newSkillsJson, _ := json.Marshal(currentSkills)
			a.ConfigDB.Exec("UPDATE agents SET skills=? WHERE id=?", string(newSkillsJson), agent.ID)
			agent.Skills = string(newSkillsJson)
			allowedDynamicTools[editorState["name"]] = true

			sysMsg := fmt.Sprintf("[SYSTEM SUCCESS] Native Go Skill '%s' compiled and mounted. You can now use /%s <args>. Please inform the user.", editorState["name"], editorState["name"])
			messages = append(messages, map[string]interface{}{"role": "user", "content": sysMsg})
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "system", sysMsg)
			continue
		}
		messages = append(messages, map[string]interface{}{"role": "assistant", "content": replyContent})
		executionLogs = append(executionLogs, "[Agent]: "+replyContent)
		AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "assistant", replyContent)

		var cmdLines []string
		for _, line := range strings.Split(replyContent, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "/") {
				cmdLines = append(cmdLines, line)
			}
		}

		if len(cmdLines) > 0 {
			var wg sync.WaitGroup
			var mu sync.Mutex
			var combinedSysMsg string

			for _, cmdLine := range cmdLines {
				wg.Add(1)
				go func(cLine string) {
					defer wg.Done()
					parts := strings.Fields(cLine)
					cmd := parts[0]
					var sysMsg string

					isProjectCmd := false
					for _, pid := range projIDs {
						var pFlow string
						a.ConfigDB.QueryRow("SELECT flow_json FROM projects WHERE id=?", pid).Scan(&pFlow)
						if strings.Contains(pFlow, cmd) {
							executionLogs = append(executionLogs, "[Pipeline]: Running Visual Router ["+pid+"]...")
							sysMsg = a.runProjectPipeline(pid, agent.ID, cLine)
							isProjectCmd = true
							break
						}
					}

					if isProjectCmd {
					} else if cmd == "/help" {
						sysMsg = "[SYSTEM]\n" + helpPanel
					} else if cmd == "/nextPage" {
						paginationMu.Lock()
						chunks, exists := paginationStore[agent.ID]
						if !exists {
							sysMsg = "[SYSTEM ERROR] No pagination."
						} else {
							idx := paginationStep[agent.ID] + 1
							if idx >= len(chunks) {
								sysMsg = "[SYSTEM RESULT] End of data."
							} else {
								paginationStep[agent.ID] = idx
								sysMsg = fmt.Sprintf("%s\n\n[Page %d/%d, use /nextPage for more]", chunks[idx], idx+1, len(chunks))
							}
						}
						paginationMu.Unlock()
					} else if cmd == "/getSourcesList" {
						res := "[AVAILABLE SOURCES]\n"
						for s := range allowedSources {
							res += "- " + s + "\n"
						}
						sysMsg = res
					} else if cmd == "/getSourceData" && len(parts) >= 2 {
						sName := parts[1]
						if !allowedSources[sName] {
							sysMsg = "[SYSTEM ERROR] Unauthorized."
						} else {
							var sID, sType, sFile string
							var sPageSize int
							a.ConfigDB.QueryRow("SELECT id, type, page_size, file_path FROM sources WHERE name=?", sName).Scan(&sID, &sType, &sPageSize, &sFile)
							var finalData string
							if sType == "File" {
								b, err := os.ReadFile(sFile)
								if err != nil {
									finalData = "Error: " + err.Error()
								} else {
									finalData = string(b)
								}
							} else {
								var d, ts string
								if a.ConfigDB.QueryRow("SELECT data, timestamp FROM source_logs WHERE source_id=? ORDER BY id DESC LIMIT 1", sID).Scan(&d, &ts) != nil {
									finalData = "No data."
								} else {
									finalData = fmt.Sprintf("[DATA AT %s]\n%s", ts, d)
								}
							}
							sysMsg = triggerPagination(agent.ID, finalData, sPageSize)
						}
					} else if cmd == "/getPrefList" {
						res := "[PREFERENCES DATABASES]\n"
						for prefID := range allowedPreferences {
							var nm, ds string
							if a.ConfigDB.QueryRow("SELECT name, description FROM preferences WHERE id=?", prefID).Scan(&nm, &ds) == nil {
								res += fmt.Sprintf("- ID: %s | Name: %s\n", prefID, nm)
							}
						}
						sysMsg = res
					} else if cmd == "/getPrefDataList" || cmd == "/addPrefData" || cmd == "/deletePrefData" || cmd == "/editPrefData" {
						if len(parts) < 2 {
							sysMsg = "Missing <PrefID>"
						} else {
							prefID := parts[1]
							if !allowedPreferences[prefID] {
								sysMsg = "Unauthorized."
							} else {
								var tName string
								a.ConfigDB.QueryRow("SELECT table_name FROM preferences WHERE id=?", prefID).Scan(&tName)
								if tName == "" {
									sysMsg = "Preference DB not found."
								} else {
									if cmd == "/getPrefDataList" {
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
										sysMsg = res
									} else if cmd == "/addPrefData" && len(parts) >= 4 {
										_, err := a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data) VALUES (?, ?)", tName), parts[2], strings.Join(parts[3:], " "))
										if err != nil {
											sysMsg = "[RESULT] Error: " + err.Error()
										} else {
											sysMsg = "[RESULT] Record Added."
										}
									} else if cmd == "/editPrefData" && len(parts) >= 4 {
										_, err := a.MemoryDB.Exec(fmt.Sprintf("UPDATE %s SET data=? WHERE record_name=?", tName), strings.Join(parts[3:], " "), parts[2])
										if err != nil {
											sysMsg = "[RESULT] Error: " + err.Error()
										} else {
											sysMsg = "[RESULT] Record Updated."
										}
									} else if cmd == "/deletePrefData" && len(parts) >= 3 {
										a.MemoryDB.Exec(fmt.Sprintf("DELETE FROM %s WHERE record_name=?", tName), parts[2])
										sysMsg = "[RESULT] Record Deleted."
									}
								}
							}
						}
					} else if cmd == "/getRAGList" {
						res := "[RAG DATABASES]\n"
						for ragID := range allowedRAGs {
							var nm, ds string
							if a.ConfigDB.QueryRow("SELECT name, description FROM rags WHERE id=?", ragID).Scan(&nm, &ds) == nil {
								res += fmt.Sprintf("- ID: %s | Name: %s\n", ragID, nm)
							}
						}
						sysMsg = res
					} else if cmd == "/createSkill" {
						if agent.CanCreateSkills {
							if len(parts) >= 3 {
								skillEditorMu.Lock()
								skillEditorStore[agent.ID] = map[string]string{
									"name": parts[1],
									"desc": strings.Join(parts[2:], " "),
								}
								skillEditorMu.Unlock()
								sysMsg = "[SYSTEM] ENTERING NATIVE GO CODE EDITOR MODE.\nYour next reply MUST contain ONLY valid Go code. No markdown formatting is required, but accepted. Reply /cancel to abort."
							} else {
								sysMsg = "[SYSTEM ERROR] Usage: /createSkill <name> <description>"
							}
						} else {
							sysMsg = "[SYSTEM ERROR] Unauthorized. You do not have permission to create skills."
						}
					} else if cmd == "/createTasks" {
						parts := strings.SplitN(cLine, " ", 4)
						if len(parts) >= 4 {
							repeat := 0
							if parts[3] == "true" {
								repeat = 1
							}
							AsyncDBExec(a.ConfigDB, "INSERT INTO tasks (id, agent_id, regex, command, repeat) VALUES (?, ?, ?, ?, ?)", "tsk_"+fmt.Sprint(time.Now().UnixNano()), agent.ID, parts[1], parts[2], repeat)
							sysMsg = "[TASK SYSTEM] Background Job Deployed. Monitoring pattern: " + parts[1]
						} else {
							sysMsg = "[SYSTEM ERROR] Usage: /createTasks <regex> <command> <true/false>"
						}
					} else if cmd == "/getRAGDataList" || cmd == "/addRAGData" || cmd == "/deleteRAGData" || cmd == "/editRAGData" {
						if len(parts) < 2 {
							sysMsg = "Missing <RAGID>"
						} else {
							ragID := parts[1]
							if !allowedRAGs[ragID] {
								sysMsg = "Unauthorized."
							} else {
								var tName string
								a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", ragID).Scan(&tName)
								if tName == "" {
									sysMsg = "RAG not found."
								} else {
									if cmd == "/getRAGDataList" {
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
										sysMsg = res
									} else if cmd == "/addRAGData" && len(parts) >= 4 {
										ragDataStr := strings.Join(parts[3:], " ")
										vector := getEmbedding(ragDataStr, a.ConfigDB)
										vectorJSON := "[]"
										if vector != nil {
											vBytes, _ := json.Marshal(vector)
											vectorJSON = string(vBytes)
										}

										_, err := a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), parts[2], ragDataStr, vectorJSON)
										if err != nil && strings.Contains(err.Error(), "has no column named vector") {
											a.MemoryDB.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN vector TEXT DEFAULT '[]'", tName))
											_, err = a.MemoryDB.Exec(fmt.Sprintf("INSERT INTO %s (record_name, data, vector) VALUES (?, ?, ?)", tName), parts[2], ragDataStr, vectorJSON)
										}

										if err != nil {
											sysMsg = "[RESULT] Error: " + err.Error()
										} else {
											sysMsg = "[RESULT] Record & Vector Embedding Added."
										}
									} else if cmd == "/editRAGData" && len(parts) >= 4 {
										_, err := a.MemoryDB.Exec(fmt.Sprintf("UPDATE %s SET data=? WHERE record_name=?", tName), strings.Join(parts[3:], " "), parts[2])
										if err != nil {
											sysMsg = "[RESULT] Error: " + err.Error()
										} else {
											sysMsg = "[RESULT] Record Updated."
										}
									} else if cmd == "/deleteRAGData" && len(parts) >= 3 {
										a.MemoryDB.Exec(fmt.Sprintf("DELETE FROM %s WHERE record_name=?", tName), parts[2])
										sysMsg = "[RESULT] Record Deleted."
									}
								}
							}
						}
					} else if strings.HasPrefix(cmd, "/mcp_") {
						mcpName := strings.TrimPrefix(cmd, "/mcp_")
						var mURL, mCmd string
						err := a.ConfigDB.QueryRow("SELECT url, command FROM mcp_servers WHERE name=?", mcpName).Scan(&mURL, &mCmd)
						if err != nil {
							sysMsg = "[MCP ERROR] Server not configured: " + mcpName
						} else {
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
								"params": map[string]interface{}{
									"name":      toolName,
									"arguments": argsData,
								},
							}
							reqBytes, _ := json.Marshal(requestPayload)

							if mURL != "" {
								apiReq, _ := http.NewRequest("POST", mURL, bytes.NewBuffer(reqBytes))
								apiReq.Header.Set("Content-Type", "application/json")
								resp, e := (&http.Client{Timeout: 30 * time.Second}).Do(apiReq)
								if e != nil {
									sysMsg = "[MCP HTTP ERROR] " + e.Error()
								} else {
									b, _ := io.ReadAll(resp.Body)
									resp.Body.Close()
									sysMsg = fmt.Sprintf("[MCP %s Response]\n%s", mcpName, string(b))
								}
							} else if mCmd != "" {
								reqStr := string(reqBytes) + "\n"
								execCmd := exec.Command("sh", "-c", mCmd)

								stdin, _ := execCmd.StdinPipe()
								stdout, _ := execCmd.StdoutPipe()

								execCmd.Start()

								stdin.Write([]byte(reqStr))
								stdin.Close()

								outBytes, e := io.ReadAll(stdout)
								execCmd.Wait()

								if e != nil {
									sysMsg = "[MCP STDIO ERROR] " + e.Error()
								} else {
									sysMsg = fmt.Sprintf("[MCP %s Response]\n%s", mcpName, string(outBytes))
								}
							} else {
								sysMsg = "[MCP ERROR] Both URL and Command are empty for " + mcpName
							}
						}
					} else {
						toolName := strings.TrimPrefix(cmd, "/")
						if allowedDynamicTools[toolName] {
							agentArgs := parts[1:]
							var sType, sContent, sMethod, sUrl, sHeaders, sBody string
							if a.ConfigDB.QueryRow("SELECT type, content, api_method, api_url, api_headers, api_body FROM skills WHERE name=?", toolName).Scan(&sType, &sContent, &sMethod, &sUrl, &sHeaders, &sBody) == nil {
								if sType == "MD" {
									sysMsg = "[INSTRUCTION]\n" + sContent
								} else if sType == "Bash" || sType == "Go" {
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
									sysMsg = "[RESULT]\n" + string(out)
									if e != nil {
										sysMsg += "\nErr: " + e.Error()
									}
								} else if sType == "API" {
									fUrl := sUrl
									fHead := sHeaders
									fBody := sBody
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
										sysMsg = "[API ERROR] " + e.Error()
									} else {
										b, _ := io.ReadAll(extResp.Body)
										extResp.Body.Close()
										sysMsg = "[API RESULT]\n" + string(b)
									}
								}
							} else {
								sysMsg = "Execution failed."
							}
						} else {
							sysMsg = "Command not found. Use /help"
						}
					}

					mu.Lock()
					combinedSysMsg += fmt.Sprintf("[Command Executed: %s]\n%s\n\n", cLine, sysMsg)
					executionLogs = append(executionLogs, "[Parallel Worker]: "+cmd+" finished.")
					mu.Unlock()
				}(cmdLine)
			}

			wg.Wait()
			combinedSysMsg = strings.TrimSpace(combinedSysMsg)
			AsyncDBExec(a.ConfigDB, "INSERT INTO chat_history (agent_id, role, content) VALUES (?, ?, ?)", agent.ID, "system", combinedSysMsg)

			messages = append(messages, map[string]interface{}{"role": "user", "content": combinedSysMsg})
			continue
		}

		AsyncDBExec(a.ConfigDB, "UPDATE agents SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ID)
		AsyncDBExec(a.ConfigDB, "UPDATE providers SET token_usage = token_usage + ? WHERE id = ?", totalTokensUsed, agent.ProviderID)

		a.logAnalytics("agent", agent.ID, totalTokensUsed, true)
		a.logAnalytics("provider", agent.ProviderID, totalTokensUsed, true)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"reply": replyContent, "logs": executionLogs})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"reply": "[SYSTEM] Agent has continously repeated the command for 5 times, stopping the agent...", "logs": executionLogs})
}

// handleFileUpload processes multipart payload files securely onto disk storage.
func (a *App) handleFileUpload(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(10 << 20)
	if err != nil {
		http.Error(w, "Invalid request", 400)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Invalid file", 400)
		return
	}
	defer file.Close()

	agentID := r.FormValue("agent_id")
	if agentID == "" {
		agentID = "global_temp" // 防呆机制，以防前端没传
	}

	fileName := fmt.Sprintf("%s___%d___%s", agentID, time.Now().UnixNano(), filepath.Base(header.Filename))
	path := filepath.Join("uploads", fileName)

	dst, err := os.Create(path)
	if err != nil {
		http.Error(w, "Failed to save file", 500)
		return
	}
	defer dst.Close()
	io.Copy(dst, file)

	json.NewEncoder(w).Encode(map[string]string{"path": path, "status": "ok"})
}

// autoMigrateColumn safely ensures column schema synchronization without truncating data.
func autoMigrateColumn(db *sql.DB, table string, column string, colDef string) {
	var name string
	err := db.QueryRow(fmt.Sprintf("SELECT name FROM pragma_table_info('%s') WHERE name='%s'", table, column)).Scan(&name)
	if err != nil {
		_, execErr := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, colDef))
		if execErr == nil {
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
	json.NewEncoder(w).Encode(list)
}

// handleClearChatHistory purges memory allocations assigned to a specific agent entity.
func (a *App) handleClearChatHistory(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agent_id")
	a.ConfigDB.Exec("DELETE FROM chat_history WHERE agent_id=?", agentID)

	paginationMu.Lock()
	delete(paginationStore, agentID)
	paginationMu.Unlock()

	files, err := os.ReadDir("./uploads")
	if err == nil {
		prefix := agentID + "___"
		for _, f := range files {
			if !f.IsDir() && strings.HasPrefix(f.Name(), prefix) {
				filePath := filepath.Join("./uploads", f.Name())
				os.Remove(filePath)
			}
		}
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleGetStats aggregates metrics utilized for the global dashboard rendering.
func (a *App) handleGetStats(w http.ResponseWriter, r *http.Request) {
	var ag, sk, rg, tk int
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM agents").Scan(&ag)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM skills").Scan(&sk)
	a.ConfigDB.QueryRow("SELECT COUNT(*) FROM rags").Scan(&rg)
	a.ConfigDB.QueryRow("SELECT COALESCE(SUM(token_usage), 0) FROM providers").Scan(&tk)
	json.NewEncoder(w).Encode(map[string]int{"agents": ag, "skills": sk, "rags": rg, "tokens": tk})
}

func (a *App) handleGetProviders(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, COALESCE(type, 'OpenAI'), endpoint, api_key, COALESCE(extra_config, '{}'), token_usage FROM providers")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateProvider(w http.ResponseWriter, r *http.Request) {
	var i Provider
	json.NewDecoder(r.Body).Decode(&i)
	SyncDBExec(a.ConfigDB, "INSERT INTO providers (id, name, type, endpoint, api_key, extra_config) VALUES (?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Endpoint, i.APIKey, i.ExtraConfig)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteProvider(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM providers WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSources(w http.ResponseWriter, r *http.Request) {
	rows, _ := a.ConfigDB.Query("SELECT id, name, type, COALESCE(api_key, ''), COALESCE(file_path, '') FROM sources")
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
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteSource(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM sources WHERE id=?", r.PathValue("id"))
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
	err := a.ConfigDB.QueryRow("SELECT id, api_key FROM sources WHERE name=?", name).Scan(&id, &dbKey)

	if err != nil || (dbKey != "" && dbKey != reqKey) {
		http.Error(w, `{"error": "Unauthorized or Invalid API Key"}`, 401)
		return
	}

	body, _ := io.ReadAll(r.Body)
	ts := time.Now().Format("2006-01-02 15:04:05")
	AsyncDBExec(a.ConfigDB, "INSERT INTO source_logs (source_id, data, timestamp) VALUES (?, ?, ?)", id, string(body), ts)

	rows, _ := a.ConfigDB.Query("SELECT id, sources FROM agents")
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
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetSkills(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, type, instruction, content, api_method, api_url, api_headers, api_body FROM skills")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateSkill(w http.ResponseWriter, r *http.Request) {
	var i Skill
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("INSERT INTO skills (id, name, type, instruction, content, api_method, api_url, api_headers, api_body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteSkill(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM skills WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetRAGs(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, table_name FROM rags")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateRAG(w http.ResponseWriter, r *http.Request) {
	var i RAG
	json.NewDecoder(r.Body).Decode(&i)
	i.TableName = fmt.Sprintf("rag_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO rags (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT)", i.TableName))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteRAG(w http.ResponseWriter, r *http.Request) {
	var t string
	a.ConfigDB.QueryRow("SELECT table_name FROM rags WHERE id=?", r.PathValue("id")).Scan(&t)
	a.MemoryDB.Exec("DROP TABLE IF EXISTS " + t)
	a.ConfigDB.Exec("DELETE FROM rags WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetOutputs(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, instruction, command_tpl FROM outputs")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateOutput(w http.ResponseWriter, r *http.Request) {
	var i Output
	json.NewDecoder(r.Body).Decode(&i)
	if i.ID == "" {
		i.ID = fmt.Sprintf("out_%d", time.Now().UnixNano())
	}
	a.ConfigDB.Exec("INSERT INTO outputs (id, name, instruction, command_tpl) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Instruction, i.CommandTpl)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateOutput(w http.ResponseWriter, r *http.Request) {
	var i Output
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE outputs SET name=?, instruction=?, command_tpl=? WHERE id=?", i.Name, i.Instruction, i.CommandTpl, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteOutput(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM outputs WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetAgents(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, provider_id, model, sources, skills, outputs, mcps, COALESCE(projects, '[]'), system_prompt, token_usage, COALESCE(input_methods, '[\"Text\"]'), COALESCE(output_methods, '[\"Text\"]'), COALESCE(user_prompt_prefix, ''), COALESCE(temperature, 0.7), COALESCE(max_tokens, 4096), COALESCE(top_p, 1.0), COALESCE(privacy_enabled, 0), COALESCE(can_create_skills, 0) FROM agents")
	if err != nil {
		fmt.Println("[DB Warning]:", err)
		json.NewEncoder(w).Encode([]Agent{})
		return
	}
	defer rows.Close()
	var list []Agent
	for rows.Next() {
		var i Agent
		rows.Scan(&i.ID, &i.Name, &i.ProviderID, &i.Model, &i.Sources, &i.Skills, &i.Outputs, &i.MCPs, &i.Projects, &i.SystemPrompt, &i.TokenUsage, &i.InputMethods, &i.OutputMethods, &i.UserPromptPrefix, &i.Temperature, &i.MaxTokens, &i.TopP, &i.PrivacyEnabled, &i.CanCreateSkills)
		list = append(list, i)
	}
	if list == nil {
		list = []Agent{}
	}
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateAgent(w http.ResponseWriter, r *http.Request) {
	var i Agent
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("INSERT INTO agents (id, name, provider_id, model, sources, skills, outputs, mcps, projects, system_prompt, input_methods, output_methods, user_prompt_prefix, temperature, max_tokens, top_p, privacy_enabled, can_create_skills) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", i.ID, i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs, i.MCPs, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods, i.UserPromptPrefix, i.Temperature, i.MaxTokens, i.TopP, i.PrivacyEnabled, i.CanCreateSkills)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateAgent(w http.ResponseWriter, r *http.Request) {
	var i Agent
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE agents SET name=?, provider_id=?, model=?, sources=?, skills=?, outputs=?, mcps=?, projects=?, system_prompt=?, input_methods=?, output_methods=?, user_prompt_prefix=?, temperature=?, max_tokens=?, top_p=?, privacy_enabled=?, can_create_skills=? WHERE id=?", i.Name, i.ProviderID, i.Model, i.Sources, i.Skills, i.Outputs, i.MCPs, i.Projects, i.SystemPrompt, i.InputMethods, i.OutputMethods, i.UserPromptPrefix, i.Temperature, i.MaxTokens, i.TopP, i.PrivacyEnabled, i.CanCreateSkills, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteAgent(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM agents WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetProjects(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, flow_json FROM projects")
	if err != nil {
		json.NewEncoder(w).Encode([]Project{})
		return
	}
	defer rows.Close()
	var list []Project
	for rows.Next() {
		var i Project
		rows.Scan(&i.ID, &i.Name, &i.Description, &i.FlowJSON)
		list = append(list, i)
	}
	if list == nil {
		list = []Project{}
	}
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateProject(w http.ResponseWriter, r *http.Request) {
	var i Project
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("INSERT INTO projects (id, name, description, flow_json) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.FlowJSON)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateProject(w http.ResponseWriter, r *http.Request) {
	var i Project
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE projects SET name=?, description=?, flow_json=? WHERE id=?", i.Name, i.Description, i.FlowJSON, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteProject(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM projects WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetMCP(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, url, command FROM mcp_servers")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreateMCP(w http.ResponseWriter, r *http.Request) {
	var i MCPServer
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("INSERT INTO mcp_servers (id, name, url, command) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.URL, i.Command)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeleteMCP(w http.ResponseWriter, r *http.Request) {
	a.ConfigDB.Exec("DELETE FROM mcp_servers WHERE id=?", r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) logAnalytics(entityType, entityID string, tokens int, success bool) {
	go func() {
		a.ConfigDB.Exec("INSERT INTO analytics_logs (entity_type, entity_id, tokens, is_success) VALUES (?, ?, ?, ?)", entityType, entityID, tokens, success)
	}()
}

// runProjectPipeline resolves node mappings in visual workflows and handles runtime execution.
func (a *App) runProjectPipeline(projectID string, agentID string, input string) string {
	var flowJson string
	a.ConfigDB.QueryRow("SELECT flow_json FROM projects WHERE id=?", projectID).Scan(&flowJson)
	var flow struct {
		Nodes []map[string]interface{} `json:"nodes"`
		Edges []map[string]interface{} `json:"edges"`
	}
	json.Unmarshal([]byte(flowJson), &flow)

	var currentNodeID string
	for _, n := range flow.Nodes {
		if n["type"] == "Start" {
			currentNodeID = n["id"].(string)
			break
		}
	}

	lastResult := input

	for loop := 0; loop < 20; loop++ {
		var nextNodeID string
		for _, e := range flow.Edges {
			if e["from"] == currentNodeID {
				nextNodeID = e["to"].(string)
				break
			}
		}
		if nextNodeID == "" {
			break
		}

		var targetNode map[string]interface{}
		for _, n := range flow.Nodes {
			if n["id"] == nextNodeID {
				targetNode = n
				break
			}
		}

		nodeType := targetNode["type"].(string)
		if nodeType == "End" {
			break
		}

		switch nodeType {
		case "Skill":
			sID, _ := targetNode["skillId"].(string)
			lastResult = fmt.Sprintf("[Skill %s Processed Data]\n%s", sID, lastResult)
		case "Agent":
			aID, _ := targetNode["agentId"].(string)

			customSys, _ := targetNode["sysPrompt"].(string)
			customPrefix, _ := targetNode["userPrefix"].(string)

			var defaultSys, defaultPrefix string
			a.ConfigDB.QueryRow("SELECT system_prompt, user_prompt_prefix FROM agents WHERE id=?", aID).Scan(&defaultSys, &defaultPrefix)

			if customSys == "" {
				customSys = defaultSys
			}
			if customPrefix == "" {
				customPrefix = defaultPrefix
			}

			lastResult = fmt.Sprintf("[Agent Task Completed]\nUsed System: %s...\nInput Processed: %s", customSys, lastResult)
		case "Code":
			code, _ := targetNode["code"].(string)

			tmp := filepath.Join(os.TempDir(), fmt.Sprintf("proj_node_%d.go", time.Now().UnixNano()))
			os.WriteFile(tmp, []byte(code), 0644)
			execCmd := exec.Command("go", "run", tmp)
			out, err := execCmd.CombinedOutput()
			os.Remove(tmp)

			if err != nil {
				lastResult = fmt.Sprintf("Code Error: %s", err.Error())
			} else {
				lastResult = strings.TrimSpace(string(out))
			}
		}
		currentNodeID = nextNodeID
	}
	return lastResult
}

func (a *App) handleUpdateProvider(w http.ResponseWriter, r *http.Request) {
	var i Provider
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE providers SET name=?, type=?, endpoint=?, api_key=?, extra_config=? WHERE id=?", i.Name, i.Type, i.Endpoint, i.APIKey, i.ExtraConfig, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSource(w http.ResponseWriter, r *http.Request) {
	var i Source
	json.NewDecoder(r.Body).Decode(&i)
	if i.Type != "Local File" && i.APIKey == "" {
		i.APIKey = fmt.Sprintf("sk_src_%d", time.Now().UnixNano())
	}
	a.ConfigDB.Exec("UPDATE sources SET name=?, type=?, api_key=?, file_path=? WHERE id=?", i.Name, i.Type, i.APIKey, i.FilePath, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateSkill(w http.ResponseWriter, r *http.Request) {
	var i Skill
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE skills SET name=?, type=?, instruction=?, content=?, api_method=?, api_url=?, api_headers=?, api_body=? WHERE id=?", i.Name, i.Type, i.Instruction, i.Content, i.APIMethod, i.APIURL, i.APIHeaders, i.APIBody, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdateRAG(w http.ResponseWriter, r *http.Request) {
	var i RAG
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE rags SET name=?, description=? WHERE id=?", i.Name, i.Description, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleGetPreferences(w http.ResponseWriter, r *http.Request) {
	rows, err := a.ConfigDB.Query("SELECT id, name, description, table_name FROM preferences")
	if err != nil {
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
	json.NewEncoder(w).Encode(list)
}

func (a *App) handleCreatePreference(w http.ResponseWriter, r *http.Request) {
	var i Preference
	json.NewDecoder(r.Body).Decode(&i)
	i.TableName = fmt.Sprintf("pref_tbl_%d", time.Now().UnixNano())
	a.ConfigDB.Exec("INSERT INTO preferences (id, name, description, table_name) VALUES (?, ?, ?, ?)", i.ID, i.Name, i.Description, i.TableName)
	a.MemoryDB.Exec(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, record_name TEXT UNIQUE, data TEXT)", i.TableName))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleUpdatePreference(w http.ResponseWriter, r *http.Request) {
	var i Preference
	json.NewDecoder(r.Body).Decode(&i)
	a.ConfigDB.Exec("UPDATE preferences SET name=?, description=? WHERE id=?", i.Name, i.Description, r.PathValue("id"))
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (a *App) handleDeletePreference(w http.ResponseWriter, r *http.Request) {
	var t string
	a.ConfigDB.QueryRow("SELECT table_name FROM preferences WHERE id=?", r.PathValue("id")).Scan(&t)
	a.MemoryDB.Exec("DROP TABLE IF EXISTS " + t)
	a.ConfigDB.Exec("DELETE FROM preferences WHERE id=?", r.PathValue("id"))
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
	json.NewEncoder(w).Encode(res)
}

func (a *App) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	var data map[string]string
	json.NewDecoder(r.Body).Decode(&data)
	for k, v := range data {
		a.ConfigDB.Exec("UPDATE settings SET value=? WHERE key=?", v, k)
	}
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
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
	case ".txt", ".md", ".csv", ".go", ".py", ".html", ".json", ".yaml":
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
