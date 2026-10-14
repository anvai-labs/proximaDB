import React, { useState, useCallback, useRef } from 'react';
import Editor, { loader, OnMount } from '@monaco-editor/react';
import './SqlQueryTab.css';

// TD-UICDN-1: resolve the Monaco runtime from our own origin (`public/vs`,
// copied from node_modules by scripts/copy-monaco-vs.js) instead of the
// jsdelivr CDN — the admin UI must have zero third-party script origins.
// Module scope so it is configured before the first <Editor> mounts; the call
// is idempotent.
loader.config({ paths: { vs: '/vs' } });

interface QueryResult {
  columns: string[];
  rows: Record<string, unknown>[];
  executionTimeMs: number;
  rowCount: number;
}

interface QueryHistoryItem {
  id: string;
  query: string;
  timestamp: Date;
  success: boolean;
  executionTimeMs?: number;
  rowCount?: number;
}

interface SavedQuery {
  id: string;
  name: string;
  query: string;
  description?: string;
}

// ProximaDB SQL keywords for syntax highlighting
const PROXIMADB_KEYWORDS = [
  'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN',
  'ORDER', 'BY', 'ASC', 'DESC', 'LIMIT', 'OFFSET', 'GROUP', 'HAVING',
  'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'ON', 'AS',
  'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE',
  'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX', 'COLLECTION', 'GRAPH',
  'VECTOR_SIMILAR', 'VECTOR_DISTANCE', 'GRAPH_TRAVERSE', 'GRAPH_NEIGHBORS',
  'EXISTS', 'NOT EXISTS', 'IS NULL', 'IS NOT NULL',
  'SIMILAR', 'FOLLOW', 'TOP_K', 'WITH_FILTER'
];

// Sample queries for users to try
const SAMPLE_QUERIES: SavedQuery[] = [
  {
    id: '1',
    name: 'List All Collections',
    query: 'SELECT * FROM collections LIMIT 10;',
    description: 'Show all vector collections'
  },
  {
    id: '2',
    name: 'Vector Similarity Search',
    query: `SELECT id, metadata, VECTOR_DISTANCE(embedding, [0.1, 0.2, 0.3]) as distance
FROM my_collection
WHERE VECTOR_SIMILAR(embedding, [0.1, 0.2, 0.3], 0.8)
ORDER BY distance
LIMIT 10;`,
    description: 'Find similar vectors with distance calculation'
  },
  {
    id: '3',
    name: 'Graph Traversal',
    query: `SELECT n.id, n.properties, e.type, e.weight
FROM GRAPH knowledge
WHERE GRAPH_TRAVERSE('entity_alice', 'KNOWS', 2)
ORDER BY e.weight DESC
LIMIT 20;`,
    description: 'Traverse graph relationships'
  },
  {
    id: '4',
    name: 'Hybrid Query',
    query: `SELECT d.id, d.$.name, g.relationship
FROM documents.products d
JOIN GRAPH knowledge ON d.id = knowledge.start_node
WHERE VECTOR_SIMILAR(d.$.embedding, ?, 0.7)
  AND GRAPH_NEIGHBORS(knowledge, d.id, 'CATEGORY');`,
    description: 'Combined vector search with graph traversal'
  },
  {
    id: '5',
    name: 'Observability Logs',
    query: `SELECT timestamp, severity, service, message
FROM logs
WHERE timestamp > NOW() - INTERVAL 1 HOUR
  AND severity >= 'ERROR'
ORDER BY timestamp DESC
LIMIT 100;`,
    description: 'Query recent error logs'
  }
];

const SqlQueryTab: React.FC = () => {
  const [query, setQuery] = useState<string>('-- Write your SQL query here\nSELECT * FROM collections LIMIT 10;');
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const [showSamples, setShowSamples] = useState<boolean>(true);
  const editorRef = useRef<unknown>(null);

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;

    // Register ProximaDB SQL language features
    monaco.languages.registerCompletionItemProvider('sql', {
      provideCompletionItems: (model, position) => {
        const word = model.getWordUntilPosition(position);
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        };

        const suggestions = PROXIMADB_KEYWORDS.map((keyword) => ({
          label: keyword,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: keyword,
          range: range,
        }));

        return { suggestions };
      },
    });

    // Custom theme for ProximaDB
    monaco.editor.defineTheme('proximadb-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'keyword', foreground: '569cd6', fontStyle: 'bold' },
        { token: 'string', foreground: 'ce9178' },
        { token: 'number', foreground: 'b5cea8' },
        { token: 'comment', foreground: '6a9955', fontStyle: 'italic' },
      ],
      colors: {
        'editor.background': '#1e1e2e',
        'editor.foreground': '#cdd6f4',
        'editorLineNumber.foreground': '#6c7086',
        'editor.selectionBackground': '#44475a',
      },
    });

    monaco.editor.setTheme('proximadb-dark');
  };

  const executeQuery = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const startTime = Date.now();
    const historyItem: QueryHistoryItem = {
      id: Date.now().toString(),
      query: query.trim(),
      timestamp: new Date(),
      success: false,
    };

    try {
      // Call ProximaDB SQL endpoint
      const response = await fetch('/api/v1/sql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      const executionTimeMs = Date.now() - startTime;

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}`);
      }

      const data = await response.json();

      const queryResult: QueryResult = {
        columns: data.columns || Object.keys(data.rows?.[0] || {}),
        rows: data.rows || [],
        executionTimeMs,
        rowCount: data.rows?.length || 0,
      };

      setResult(queryResult);
      historyItem.success = true;
      historyItem.executionTimeMs = executionTimeMs;
      historyItem.rowCount = queryResult.rowCount;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      historyItem.success = false;
    } finally {
      setLoading(false);
      setHistory((prev) => [historyItem, ...prev.slice(0, 49)]); // Keep last 50
    }
  }, [query]);

  const loadSampleQuery = (sample: SavedQuery) => {
    setQuery(sample.query);
    setShowSamples(false);
  };

  const loadHistoryQuery = (item: QueryHistoryItem) => {
    setQuery(item.query);
    setShowHistory(false);
  };

  const formatQuery = useCallback(() => {
    // Basic SQL formatting
    let formatted = query
      .replace(/\s+/g, ' ')
      .replace(/,\s*/g, ',\n  ')
      .replace(/\s+(FROM|WHERE|AND|OR|ORDER BY|GROUP BY|HAVING|LIMIT|JOIN|ON)\s+/gi, '\n$1 ')
      .replace(/\s+/g, ' ')
      .trim();

    // Add proper indentation
    const lines = formatted.split('\n');
    formatted = lines.map((line, i) => {
      if (i === 0) return line;
      if (/^(FROM|WHERE|ORDER BY|GROUP BY|HAVING|LIMIT)/i.test(line.trim())) {
        return line.trim();
      }
      if (/^(AND|OR|JOIN|ON)/i.test(line.trim())) {
        return '  ' + line.trim();
      }
      return '  ' + line.trim();
    }).join('\n');

    setQuery(formatted);
  }, [query]);

  return (
    <div className="sql-query-container">
      <div className="sql-query-header">
        <h2>SQL Query Editor</h2>
        <div className="query-actions">
          <button
            className="action-btn samples-btn"
            onClick={() => setShowSamples(!showSamples)}
          >
            {showSamples ? 'Hide' : 'Show'} Samples
          </button>
          <button
            className="action-btn history-btn"
            onClick={() => setShowHistory(!showHistory)}
          >
            History ({history.length})
          </button>
        </div>
      </div>

      <div className="sql-query-content">
        {/* Sample Queries Panel */}
        {showSamples && (
          <div className="samples-panel">
            <h3>Sample Queries</h3>
            <div className="samples-list">
              {SAMPLE_QUERIES.map((sample) => (
                <div
                  key={sample.id}
                  className="sample-item"
                  onClick={() => loadSampleQuery(sample)}
                >
                  <div className="sample-name">{sample.name}</div>
                  <div className="sample-description">{sample.description}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Query History Panel */}
        {showHistory && (
          <div className="history-panel">
            <h3>Query History</h3>
            <div className="history-list">
              {history.length === 0 ? (
                <div className="history-empty">No query history yet</div>
              ) : (
                history.map((item) => (
                  <div
                    key={item.id}
                    className={`history-item ${item.success ? 'success' : 'error'}`}
                    onClick={() => loadHistoryQuery(item)}
                  >
                    <div className="history-query">
                      {item.query.substring(0, 80)}
                      {item.query.length > 80 ? '...' : ''}
                    </div>
                    <div className="history-meta">
                      <span className="history-time">
                        {item.timestamp.toLocaleTimeString()}
                      </span>
                      {item.executionTimeMs && (
                        <span className="history-duration">
                          {item.executionTimeMs}ms
                        </span>
                      )}
                      {item.rowCount !== undefined && (
                        <span className="history-rows">{item.rowCount} rows</span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Editor Section */}
        <div className="editor-section">
          <div className="editor-toolbar">
            <button
              className="toolbar-btn execute-btn"
              onClick={executeQuery}
              disabled={loading}
            >
              {loading ? 'Executing...' : 'Execute (Ctrl+Enter)'}
            </button>
            <button className="toolbar-btn format-btn" onClick={formatQuery}>
              Format Query
            </button>
            <button
              className="toolbar-btn clear-btn"
              onClick={() => setQuery('')}
            >
              Clear
            </button>
          </div>

          <div className="editor-wrapper">
            <Editor
              height="300px"
              defaultLanguage="sql"
              value={query}
              onChange={(value) => setQuery(value || '')}
              onMount={handleEditorDidMount}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
                wordWrap: 'on',
                suggestOnTriggerCharacters: true,
                quickSuggestions: true,
              }}
            />
          </div>
        </div>

        {/* Results Section */}
        <div className="results-section">
          {loading && (
            <div className="results-loading">
              <div className="loading-spinner"></div>
              <span>Executing query...</span>
            </div>
          )}

          {error && (
            <div className="results-error">
              <h4>Query Error</h4>
              <pre>{error}</pre>
            </div>
          )}

          {result && !loading && !error && (
            <div className="results-success">
              <div className="results-meta">
                <span className="result-count">{result.rowCount} rows</span>
                <span className="result-time">{result.executionTimeMs}ms</span>
              </div>

              {result.rowCount > 0 ? (
                <div className="results-table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        {result.columns.map((col, i) => (
                          <th key={i}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {result.columns.map((col, colIndex) => (
                            <td key={colIndex}>
                              {typeof row[col] === 'object'
                                ? JSON.stringify(row[col])
                                : String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="results-empty">Query executed successfully. No rows returned.</div>
              )}
            </div>
          )}

          {!loading && !error && !result && (
            <div className="results-placeholder">
              Execute a query to see results here
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SqlQueryTab;
