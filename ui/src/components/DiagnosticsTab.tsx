import React, { useEffect, useState } from 'react';
import './DiagnosticsTab.css';

interface SystemInfo {
  version: string;
  build_info: string;
  platform: string;
  runtime_info: {
    uptime_seconds: number;
    memory_allocated_mb: number;
    gc_cycles: number;
    thread_count: number;
  };
  configuration: {
    storage_engines: string[];
    cache_size_mb: number;
    max_connections: number;
    query_timeout_ms: number;
  };
}

interface HealthCheck {
  component: string;
  status: 'Healthy' | 'Warning' | 'Critical' | 'Unknown';
  message: string;
  last_check: string;
  response_time_ms: number;
  details?: string;
}

interface DiagnosticTest {
  name: string;
  status: 'Passed' | 'Failed' | 'Running' | 'Skipped';
  duration_ms: number;
  details: string;
}

const DiagnosticsTab: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([]);
  const [diagnosticTests, setDiagnosticTests] = useState<DiagnosticTest[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningDiagnostics, setRunningDiagnostics] = useState<boolean>(false);

  useEffect(() => {
    const fetchDiagnosticData = async () => {
      setLoading(true);
      // This would be replaced with actual API calls
      setTimeout(() => {
        setSystemInfo({
          version: '1.0.4',
          build_info: 'proximadb-1.0.4-release-ba45e8ac',
          platform: 'Linux x86_64',
          runtime_info: {
            uptime_seconds: 1320540, // 15 days, 7 hours, 23 minutes
            memory_allocated_mb: 2048,
            gc_cycles: 15847,
            thread_count: 64,
          },
          configuration: {
            storage_engines: ['SST', 'NOVA', 'SWIFT', 'RAPTOR', 'PRISM', 'HELIX'],
            cache_size_mb: 2048,
            max_connections: 1000,
            query_timeout_ms: 30000,
          },
        });

        setHealthChecks([
          {
            component: 'Database Engine',
            status: 'Healthy',
            message: 'All storage engines operational',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 12,
          },
          {
            component: 'Cache System',
            status: 'Healthy',
            message: 'Unified cache orchestrator active',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 8,
          },
          {
            component: 'Query Engine',
            status: 'Healthy',
            message: 'Query processing normal',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 15,
          },
          {
            component: 'Index System',
            status: 'Healthy',
            message: 'AXIS engine operational',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 18,
          },
          {
            component: 'Network Services',
            status: 'Warning',
            message: 'High connection count detected',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 25,
            details: '1247 active connections (threshold: 1000)',
          },
          {
            component: 'Security Services',
            status: 'Healthy',
            message: 'All security features active',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 11,
          },
          {
            component: 'Backup System',
            status: 'Healthy',
            message: 'Continuous backup active',
            last_check: '2025-09-12T14:59:45Z',
            response_time_ms: 22,
          },
        ]);

        setDiagnosticTests([
          {
            name: 'Storage Engine Connectivity',
            status: 'Passed',
            duration_ms: 145,
            details: 'All 7 storage engines responding correctly',
          },
          {
            name: 'Query Performance Test',
            status: 'Passed',
            duration_ms: 890,
            details: 'Average query latency: 23.4ms (target: <100ms)',
          },
          {
            name: 'Cache Coherence Test',
            status: 'Passed',
            duration_ms: 320,
            details: 'Cache hit rate: 94.2% (target: >80%)',
          },
          {
            name: 'Index Integrity Check',
            status: 'Passed',
            duration_ms: 2150,
            details: 'All indexes verified and consistent',
          },
          {
            name: 'Network Latency Test',
            status: 'Passed',
            duration_ms: 180,
            details: 'Network response time: 12ms (target: <50ms)',
          },
          {
            name: 'Security Validation',
            status: 'Passed',
            duration_ms: 95,
            details: 'All security policies enforced correctly',
          },
        ]);

        setLoading(false);
      }, 1000);
    };

    fetchDiagnosticData();
  }, []);

  const runDiagnostics = async () => {
    setRunningDiagnostics(true);
    // Simulate running diagnostics
    setTimeout(() => {
      setRunningDiagnostics(false);
      // Refresh the diagnostic tests with new timestamps
      setDiagnosticTests(prev => prev.map(test => ({
        ...test,
        status: Math.random() > 0.1 ? 'Passed' : 'Failed',
        duration_ms: Math.floor(Math.random() * 3000) + 100,
      })));
    }, 5000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Healthy':
      case 'Passed':
        return 'healthy';
      case 'Warning':
        return 'warning';
      case 'Critical':
      case 'Failed':
        return 'critical';
      case 'Running':
        return 'running';
      case 'Unknown':
      case 'Skipped':
        return 'unknown';
      default:
        return 'unknown';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const checkTime = new Date(timestamp);
    const diffMs = now.getTime() - checkTime.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    
    if (diffSecs < 60) return `${diffSecs}s ago`;
    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    return `${diffHours}h ago`;
  };

  if (loading) {
    return <div className="diagnostics-loading">Loading diagnostics...</div>;
  }

  return (
    <div className="diagnostics-container">
      <h2>System Diagnostics & Health</h2>
      
      {systemInfo && (
        <div className="system-info-section">
          <h3>System Information</h3>
          <div className="info-grid">
            <div className="info-card">
              <h4>Version Information</h4>
              <div className="info-items">
                <div className="info-item">
                  <span className="label">Version:</span>
                  <span className="value">{systemInfo.version}</span>
                </div>
                <div className="info-item">
                  <span className="label">Build:</span>
                  <span className="value">{systemInfo.build_info}</span>
                </div>
                <div className="info-item">
                  <span className="label">Platform:</span>
                  <span className="value">{systemInfo.platform}</span>
                </div>
              </div>
            </div>

            <div className="info-card">
              <h4>Runtime Information</h4>
              <div className="info-items">
                <div className="info-item">
                  <span className="label">Uptime:</span>
                  <span className="value">{formatUptime(systemInfo.runtime_info.uptime_seconds)}</span>
                </div>
                <div className="info-item">
                  <span className="label">Memory:</span>
                  <span className="value">{systemInfo.runtime_info.memory_allocated_mb} MB</span>
                </div>
                <div className="info-item">
                  <span className="label">GC Cycles:</span>
                  <span className="value">{systemInfo.runtime_info.gc_cycles.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span className="label">Threads:</span>
                  <span className="value">{systemInfo.runtime_info.thread_count}</span>
                </div>
              </div>
            </div>

            <div className="info-card">
              <h4>Configuration</h4>
              <div className="info-items">
                <div className="info-item">
                  <span className="label">Storage Engines:</span>
                  <span className="value">{systemInfo.configuration.storage_engines.join(', ')}</span>
                </div>
                <div className="info-item">
                  <span className="label">Cache Size:</span>
                  <span className="value">{systemInfo.configuration.cache_size_mb} MB</span>
                </div>
                <div className="info-item">
                  <span className="label">Max Connections:</span>
                  <span className="value">{systemInfo.configuration.max_connections}</span>
                </div>
                <div className="info-item">
                  <span className="label">Query Timeout:</span>
                  <span className="value">{systemInfo.configuration.query_timeout_ms} ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="health-checks-section">
        <h3>Health Checks</h3>
        <div className="health-checks-grid">
          {healthChecks.map((check, index) => (
            <div key={index} className={`health-check-card ${getStatusColor(check.status)}`}>
              <div className="check-header">
                <h4>{check.component}</h4>
                <span className={`status-badge ${getStatusColor(check.status)}`}>
                  {check.status}
                </span>
              </div>
              <div className="check-message">{check.message}</div>
              {check.details && (
                <div className="check-details">{check.details}</div>
              )}
              <div className="check-meta">
                <span>Response: {check.response_time_ms}ms</span>
                <span>Checked: {formatTimeAgo(check.last_check)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="diagnostic-tests-section">
        <div className="tests-header">
          <h3>Diagnostic Tests</h3>
          <button 
            className="run-diagnostics-btn"
            onClick={runDiagnostics}
            disabled={runningDiagnostics}
          >
            {runningDiagnostics ? '🔄 Running...' : '▶️ Run Diagnostics'}
          </button>
        </div>
        
        <div className="diagnostic-tests-list">
          {diagnosticTests.map((test, index) => (
            <div key={index} className={`test-item ${getStatusColor(test.status)}`}>
              <div className="test-info">
                <div className="test-name">{test.name}</div>
                <div className="test-details">{test.details}</div>
              </div>
              <div className="test-results">
                <span className={`test-status ${getStatusColor(test.status)}`}>
                  {test.status}
                </span>
                <span className="test-duration">{test.duration_ms}ms</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="troubleshooting-section">
        <h3>Troubleshooting Tools</h3>
        <div className="tools-grid">
          <button className="tool-btn">🔍 Query Analyzer</button>
          <button className="tool-btn">📊 Performance Profiler</button>
          <button className="tool-btn">🧩 Memory Debugger</button>
          <button className="tool-btn">📋 Log Analyzer</button>
          <button className="tool-btn">🔧 Config Validator</button>
          <button className="tool-btn">📈 Metrics Exporter</button>
        </div>
      </div>

      <div className="support-section">
        <h3>Support Information</h3>
        <div className="support-info">
          <p>🆔 <strong>Instance ID:</strong> proximadb-prod-us-east-1-a7d3f2</p>
          <p>📧 <strong>Support:</strong> support@proximadb.com</p>
          <p>📚 <strong>Documentation:</strong> https://docs.proximadb.com</p>
          <p>🐛 <strong>Issues:</strong> https://github.com/proximadb/proximadb/issues</p>
        </div>
      </div>
    </div>
  );
};

export default DiagnosticsTab;