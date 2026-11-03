import React, { useEffect, useRef, useState } from 'react';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');

const formatValue = (value) => (value === null || value === undefined ? 'None' : String(value));

const formatParams = (params) => {
  const entries = Object.entries(params || {});
  return entries.length > 0
    ? entries.map(([name, value]) => `${name}=${formatValue(value)}`).join(', ')
    : '–';
};

const formatTime = (isoString) => new Date(isoString).toLocaleTimeString();

const formatNoise = (run) => {
  if (run.n_noise === null || run.n_noise === undefined) return '–';
  const total = run.dataset.n_observations;
  return total ? `${run.n_noise} (${((100 * run.n_noise) / total).toFixed(1)}%)` : String(run.n_noise);
};

function PreprocessingSummaryLine({ summary }) {
  const removedColumns = [
    ...summary.columns_removed_low_variance,
    ...summary.columns_removed_correlated.map(item => item.column),
  ];
  return (
    <>
      <br />
      Used {summary.rows_used} of {summary.rows_in} rows and {summary.columns_used} of {summary.columns_in} columns
      {summary.rows_removed_missing > 0 && `; ${summary.rows_removed_missing} rows removed for missing values`}
      {summary.rows_removed_zero > 0 && `; ${summary.rows_removed_zero} all-zero rows removed`}
      {summary.cells_imputed > 0 && `; ${summary.cells_imputed} missing cells imputed`}
      {removedColumns.length > 0 && `; columns removed: ${removedColumns.slice(0, 8).join(', ')}${removedColumns.length > 8 ? ' …' : ''}`}
    </>
  );
}

function RunDetails({ run }) {
  const { dataset, preprocessing, reduction, clustering, metrics } = run;

  return (
    <dl className="run-details">
      <dt>Dataset</dt>
      <dd>
        {dataset.filename}
        {dataset.n_observations !== null && ` (${dataset.n_observations} rows × ${dataset.n_features} features after preprocessing)`}
        <br />
        <span className="run-hash">SHA-256: {dataset.sha256}</span>
      </dd>

      <dt>Preprocessing</dt>
      <dd>
        Name column: {formatValue(preprocessing.name_column)}; data start column index: {preprocessing.intensity_start_index}
        {preprocessing.sheet_name && `; sheet: ${preprocessing.sheet_name}`}
        <br />
        Missing values {preprocessing.missing_values}; transform {preprocessing.transform}; all-zero rows {preprocessing.zero_rows};
        {' '}column filters: {preprocessing.filters || 'none'}; scaling: {preprocessing.scaling}
        {run.preprocessing_summary && <PreprocessingSummaryLine summary={run.preprocessing_summary} />}
      </dd>

      <dt>Dimensionality reduction</dt>
      <dd>{reduction.method}{reduction.method !== 'None' && ` — ${formatParams(reduction.params)}`}</dd>

      <dt>Clustering</dt>
      <dd>{clustering.algorithm} — {formatParams(clustering.params)}</dd>

      <dt>Random seed</dt>
      <dd>
        Reduction: {formatValue(run.random_seed.reduction)}; clustering: {formatValue(run.random_seed.clustering)}
      </dd>

      <dt>Validation metrics</dt>
      <dd>
        {metrics
          ? `Silhouette ${formatNumber(metrics.silhouette_score)}, Calinski–Harabasz ${formatNumber(metrics.calinski_harabasz_score)}, Davies–Bouldin ${formatNumber(metrics.davies_bouldin_score)}`
          : '–'}
      </dd>

      <dt>Runtime</dt>
      <dd>
        Total {formatNumber(run.runtime_seconds.total, 2)} s; clustering step {formatNumber(run.runtime_seconds.clustering, 2)} s
      </dd>

      <dt>Peak memory</dt>
      <dd>{formatNumber(run.peak_memory_mb, 1)} MB (peak resident memory of the analysis process)</dd>

      {run.limits && (
        <>
          <dt>Server limits</dt>
          <dd>
            Runs are stopped after {Math.round(run.limits.timeout_seconds / 60 * 10) / 10} min or above {Math.round(run.limits.memory_mb / 1024 * 10) / 10} GB memory
          </dd>
        </>
      )}

      {run.sampling?.sampled && (
        <>
          <dt>Interactive view</dt>
          <dd>
            {run.sampling.shown} of {run.sampling.total} rows shown in the plots ({run.sampling.method}); downloads contain all rows
          </dd>
        </>
      )}

      <dt>Warnings</dt>
      <dd>
        {run.warnings.length > 0 ? (
          <ul>
            {run.warnings.map(warning => <li key={warning}>{warning}</li>)}
          </ul>
        ) : 'None'}
      </dd>

      {run.error && (
        <>
          <dt>{run.status === 'cancelled' ? 'Cancelled' : 'Error'}</dt>
          <dd className="run-error">
            {run.error}
            {run.error_hint && <><br /><span className="run-error-hint">{run.error_hint}</span></>}
            {run.error_code && <><br /><span className="run-hash">Error code: {run.error_code}</span></>}
          </dd>
        </>
      )}
    </dl>
  );
}

// Runs created before the report link existed still have the same URL layout.
const reportLink = (run) => run.links.report || `${run.links.result}/report.html`;

// "Download ▾" button with the run's files.
const MENU_WIDTH = 190;
function DownloadMenu({ run, apiBase }) {
  const [position, setPosition] = useState(null);  // fixed position, so the scrolling table does not clip it
  const menu = useRef(null);
  const button = useRef(null);
  const open = position !== null;

  useEffect(() => {
    if (!open) return undefined;
    const close = () => setPosition(null);
    const closeOutside = (e) => { if (menu.current && !menu.current.contains(e.target)) close(); };
    const closeOnEscape = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('mousedown', closeOutside);
    document.addEventListener('keydown', closeOnEscape);
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('mousedown', closeOutside);
      document.removeEventListener('keydown', closeOnEscape);
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);

  const toggle = () => {
    if (open) {
      setPosition(null);
      return;
    }
    const rect = button.current.getBoundingClientRect();
    setPosition({ top: rect.bottom + 4, left: Math.max(8, rect.right - MENU_WIDTH) });
  };

  return (
    <div className="download-menu" ref={menu}>
      <button type="button" ref={button} onClick={toggle} aria-expanded={open} aria-haspopup="true">
        Download ▾
      </button>
      {open && (
        <div className="download-menu-list" role="menu" style={{ ...position, width: MENU_WIDTH }}>
          <a role="menuitem" href={`${apiBase}${run.links.download_zip}`} download onClick={() => setPosition(null)}>
            Download All (ZIP)
          </a>
          <a role="menuitem" href={`${apiBase}${run.links.labels_tsv}`} download onClick={() => setPosition(null)}>
            Cluster labels (TSV)
          </a>
        </div>
      )}
    </div>
  );
}

function RunHistory({ runs, activeRunId, apiBase, loading, onView, onLoadSettings }) {
  const [expandedRunId, setExpandedRunId] = useState(null);
  const [collapsed, setCollapsed] = useState(false);

  if (runs.length === 0) return null;

  const newestFirst = [...runs].sort((a, b) => b.number - a.number);

  return (
    <section className="run-history">
      <div className="section-header">
        <h2>Run History ({runs.length})<HelpTip text={HELP.runHistory} label="run history" /></h2>
        <button type="button" className="details-toggle" onClick={() => setCollapsed(value => !value)} aria-expanded={!collapsed}>
          {collapsed ? 'Show ▼' : 'Hide ▲'}
        </button>
      </div>
      {!collapsed && (<>
      <p className="run-history-note">
        Every run is saved for this browser session. The uploaded file is reused, so you can try new settings without
        uploading again. All runs are deleted when you close or reload the page.
      </p>
      <div className="run-history-table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Dataset</th>
              <th>Reduction</th>
              <th>Algorithm</th>
              <th>Clusters</th>
              <th>Noise</th>
              <th>Silhouette</th>
              <th>Runtime (s)</th>
              <th>Peak mem. (MB)</th>
              <th>Status</th>
              <th>Actions</th>
              <th>Report<HelpTip text={HELP.runReport} label="report" /></th>
              <th>Download</th>
            </tr>
          </thead>
          <tbody>
            {newestFirst.map(run => {
              const isActive = run.run_id === activeRunId;
              const isExpanded = run.run_id === expandedRunId;

              return (
                <React.Fragment key={run.run_id}>
                  <tr className={isActive ? 'active-run' : ''}>
                    <td>{run.number}</td>
                    <td>{formatTime(run.created_at)}</td>
                    <td title={`SHA-256: ${run.dataset.sha256}`}>{run.dataset.filename}</td>
                    <td>{run.reduction.method}</td>
                    <td>
                      {run.clustering.algorithm}
                      {run.sweep && (
                        <span className="sweep-tag" title={`Trial ${run.sweep.trial} of a parameter sweep: ${Object.entries(run.sweep.values).map(([k, v]) => `${k}=${v === null ? 'None' : v}`).join(', ')}`}>
                          sweep
                        </span>
                      )}
                    </td>
                    <td>{run.n_clusters ?? '–'}</td>
                    <td>{formatNoise(run)}</td>
                    <td>{formatNumber(run.metrics?.silhouette_score)}</td>
                    <td>{formatNumber(run.runtime_seconds.total, 2)}</td>
                    <td>{formatNumber(run.peak_memory_mb, 1)}</td>
                    <td>
                      <span
                        className={`run-status run-status-${run.status}`}
                        title={run.error ? `${run.error}${run.error_hint ? `\n${run.error_hint}` : ''}` : undefined}
                      >
                        {run.status}
                      </span>
                      {run.warnings.length > 0 && (
                        <span className="run-warning-badge" title={run.warnings.join('\n')}>
                          ⚠ {run.warnings.length}
                        </span>
                      )}
                    </td>
                    <td className="run-actions">
                      {run.status === 'running' && <span className="run-running-note">running…</span>}
                      {run.status === 'success' && (
                        <button type="button" onClick={() => onView(run)} disabled={loading || isActive}>
                          {isActive ? 'Viewing' : 'View'}
                        </button>
                      )}
                      {run.status !== 'success' && run.status !== 'running' && (
                        <button type="button" onClick={() => onLoadSettings(run)} disabled={loading}>
                          Load settings
                        </button>
                      )}
                      <button
                        type="button"
                        className="link-button"
                        onClick={() => setExpandedRunId(isExpanded ? null : run.run_id)}
                      >
                        {isExpanded ? 'Hide' : 'Details'}
                      </button>
                    </td>
                    <td>
                      {run.status === 'success' ? (
                        <a
                          className="report-button"
                          href={`${apiBase}${reportLink(run)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          title="Open the HTML report of this run in a new tab"
                        >
                          Report
                        </a>
                      ) : '–'}
                    </td>
                    <td>
                      {run.status === 'success' ? <DownloadMenu run={run} apiBase={apiBase} /> : '–'}
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className="run-details-row">
                      <td colSpan={14}>
                        <RunDetails run={run} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      </>)}
    </section>
  );
}

export default RunHistory;
