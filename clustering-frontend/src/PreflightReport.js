import React from 'react';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const LEVEL_TEXT = {
  recommended: 'Recommended',
  may_be_slow: 'May be slow',
  not_recommended: 'Not recommended on the hosted server',
};

const CHECK_ICON = { ok: '✓', info: 'ℹ', warning: '⚠', error: '✖' };

const formatLimit = (value) => (value >= 1000 ? `${(value / 1000).toLocaleString()}k` : String(value));

function LevelBadge({ level }) {
  return <span className={`preflight-badge preflight-${level}`}>{LEVEL_TEXT[level] || level}</span>;
}

function PreflightReport({ report, awaitingConfirmation, onRunAnyway, onCancel, onClose, loading }) {
  if (!report) return null;

  const { dataset, compatibility, checks } = report;
  const flaggedChecks = checks.filter(check => check.level !== 'ok');

  return (
    <section className={`preflight preflight-border-${report.has_data_errors ? 'not_recommended' : compatibility.overall}`}>
      <div className="preflight-header">
        <h2>Preflight Check<HelpTip text={HELP.preflight} label="preflight check" /></h2>
        <LevelBadge level={compatibility.overall} />
        {onClose && (
          <button type="button" className="details-toggle preflight-close" onClick={onClose}>Close ✕</button>
        )}
      </div>
      <p className="compare-note">
        {dataset.filename}: {dataset.n_rows_used} of {dataset.n_rows} rows used × {dataset.n_features} features;
        clustering runs on {report.effective_dimensions} dimension{report.effective_dimensions === 1 ? '' : 's'}.
      </p>

      {awaitingConfirmation && (
        <div className="preflight-confirm">
          <strong>
            {report.has_data_errors
              ? 'The data has problems that will likely make this run fail.'
              : compatibility.overall === 'not_recommended'
                ? 'This algorithm–dataset combination is not recommended on the hosted server. It may time out or run out of memory.'
                : 'This configuration may be slow.'}
          </strong>
          <div className="preflight-confirm-actions">
            <button type="button" className="preflight-run-anyway" onClick={onRunAnyway} disabled={loading}>Run anyway</button>
            <button type="button" onClick={onCancel} disabled={loading}>Cancel and change settings</button>
          </div>
        </div>
      )}

      {flaggedChecks.length > 0 && !awaitingConfirmation && (
        <ul className="preflight-checks">
          {flaggedChecks.map(check => (
            <li key={check.check} className={`preflight-check-${check.level}`}>
              <span className="preflight-check-icon">{CHECK_ICON[check.level]}</span>
              <strong>{check.check}:</strong> {check.value}
              {check.message && <span className="preflight-check-message"> — {check.message}</span>}
            </li>
          ))}
        </ul>
      )}

      {/* Remount on confirmation so the details open automatically when a decision is needed. */}
      <details key={awaitingConfirmation ? 'open' : 'closed'} open={awaitingConfirmation} className="preflight-details">
      <summary>Show all checks and capacity guidance</summary>
      <table className="compare-table preflight-table">
        <thead>
          <tr><th>Step</th><th>Method</th><th>Indicator</th><th>Details</th></tr>
        </thead>
        <tbody>
          {compatibility.items.map(item => (
            <tr key={item.component}>
              <td>{item.component}</td>
              <td>{item.name}</td>
              <td><LevelBadge level={item.level} /></td>
              <td>{item.message}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Data checks</h3>
      {flaggedChecks.length === 0 && <p className="compare-note">No data problems found.</p>}
      <ul className="preflight-checks">
        {checks.map(check => (
          <li key={check.check} className={`preflight-check-${check.level}`}>
            <span className="preflight-check-icon">{CHECK_ICON[check.level]}</span>
            <strong>{check.check}:</strong> {check.value}
            {check.message && <span className="preflight-check-message"> — {check.message}</span>}
          </li>
        ))}
      </ul>

      <details className="preflight-guidance">
        <summary>Capacity guidance for all algorithms with this dataset ({dataset.n_rows_used} rows)</summary>
        <p className="compare-note">
          There is no single maximum dataset size: algorithms differ strongly in how they scale. Limits below are rough guides
          for the hosted server (default parameters).
        </p>
        <table className="compare-table">
          <thead>
            <tr><th>Algorithm</th><th>For this dataset</th><th>Comfortable up to</th><th>Workable up to</th><th>Why</th></tr>
          </thead>
          <tbody>
            {report.algorithm_guidance.map(guide => (
              <tr key={guide.algorithm}>
                <td>{guide.algorithm}</td>
                <td><LevelBadge level={guide.level} /></td>
                <td>{guide.comfortable_up_to ? `${formatLimit(guide.comfortable_up_to)} rows` : '–'}</td>
                <td>{guide.workable_up_to ? `${formatLimit(guide.workable_up_to)} rows` : '–'}</td>
                <td>{guide.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      </details>
    </section>
  );
}

export default PreflightReport;
