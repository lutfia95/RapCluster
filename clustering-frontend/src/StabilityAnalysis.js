import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { fetchJson, formatApiError, pollJob } from './api';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const PLOT_CONFIG = { responsive: true, displaylogo: false };
const PLOT_COLORS = { plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc' };
const VERDICT_COLORS = { stable: '#28a745', doubtful: '#e6a700', unstable: '#dc3545' };
const ROW_PREVIEW = 20;

const clusterLabel = (cluster) => (cluster === -1 ? 'Noise' : `Cluster ${cluster}`);
const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
const percent = (value) => (Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '–');

// Details per run, keyed by when the analysis finished, so a recomputation is fetched again.
const cache = {};

function StabilityAnalysis({ run, apiBase, onRunUpdated }) {
  const seeded = run.clustering?.params?.random_state !== undefined && run.clustering?.params?.random_state !== null;
  const [resamples, setResamples] = useState(10);
  const [seeds, setSeeds] = useState(seeded ? 5 : 0);
  const [details, setDetails] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [showAllRows, setShowAllRows] = useState(false);
  const cancelledRef = useRef(false);

  const hasDetails = Boolean(run.stability && run.stability.mean_row_stability !== undefined);
  const cacheKey = `${run.run_id}:${run.stability?.runtime_seconds ?? ''}`;

  useEffect(() => {
    setError(null);
    if (!hasDetails) {
      setDetails(null);
      return undefined;
    }
    if (cache[cacheKey]) {
      setDetails(cache[cacheKey]);
      return undefined;
    }
    let cancelled = false;
    fetchJson(`${apiBase}${run.links.stability}`)
      .then(result => {
        cache[cacheKey] = result;
        if (!cancelled) setDetails(result);
      })
      .catch(e => { if (!cancelled) setError(formatApiError(e)); });
    return () => { cancelled = true; };
  }, [apiBase, cacheKey, hasDetails, run.links.stability]);

  // Set on mount as well: React's development mode mounts, unmounts and mounts again, and a flag left
  // at true by that test unmount stopped the polling before it started.
  useEffect(() => {
    cancelledRef.current = false;
    return () => { cancelledRef.current = true; };
  }, []);

  // The inputs show the settings of the last analysis of this run, if any.
  useEffect(() => {
    setResamples(run.stability?.n_resamples ?? 10);
    setSeeds(seeded ? (run.stability?.n_seeds ?? 5) : 0);
  }, [run.run_id, run.stability, seeded]);

  const compute = async () => {
    setError(null);
    setProgress({ message: 'Starting…' });
    try {
      await fetchJson(`${apiBase}${run.links.stability}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resamples: Number(resamples), seeds: seeded ? Number(seeds) : 0 }),
      });
      const status = await pollJob(apiBase, run.links.status, {
        kind: 'stability',
        intervalMs: 1000,
        onProgress: (s) => setProgress({ message: s.message, elapsed: s.elapsed_seconds }),
        isCancelled: () => cancelledRef.current,
      });
      if (!status) return;
      if (status.run) onRunUpdated(status.run);
      if (status.state !== 'success') {
        setError(status.state === 'cancelled' ? 'The stability analysis was cancelled.'
          : formatApiError({ message: status.error, hint: status.hint }));
      }
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setProgress(null);
    }
  };

  const cancel = async () => {
    try {
      await fetchJson(`${apiBase}${run.links.cancel}?kind=stability`, { method: 'POST' });
    } catch (e) {
      setError(formatApiError(e));
    }
  };

  const solutionsText = (resampleCount, seedCount) => (
    `${resampleCount} random 80% subsamples${seedCount > 0 ? ` and ${seedCount} runs on all rows with other random seeds` : ''}`
  );

  const controls = (
    <div className="stability-controls">
      <label>
        Subsamples (80%)<HelpTip text={HELP.resamples} label="subsamples" />
        <input id="stability-resamples" type="number" min="3" max="30" value={resamples} disabled={Boolean(progress)}
          onChange={(e) => setResamples(e.target.value)} />
      </label>
      <label title={seeded ? '' : 'This algorithm has no random seed; it gives the same result on the same rows.'}>
        Other random seeds<HelpTip text={HELP.seeds} label="random seeds" />
        <input id="stability-seeds" type="number" min="0" max="10" value={seeded ? seeds : 0} disabled={!seeded || Boolean(progress)}
          onChange={(e) => setSeeds(e.target.value)} />
      </label>
      {progress ? (
        <>
          <span className="stability-progress">
            {progress.message}{Number.isFinite(progress.elapsed) && ` (${progress.elapsed.toFixed(0)} s)`}
          </span>
          <button type="button" className="cancel-button" onClick={cancel}>Cancel</button>
        </>
      ) : (
        <button type="button" className="stability-compute" onClick={compute}>
          {run.stability ? 'Recompute stability' : 'Compute stability'}
        </button>
      )}
      {!seeded && <span className="compare-note">({run.clustering.algorithm} has no random seed, so only subsamples are used.)</span>}
    </div>
  );

  if (!hasDetails || !details) {
    return (
      <div className="stability-analysis">
        <p className="compare-note">
          A high internal score does not mean that a clustering is reproducible. The stability analysis re-clusters
          subsamples of the data (and, for seeded algorithms, all rows with other random seeds) with the same settings and
          checks which clusters and which rows come back. It costs one extra clustering per solution.
        </p>
        {run.stability && !hasDetails && (
          <p className="compare-note">
            This run only has the earlier summary (mean ARI {formatNumber(run.stability.mean_ari)}); recompute to see
            cluster and row stability.
          </p>
        )}
        {controls}
        {error && <div className="error-message">{error}</div>}
        {hasDetails && !details && !error && <p>Loading stability details…</p>}
      </div>
    );
  }

  const { summary, settings, clusters, distributions, rows, thresholds } = details;
  const realClusters = clusters.filter(entry => entry.mean_jaccard !== undefined);
  const histogramCenters = rows.histogram.counts.map((_, i) => (rows.histogram.edges[i] + rows.histogram.edges[i + 1]) / 2);
  const subsampleAri = distributions.vs_reference.filter(item => item.kind === 'subsample').map(item => item.ari);
  const seedAri = distributions.vs_reference.filter(item => item.kind === 'seed').map(item => item.ari);
  const shownRows = showAllRows ? rows.least_stable : rows.least_stable.slice(0, ROW_PREVIEW);

  return (
    <div className="stability-analysis">
      <p className="stability-summary">
        Based on {solutionsText(settings.n_resamples, settings.n_seeds)}:{' '}
        <strong>{summary.n_stable_clusters}</strong> stable, <strong>{summary.n_doubtful_clusters}</strong> doubtful and{' '}
        <strong>{summary.n_unstable_clusters}</strong> unstable clusters; mean ARI with this run{' '}
        <strong>{formatNumber(summary.mean_ari)}</strong> (± {formatNumber(summary.std_ari)}); {percent(summary.share_stable_rows)} of
        the rows keep their cluster in at least {100 * thresholds.stable_row}% of the solutions.
        {summary.consensus_changed > 0 && ` ${summary.consensus_changed.toLocaleString()} rows are assigned to another cluster more often than to their own (consensus cluster in the labels file).`}
      </p>
      {controls}
      {error && <div className="error-message">{error}</div>}

      <h4>Cluster stability (Jaccard overlap with the best-matching cluster)<HelpTip text={HELP.jaccard} label="cluster stability" /></h4>
      <Plot
        data={[{
          type: 'bar',
          x: realClusters.map(entry => clusterLabel(entry.cluster)),
          y: realClusters.map(entry => entry.mean_jaccard),
          error_y: { type: 'data', array: realClusters.map(entry => entry.std_jaccard), visible: true, color: '#555' },
          marker: { color: realClusters.map(entry => VERDICT_COLORS[entry.verdict]) },
          customdata: realClusters.map(entry => [entry.verdict, entry.size, entry.min_jaccard]),
          hovertemplate: '%{x}: %{y:.3f} (%{customdata[0]})<br>size %{customdata[1]}, worst solution %{customdata[2]:.3f}<extra></extra>',
        }]}
        layout={{
          autosize: true, height: 340, margin: { l: 50, r: 10, t: 20, b: 80 },
          xaxis: { type: 'category' }, yaxis: { range: [0, 1.05], title: { text: 'Mean Jaccard' } },
          shapes: [thresholds.doubtful_jaccard, thresholds.stable_jaccard].map(y => ({
            type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y, line: { dash: 'dash', color: '#888', width: 1 },
          })),
          ...PLOT_COLORS,
        }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <p className="compare-note">
        Dashed lines: {thresholds.doubtful_jaccard} and {thresholds.stable_jaccard}. A mean Jaccard of at least{' '}
        {thresholds.stable_jaccard} is commonly read as a stable cluster; values between {thresholds.doubtful_jaccard} and{' '}
        {thresholds.stable_jaccard} indicate a pattern whose membership is doubtful, and lower values a cluster that tends to
        dissolve when the data change slightly (Hennig, 2007). Error bars: standard deviation over the solutions.
      </p>
      <div className="characterization-table-wrapper">
        <table className="compare-table stability-table">
          <thead>
            <tr><th>Cluster</th><th>Size</th><th>Mean Jaccard</th><th>Worst solution</th><th>Verdict</th><th>Mean row stability</th></tr>
          </thead>
          <tbody>
            {clusters.map(entry => (
              <tr key={entry.cluster}>
                <td>{clusterLabel(entry.cluster)}</td>
                <td>{entry.size.toLocaleString()}</td>
                <td>{entry.mean_jaccard !== undefined ? `${formatNumber(entry.mean_jaccard)} ± ${formatNumber(entry.std_jaccard)}` : '–'}</td>
                <td>{formatNumber(entry.min_jaccard)}</td>
                <td>{entry.verdict ? <span className={`verdict verdict-${entry.verdict}`}>{entry.verdict}</span> : 'not a cluster'}</td>
                <td>{formatNumber(entry.mean_row_stability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4>Agreement between solutions<HelpTip text={HELP.agreement} label="agreement" /></h4>
      <Plot
        data={[
          { type: 'box', y: subsampleAri, name: 'ARI vs. this run (subsamples)', boxpoints: 'all', jitter: 0.4, marker: { color: '#1f77b4' } },
          ...(seedAri.length > 0 ? [{ type: 'box', y: seedAri, name: 'ARI vs. this run (other seeds)', boxpoints: 'all', jitter: 0.4, marker: { color: '#9467bd' } }] : []),
          { type: 'box', y: distributions.pairwise_ari, name: 'Pairwise ARI', marker: { color: '#2ca02c' } },
          { type: 'box', y: distributions.pairwise_nmi, name: 'Pairwise NMI', marker: { color: '#ff7f0e' } },
        ]}
        layout={{
          autosize: true, height: 340, margin: { l: 50, r: 10, t: 20, b: 60 }, showlegend: false,
          yaxis: { title: { text: 'Agreement (1 = identical)' } }, ...PLOT_COLORS,
        }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <p className="compare-note">
        Pairwise values compare the repeated solutions with each other on the rows they share. A wide spread means the
        result depends strongly on the exact rows or on the random seed.
      </p>

      <h4>Row stability<HelpTip text={HELP.rowStability} label="row stability" /></h4>
      <Plot
        data={[{ type: 'bar', x: histogramCenters, y: rows.histogram.counts, marker: { color: '#6c757d' } }]}
        layout={{
          autosize: true, height: 280, margin: { l: 50, r: 10, t: 20, b: 50 }, bargap: 0.05,
          xaxis: { title: { text: 'Share of solutions in which the row keeps its cluster' }, range: [0, 1] },
          yaxis: { title: { text: 'Rows' } }, ...PLOT_COLORS,
        }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <p className="compare-note">
        Row stability is also a column in Cluster Details and in the labels file (with the consensus cluster). Noise rows
        count as stable when they stay noise.
      </p>
      <h5>Least stable rows</h5>
      <table className="compare-table">
        <thead>
          <tr><th>Name</th><th>Cluster</th><th>Row stability</th><th>In solutions</th><th>Consensus cluster<HelpTip text={HELP.consensus} label="consensus cluster" /></th><th>Agreement</th></tr>
        </thead>
        <tbody>
          {shownRows.map(row => (
            <tr key={`${row.name}-${row.cluster}`}>
              <td>{row.name}</td><td>{clusterLabel(row.cluster)}</td><td>{formatNumber(row.stability, 2)}</td>
              <td>{row.times_included}</td>
              <td>{row.consensus === -2 ? 'unmatched' : clusterLabel(row.consensus)}</td>
              <td>{percent(row.agreement)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.least_stable.length > ROW_PREVIEW && (
        <button type="button" className="link-button" onClick={() => setShowAllRows(show => !show)}>
          {showAllRows ? 'Show fewer' : `Show all ${rows.least_stable.length}`}
        </button>
      )}
      <p className="compare-note">{details.method} Computed in {formatNumber(details.runtime_seconds, 1)} s.</p>
    </div>
  );
}

export default StabilityAnalysis;
