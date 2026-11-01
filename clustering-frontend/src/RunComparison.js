import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { fetchJson, formatApiError } from './api';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const NOISE_COLOR = '#808080';
// Colors for Run B clusters that have no overlap partner in Run A (kept distinct from the tab20 palette).
const UNMATCHED_COLORS = ['#000000', '#8b4513', '#556b2f', '#483d8b', '#b8860b', '#008b8b', '#8b008b', '#2f4f4f'];
const MAX_CHANGED_ROWS = 200;
const SCATTERGL_THRESHOLD = 5000;
const PLOT_CONFIG = { responsive: true, displayModeBar: true, displaylogo: false };
const PLOT_COLORS = { plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc' };

const clusterLabel = (cluster) => (cluster === -1 ? 'Noise' : `C${cluster}`);
const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
const formatValue = (value) => (value === null || value === undefined ? 'None' : String(value));
const formatParams = (params) => {
  const entries = Object.entries(params || {});
  return entries.length > 0 ? entries.map(([name, value]) => `${name}=${formatValue(value)}`).join(', ') : '–';
};
// Real clusters in numeric order, noise last.
const sortClusters = (clusters) => [...clusters].sort((a, b) => (a === -1) - (b === -1) || a - b);

const median = (values) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
};

const runOptionLabel = (run) => (
  `#${run.number} — ${run.clustering.algorithm} (${run.reduction.method}), ${run.n_clusters} clusters`
);

function parameterRows(runA, runB) {
  const paramRows = (prefix, paramsA, paramsB) => (
    Array.from(new Set([...Object.keys(paramsA), ...Object.keys(paramsB)])).map(name => [
      `${prefix}: ${name}`,
      name in paramsA ? formatValue(paramsA[name]) : '–',
      name in paramsB ? formatValue(paramsB[name]) : '–',
    ])
  );

  return [
    ['Dataset', runA.dataset.filename, runB.dataset.filename],
    ['Dataset SHA-256', runA.dataset.sha256.slice(0, 12), runB.dataset.sha256.slice(0, 12)],
    ['Name column', formatValue(runA.preprocessing.name_column), formatValue(runB.preprocessing.name_column)],
    ['Data start column index', String(runA.preprocessing.intensity_start_index), String(runB.preprocessing.intensity_start_index)],
    ['Sheet', formatValue(runA.preprocessing.sheet_name), formatValue(runB.preprocessing.sheet_name)],
    ...paramRows('Preprocessing', runA.preprocessing.options || {}, runB.preprocessing.options || {}),
    ['Dimensionality reduction', runA.reduction.method, runB.reduction.method],
    ...paramRows('Reduction', runA.reduction.params, runB.reduction.params),
    ['Clustering algorithm', runA.clustering.algorithm, runB.clustering.algorithm],
    ...paramRows('Clustering', runA.clustering.params, runB.clustering.params),
  ].map(([label, a, b]) => ({ label, a, b, differs: a !== b }));
}

function RunPanel({ title, run, result, colorFor }) {
  const rows = result.cluster_data;

  const { clusters, rowsByCluster } = useMemo(() => {
    const grouped = new Map();
    rows.forEach(row => {
      if (!grouped.has(row.cluster)) grouped.set(row.cluster, []);
      grouped.get(row.cluster).push(row);
    });
    return { clusters: sortClusters(grouped.keys()), rowsByCluster: grouped };
  }, [rows]);

  const profileColumns = useMemo(() => {
    const profileLength = rows[0]?.profile_values?.length || 0;
    const names = result.intensity_columns || [];
    return names.length === profileLength
      ? names
      : Array.from({ length: profileLength }, (_, i) => `Dimension ${i + 1}`);
  }, [rows, result.intensity_columns]);

  // Sizes and medians come from the server's full-data summary; the points may be a sample.
  const fullSummary = result.cluster_profiles;
  const sizeOf = useMemo(() => {
    const sizes = new Map();
    if (fullSummary) fullSummary.clusters.forEach((cluster, index) => sizes.set(cluster, fullSummary.sizes[index]));
    return (cluster) => sizes.get(cluster) ?? rowsByCluster.get(cluster)?.length ?? 0;
  }, [fullSummary, rowsByCluster]);

  const profileMedians = useMemo(() => clusters.map(cluster => {
    const summaryIndex = fullSummary ? fullSummary.clusters.indexOf(cluster) : -1;
    if (summaryIndex >= 0) return fullSummary.median[summaryIndex];
    return profileColumns.map((_, columnIndex) => median(
      rowsByCluster.get(cluster)
        .map(row => row.profile_values[columnIndex])
        .filter(value => Number.isFinite(value))
    ));
  }), [clusters, rowsByCluster, profileColumns, fullSummary]);

  const total = result.sampling?.total ?? rows.length;
  const noiseCount = sizeOf(-1);

  const scatterTrace = {
    type: rows.length > SCATTERGL_THRESHOLD ? 'scattergl' : 'scatter',
    mode: 'markers',
    x: rows.map(row => row.x),
    y: rows.map(row => row.y),
    marker: { color: rows.map(row => colorFor(row.cluster)), size: 5, line: { color: 'rgba(0,0,0,0.3)', width: 0.3 } },
    text: rows.map(row => `${row.name}<br>Cluster: ${clusterLabel(row.cluster)}`),
    hoverinfo: 'text',
  };

  const sizeTrace = {
    type: 'bar',
    x: clusters.map(clusterLabel),
    y: clusters.map(sizeOf),
    marker: { color: clusters.map(colorFor) },
    hovertemplate: '%{x}: %{y} samples<extra></extra>',
  };

  const profileTrace = {
    type: 'heatmap',
    z: profileMedians,
    x: profileColumns,
    y: clusters.map(cluster => `${clusterLabel(cluster)} (n=${sizeOf(cluster)})`),
    colorscale: 'RdBu',
    reversescale: true,
    zmid: 0,
    colorbar: { title: { text: 'Median' }, thickness: 12 },
    hovertemplate: '%{y}<br>%{x}: %{z:.3f}<extra></extra>',
  };

  return (
    <div className="compare-panel">
      <h3>{title}: Run #{run.number} — {run.clustering.algorithm}</h3>
      <p className="compare-params">
        <strong>Reduction:</strong> {run.reduction.method}
        {run.reduction.method !== 'None' && ` (${formatParams(run.reduction.params)})`}
        <br />
        <strong>Parameters:</strong> {formatParams(run.clustering.params)}
      </p>

      <table className="compare-stats">
        <tbody>
          <tr><th>Clusters</th><td>{run.n_clusters}</td></tr>
          <tr><th>Noise</th><td>{noiseCount} ({formatNumber((100 * noiseCount) / total, 1)}%)</td></tr>
          <tr><th>Silhouette</th><td>{formatNumber(run.metrics.silhouette_score)}</td></tr>
          <tr><th>Calinski–Harabasz</th><td>{formatNumber(run.metrics.calinski_harabasz_score)}</td></tr>
          <tr><th>Davies–Bouldin</th><td>{formatNumber(run.metrics.davies_bouldin_score)}</td></tr>
          <tr>
            <th>Runtime</th>
            <td>{formatNumber(run.runtime_seconds.total, 2)} s total ({formatNumber(run.runtime_seconds.clustering, 2)} s clustering)</td>
          </tr>
          <tr><th>Peak memory</th><td>{formatNumber(run.peak_memory_mb, 1)} MB</td></tr>
          <tr>
            <th>Warnings</th>
            <td>
              {run.warnings.length > 0 ? (
                <ul>{run.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>
              ) : 'None'}
            </td>
          </tr>
        </tbody>
      </table>

      <Plot
        data={[scatterTrace]}
        layout={{
          title: { text: result.sampling?.sampled ? `Scatter plot (${result.sampling.shown} of ${result.sampling.total} rows)` : 'Scatter plot' },
          autosize: true,
          height: 380,
          margin: { l: 40, r: 10, t: 40, b: 40 },
          hovermode: 'closest',
          ...PLOT_COLORS,
        }}
        config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: `run_${run.number}_scatter` } }}
        useResizeHandler
        style={{ width: '100%' }}
      />

      <Plot
        data={[sizeTrace]}
        layout={{
          title: { text: 'Cluster sizes' },
          autosize: true,
          height: 260,
          margin: { l: 50, r: 10, t: 40, b: 40 },
          xaxis: { type: 'category' },
          yaxis: { title: { text: 'Samples' } },
          ...PLOT_COLORS,
        }}
        config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: `run_${run.number}_cluster_sizes` } }}
        useResizeHandler
        style={{ width: '100%' }}
      />

      {profileColumns.length > 0 && (
        <Plot
          data={[profileTrace]}
          layout={{
            title: { text: `Cluster median profiles (${result.value_label || 'log2(x + 1), per-row z-score'})` },
            autosize: true,
            height: Math.max(260, clusters.length * 26 + 140),
            margin: { l: 110, r: 10, t: 40, b: 80 },
            yaxis: { autorange: 'reversed', type: 'category' },
            xaxis: { tickangle: profileColumns.length > 12 ? -45 : 0 },
            ...PLOT_COLORS,
          }}
          config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: `run_${run.number}_profiles` } }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      )}
    </div>
  );
}

function RunComparison({ runs, apiBase, sessionId }) {
  const [isOpen, setIsOpen] = useState(false);
  const [runAId, setRunAId] = useState(null);
  const [runBId, setRunBId] = useState(null);
  const [loaded, setLoaded] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [alignColors, setAlignColors] = useState(true);
  const [flowView, setFlowView] = useState('matrix');
  const [onlyDifferences, setOnlyDifferences] = useState(false);
  const [changedFilter, setChangedFilter] = useState('');
  const resultCache = useRef({});

  const completedRuns = useMemo(() => runs.filter(run => run.status === 'success'), [runs]);
  const completedKey = completedRuns.map(run => run.run_id).join(',');
  const runA = completedRuns.find(run => run.run_id === runAId);
  const runB = completedRuns.find(run => run.run_id === runBId);

  // Default to the two most recent completed runs; keep the user's choice while it stays valid.
  useEffect(() => {
    const ids = completedKey ? completedKey.split(',') : [];
    if (ids.length < 2) return;
    setRunAId(previous => (ids.includes(previous) ? previous : ids[ids.length - 2]));
    setRunBId(previous => (ids.includes(previous) ? previous : ids[ids.length - 1]));
  }, [completedKey]);

  useEffect(() => {
    if (!isOpen || !runA || !runB || runA.run_id === runB.run_id) return undefined;

    let cancelled = false;
    const getResult = async (run) => {
      if (!resultCache.current[run.run_id]) {
        resultCache.current[run.run_id] = await fetchJson(`${apiBase}${run.links.result}`);
      }
      return resultCache.current[run.run_id];
    };

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [resultA, resultB, comparison] = await Promise.all([
          getResult(runA),
          getResult(runB),
          fetchJson(`${apiBase}/sessions/${sessionId}/compare?run_a=${runA.run_id}&run_b=${runB.run_id}`),
        ]);
        if (!cancelled) {
          setLoaded({ runAId: runA.run_id, runBId: runB.run_id, resultA, resultB, comparison });
        }
      } catch (e) {
        if (!cancelled) setError(`Comparison failed: ${formatApiError(e)}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();

    return () => { cancelled = true; };
  }, [isOpen, runA, runB, apiBase, sessionId]);

  const current = loaded && runA && runB && loaded.runAId === runA.run_id && loaded.runBId === runB.run_id
    ? loaded
    : null;

  const colorForA = useMemo(() => {
    const colors = current?.resultA.cluster_colors || {};
    return (cluster) => colors[cluster] || NOISE_COLOR;
  }, [current]);

  // With aligned colors, a Run B cluster takes the color of its matched Run A cluster.
  const colorForB = useMemo(() => {
    if (!current) return () => NOISE_COLOR;
    const ownColors = current.resultB.cluster_colors || {};
    if (!alignColors) return (cluster) => ownColors[cluster] || NOISE_COLOR;

    const partnerInA = new Map(current.comparison.matches.map(m => [m.cluster_b, m.cluster_a]));
    const unmatched = current.comparison.unmatched_b;
    return (cluster) => {
      if (cluster === -1) return NOISE_COLOR;
      if (partnerInA.has(cluster)) return colorForA(partnerInA.get(cluster));
      return UNMATCHED_COLORS[unmatched.indexOf(cluster) % UNMATCHED_COLORS.length];
    };
  }, [current, alignColors, colorForA]);

  const flowOrder = useMemo(() => {
    if (!current) return null;
    const { contingency, matches } = current.comparison;
    const rowsA = sortClusters(contingency.clusters_a);
    const partnerOfA = new Map(matches.map(m => [m.cluster_a, m.cluster_b]));
    if (contingency.clusters_a.includes(-1) && contingency.clusters_b.includes(-1)) partnerOfA.set(-1, -1);
    // Order Run B columns so matched clusters sit on the diagonal.
    const matchedColumns = rowsA.map(cluster => partnerOfA.get(cluster)).filter(cluster => cluster !== undefined);
    const columnsB = [
      ...matchedColumns,
      ...sortClusters(contingency.clusters_b.filter(cluster => !matchedColumns.includes(cluster))),
    ];
    const indexA = new Map(contingency.clusters_a.map((cluster, i) => [cluster, i]));
    const indexB = new Map(contingency.clusters_b.map((cluster, j) => [cluster, j]));
    const counts = rowsA.map(a => columnsB.map(b => contingency.counts[indexA.get(a)][indexB.get(b)]));
    return { rowsA, columnsB, counts, partnerOfA };
  }, [current]);

  if (completedRuns.length < 2) return null;

  const paramRows = runA && runB ? parameterRows(runA, runB) : [];
  const visibleParamRows = onlyDifferences ? paramRows.filter(row => row.differs) : paramRows;

  const filterText = changedFilter.trim().toLowerCase();
  const changedSamples = current?.comparison.changed_samples || [];
  const filteredChanged = filterText
    ? changedSamples.filter(sample => sample.name.toLowerCase().includes(filterText))
    : changedSamples;

  const downloadChangedSamples = () => {
    const lines = [
      'Name\tRun A cluster\tRun B cluster\tRun B cluster matched to Run A cluster',
      ...changedSamples.map(sample => {
        const expected = flowOrder.partnerOfA.get(sample.cluster_a);
        return `${sample.name}\t${sample.cluster_a}\t${sample.cluster_b}\t${expected === undefined ? 'unmatched' : expected}`;
      }),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/tab-separated-values;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `changed_samples_run_${runA.number}_vs_run_${runB.number}.tsv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const renderFlowPlot = () => {
    const { rowsA, columnsB, counts } = flowOrder;
    const labelsA = rowsA.map(cluster => `A ${clusterLabel(cluster)}`);
    const labelsB = columnsB.map(cluster => `B ${clusterLabel(cluster)}`);
    const height = Math.max(320, Math.max(rowsA.length, columnsB.length) * 34 + 150);

    if (flowView === 'sankey') {
      const links = { source: [], target: [], value: [] };
      counts.forEach((row, i) => row.forEach((count, j) => {
        if (count > 0) {
          links.source.push(i);
          links.target.push(rowsA.length + j);
          links.value.push(count);
        }
      }));
      return (
        <Plot
          data={[{
            type: 'sankey',
            arrangement: 'snap',
            node: {
              label: [...labelsA, ...labelsB],
              color: [...rowsA.map(colorForA), ...columnsB.map(colorForB)],
              pad: 12,
              thickness: 16,
            },
            link: { ...links, color: 'rgba(120,120,120,0.35)' },
          }]}
          layout={{
            title: { text: `Run #${runA.number} → Run #${runB.number}` },
            autosize: true,
            height,
            margin: { l: 20, r: 20, t: 50, b: 20 },
            ...PLOT_COLORS,
          }}
          config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: 'run_comparison_sankey' } }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      );
    }

    return (
      <Plot
        data={[{
          type: 'heatmap',
          z: counts,
          x: labelsB,
          y: labelsA,
          colorscale: 'Blues',
          texttemplate: '%{z}',
          hovertemplate: 'Run A %{y} → Run B %{x}: %{z} samples<extra></extra>',
          colorbar: { title: { text: 'Samples' }, thickness: 12 },
        }]}
        layout={{
          title: { text: `Contingency matrix: Run #${runA.number} (rows) vs Run #${runB.number} (columns)` },
          autosize: true,
          height,
          margin: { l: 90, r: 20, t: 50, b: 80 },
          xaxis: { type: 'category', side: 'bottom' },
          yaxis: { type: 'category', autorange: 'reversed' },
          ...PLOT_COLORS,
        }}
        config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: 'run_comparison_contingency' } }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    );
  };

  return (
    <section className="run-comparison">
      <div className="run-comparison-header">
        <h2>Compare Runs<HelpTip text={HELP.compare} label="run comparison" /></h2>
        <button type="button" onClick={() => setIsOpen(open => !open)}>
          {isOpen ? 'Hide comparison' : 'Open comparison'}
        </button>
      </div>

      {isOpen && (
        <>
          <div className="compare-selectors">
            <label>
              Run A
              <select value={runAId || ''} onChange={(e) => setRunAId(e.target.value)}>
                {completedRuns.map(run => <option key={run.run_id} value={run.run_id}>{runOptionLabel(run)}</option>)}
              </select>
            </label>
            <button
              type="button"
              className="compare-swap"
              onClick={() => { setRunAId(runBId); setRunBId(runAId); }}
              title="Swap Run A and Run B"
            >
              ⇄
            </button>
            <label>
              Run B
              <select value={runBId || ''} onChange={(e) => setRunBId(e.target.value)}>
                {completedRuns.map(run => <option key={run.run_id} value={run.run_id}>{runOptionLabel(run)}</option>)}
              </select>
            </label>
            <label className="compare-checkbox">
              <input type="checkbox" checked={alignColors} onChange={(e) => setAlignColors(e.target.checked)} />
              Color Run B by matched Run A clusters<HelpTip text={HELP.alignColors} label="color alignment" />
            </label>
          </div>

          {runAId === runBId && <p className="compare-note">Select two different runs to compare.</p>}
          {error && <div className="error-message">{error}</div>}
          {loading && <p className="compare-note">Loading comparison…</p>}

          {current && flowOrder && (
            <>
              <div className="compare-agreement">
                <div>
                  <span className="compare-agreement-value">{formatNumber(current.comparison.adjusted_rand_index)}</span>
                  <span>Adjusted Rand Index<HelpTip text={HELP.ari} label="adjusted Rand index" /></span>
                </div>
                <div>
                  <span className="compare-agreement-value">{formatNumber(current.comparison.normalized_mutual_info)}</span>
                  <span>Normalized Mutual Information<HelpTip text={HELP.nmi} label="normalized mutual information" /></span>
                </div>
                <div>
                  <span className="compare-agreement-value">
                    {current.comparison.n_changed} ({formatNumber((100 * current.comparison.n_changed) / current.comparison.n_common, 1)}%)
                  </span>
                  <span>Samples that changed cluster</span>
                </div>
                <div>
                  <span className="compare-agreement-value">{current.comparison.n_common}</span>
                  <span>Shared samples</span>
                </div>
              </div>
              <p className="compare-note">
                ARI and NMI are 1 for identical partitions regardless of cluster numbering; noise (−1) is treated as its own group.
                {(current.comparison.n_only_a > 0 || current.comparison.n_only_b > 0) &&
                  ` ${current.comparison.n_only_a} samples are only in Run A and ${current.comparison.n_only_b} only in Run B; they are excluded.`}
              </p>

              <div className="compare-panels">
                <RunPanel title="Run A" run={runA} result={current.resultA} colorFor={colorForA} />
                <RunPanel title="Run B" run={runB} result={current.resultB} colorFor={colorForB} />
              </div>

              <div className="compare-block">
                <div className="compare-block-header">
                  <h3>Parameter differences</h3>
                  <label className="compare-checkbox">
                    <input type="checkbox" checked={onlyDifferences} onChange={(e) => setOnlyDifferences(e.target.checked)} />
                    Show only differences
                  </label>
                </div>
                <table className="compare-table">
                  <thead>
                    <tr><th>Setting</th><th>Run A (#{runA.number})</th><th>Run B (#{runB.number})</th></tr>
                  </thead>
                  <tbody>
                    {visibleParamRows.map(row => (
                      <tr key={row.label} className={row.differs ? 'param-differs' : ''}>
                        <td>{row.label}</td><td>{row.a}</td><td>{row.b}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {visibleParamRows.length === 0 && <p className="compare-note">The two runs use identical settings.</p>}
              </div>

              <div className="compare-block">
                <div className="compare-block-header">
                  <h3>How samples move between clusterings<HelpTip text={HELP.flow} label="contingency and Sankey" /></h3>
                  <div className="compare-toggle">
                    <button type="button" className={flowView === 'matrix' ? 'active' : ''} onClick={() => setFlowView('matrix')}>
                      Contingency matrix
                    </button>
                    <button type="button" className={flowView === 'sankey' ? 'active' : ''} onClick={() => setFlowView('sankey')}>
                      Sankey diagram
                    </button>
                  </div>
                </div>
                {renderFlowPlot()}
              </div>

              <div className="compare-block">
                <h3>Cluster matching (by overlap)<HelpTip text={HELP.matching} label="cluster matching" /></h3>
                <p className="compare-note">
                  Cluster numbers have no meaning across runs, so clusters are paired one-to-one by maximal Jaccard overlap
                  (shared samples ÷ samples in either cluster).
                </p>
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th>Run A cluster</th><th>Size</th><th>Matched Run B cluster</th><th>Size</th><th>Shared samples</th><th>Jaccard</th>
                    </tr>
                  </thead>
                  <tbody>
                    {current.comparison.matches.map(match => (
                      <tr key={`${match.cluster_a}-${match.cluster_b}`}>
                        <td><span className="cluster-swatch" style={{ backgroundColor: colorForA(match.cluster_a) }} />{clusterLabel(match.cluster_a)}</td>
                        <td>{match.size_a}</td>
                        <td><span className="cluster-swatch" style={{ backgroundColor: colorForB(match.cluster_b) }} />{clusterLabel(match.cluster_b)}</td>
                        <td>{match.size_b}</td>
                        <td>{match.overlap}</td>
                        <td>{formatNumber(match.jaccard)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {(current.comparison.unmatched_a.length > 0 || current.comparison.unmatched_b.length > 0) && (
                  <p className="compare-note">
                    Without a partner — Run A: {current.comparison.unmatched_a.map(clusterLabel).join(', ') || 'none'};
                    Run B: {current.comparison.unmatched_b.map(clusterLabel).join(', ') || 'none'}.
                  </p>
                )}
              </div>

              <div className="compare-block">
                <div className="compare-block-header">
                  <h3>Samples that changed cluster ({current.comparison.n_changed})<HelpTip text={HELP.changedSamples} label="changed samples" /></h3>
                  <div className="compare-toggle">
                    <input
                      type="text"
                      placeholder="Filter by name"
                      value={changedFilter}
                      onChange={(e) => setChangedFilter(e.target.value)}
                    />
                    <button type="button" onClick={downloadChangedSamples} disabled={changedSamples.length === 0}>
                      Download TSV
                    </button>
                  </div>
                </div>
                <p className="compare-note">
                  A sample counts as changed when its Run B cluster is not the cluster matched to its Run A cluster.
                  {current.comparison.changed_truncated &&
                    ` Only the first ${changedSamples.length.toLocaleString()} of ${current.comparison.n_changed.toLocaleString()} changed samples are listed and included in the TSV.`}
                </p>
                {filteredChanged.length > 0 ? (
                  <>
                    <table className="compare-table">
                      <thead>
                        <tr><th>Name</th><th>Run A cluster</th><th>Run B cluster</th><th>Expected in Run B (matched)</th></tr>
                      </thead>
                      <tbody>
                        {filteredChanged.slice(0, MAX_CHANGED_ROWS).map((sample, index) => {
                          const expected = flowOrder.partnerOfA.get(sample.cluster_a);
                          return (
                            <tr key={`${sample.name}-${index}`}>
                              <td>{sample.name}</td>
                              <td>{clusterLabel(sample.cluster_a)}</td>
                              <td>{clusterLabel(sample.cluster_b)}</td>
                              <td>{expected === undefined ? 'no match' : clusterLabel(expected)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    {filteredChanged.length > MAX_CHANGED_ROWS && (
                      <p className="compare-note">
                        Showing {MAX_CHANGED_ROWS} of {filteredChanged.length}. Download the TSV for the full list.
                      </p>
                    )}
                  </>
                ) : (
                  <p className="compare-note">No samples changed cluster{filterText && ' for this filter'}.</p>
                )}
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}

export default RunComparison;
