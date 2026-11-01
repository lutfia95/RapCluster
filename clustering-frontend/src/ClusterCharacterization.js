import React, { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { fetchJson, formatApiError } from './api';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const PLOT_CONFIG = { responsive: true, displaylogo: false };
const PLOT_COLORS = { plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc' };
const DEFAULT_PROFILE_CLUSTERS = 5;
const ROW_PREVIEW = 20;

const clusterLabel = (cluster) => (cluster === -1 ? 'Noise' : `Cluster ${cluster}`);
const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
const formatP = (value) => {
  if (value === null || value === undefined) return '–';
  return value < 1e-300 ? '< 1e-300' : value.toExponential(2);
};

const withAlpha = (hex, alpha) => {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!match) return `rgba(128, 128, 128, ${alpha})`;
  const value = parseInt(match[1], 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
};

// Per-run cache so switching tabs does not refetch.
const cache = {};

function ClusterCharacterization({ run, apiBase, clusterColors }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [profileClusters, setProfileClusters] = useState([]);
  const [featureCluster, setFeatureCluster] = useState(null);
  const [boxFeature, setBoxFeature] = useState('');
  const [showAllTests, setShowAllTests] = useState(false);
  const [showAllOutliers, setShowAllOutliers] = useState(false);
  const [copied, setCopied] = useState(null);

  useEffect(() => {
    if (!run?.links?.characterization) return undefined;
    let cancelled = false;
    const apply = (result) => {
      const real = result.clusters.filter(entry => entry.cluster !== -1);
      const largest = [...real].sort((a, b) => b.size - a.size).slice(0, DEFAULT_PROFILE_CLUSTERS).map(entry => entry.cluster);
      setData(result);
      setProfileClusters(largest.filter(cluster => result.profiles.clusters.includes(cluster)));
      setFeatureCluster(real[0]?.cluster ?? null);
      setBoxFeature(result.feature_tests?.rows[0]?.feature || result.columns[0]);
    };
    // A stability analysis adds per-cluster stability, so the cache key changes with it.
    const cacheKey = `${run.run_id}:${run.stability?.runtime_seconds ?? ''}`;
    if (cache[cacheKey]) {
      apply(cache[cacheKey]);
      return undefined;
    }
    setLoading(true);
    setError(null);
    setData(null);
    fetchJson(`${apiBase}${run.links.characterization}`)
      .then(result => {
        cache[cacheKey] = result;
        if (!cancelled) apply(result);
      })
      .catch(e => { if (!cancelled) setError(formatApiError(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [run, apiBase]);

  const colorFor = (cluster) => (cluster === -1 ? '#808080' : clusterColors[cluster] || '#333333');

  const profileTraces = useMemo(() => {
    if (!data) return [];
    const { profiles, columns } = data;
    return profileClusters.flatMap(cluster => {
      const i = profiles.clusters.indexOf(cluster);
      if (i < 0) return [];
      const color = colorFor(cluster);
      const name = clusterLabel(cluster);
      return [
        { x: columns, y: profiles.q3[i], mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: name },
        { x: columns, y: profiles.q1[i], mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: withAlpha(color, 0.2),
          hoverinfo: 'skip', showlegend: false, legendgroup: name },
        { x: columns, y: profiles.median[i], mode: 'lines', line: { color, width: 2 }, name, legendgroup: name,
          hovertemplate: `${name}<br>%{x}: median %{y:.3f}<extra></extra>` },
      ];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, profileClusters, clusterColors]);

  if (loading) return <p>Loading cluster characterization…</p>;
  if (error) return <div className="error-message">Cluster characterization: {error}</div>;
  if (!data) return null;

  const realClusters = data.clusters.filter(entry => entry.cluster !== -1);
  const selectedEntry = data.clusters.find(entry => entry.cluster === featureCluster);
  const tests = data.feature_tests;
  const featureIndex = data.columns.indexOf(boxFeature);
  const boxClusters = data.profiles.clusters;

  const toggleProfileCluster = (cluster) => setProfileClusters(previous => (
    previous.includes(cluster) ? previous.filter(c => c !== cluster) : [...previous, cluster]
  ));

  const copyMembers = async (cluster) => {
    try {
      const response = await fetch(`${apiBase}${run.links.members}?cluster=${cluster}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      await navigator.clipboard.writeText(await response.text());
      setCopied(cluster);
      setTimeout(() => setCopied(current => (current === cluster ? null : current)), 2000);
    } catch (e) {
      setCopied(`failed:${cluster}`);
    }
  };

  // Optional model columns: mean membership (HDBSCAN, mixtures), HDBSCAN persistence, mixture weights.
  const optionalColumns = [
    ['mean_membership', 'Mean membership', 'Average membership probability of the cluster members'],
    ['persistence', 'Persistence', 'HDBSCAN cluster persistence: how long the cluster survives in the density hierarchy (higher = more distinct)'],
    ['component_weight', 'Weight', 'Mixture weight of the component'],
    ['stability_jaccard', 'Stability (Jaccard)', 'Mean Jaccard overlap with the best-matching cluster in the stability solutions (≥ 0.75 stable)'],
  ].filter(([key]) => data.clusters.some(entry => entry[key] !== undefined));

  return (
    <div className="characterization">
      <p className="compare-note">
        Computed on <strong>all {data.clusters.reduce((sum, entry) => sum + entry.size, 0).toLocaleString()} rows</strong> of
        the data that was clustered ({data.value_label}), not on the plotted sample. The medoid is the most central member
        {` (estimated on up to ${data.settings.medoid_sample.toLocaleString()} members for large clusters)`}; representative
        members are those closest to it.
      </p>

      <h4>Cluster summary</h4>
      <div className="characterization-table-wrapper">
        <table className="compare-table characterization-summary">
          <thead>
            <tr>
              <th>Cluster</th><th>Size</th><th>Share</th><th>Medoid<HelpTip text={HELP.medoid} label="medoid" /></th><th>Representative members<HelpTip text={HELP.representatives} label="representative members" /></th>
              {optionalColumns.map(([key, label, title]) => <th key={key} title={title}>{label}</th>)}
              <th>Most distinguishing features (Cliff's δ)<HelpTip text={HELP.cliffsDelta} label="Cliff's delta" /></th><th>Members<HelpTip text={HELP.members} label="member lists" /></th>
            </tr>
          </thead>
          <tbody>
            {data.clusters.map(entry => (
              <tr key={entry.cluster}>
                <td><span className="color-swatch" style={{ backgroundColor: colorFor(entry.cluster) }} /> {clusterLabel(entry.cluster)}</td>
                <td>{entry.size.toLocaleString()}</td>
                <td>{(100 * entry.fraction).toFixed(1)}%</td>
                <td>{entry.medoid}</td>
                <td>{entry.representatives.join(', ')}</td>
                {optionalColumns.map(([key]) => <td key={key}>{formatNumber(entry[key])}</td>)}
                <td>
                  {entry.top_features.slice(0, 3).map(f => `${f.feature} (${f.cliffs_delta > 0 ? '+' : ''}${formatNumber(f.cliffs_delta, 2)})`).join(', ') || '–'}
                </td>
                <td className="run-actions">
                  <button type="button" className="link-button" onClick={() => copyMembers(entry.cluster)}>
                    {copied === entry.cluster ? 'Copied ✓' : copied === `failed:${entry.cluster}` ? 'Copy failed' : 'Copy'}
                  </button>
                  <a href={`${apiBase}${run.links.members}?cluster=${entry.cluster}`} download>Download</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="compare-note">
        Member lists can be pasted into enrichment tools such as g:Profiler or STRING.
      </p>

      <h4>Cluster profiles: median with interquartile range<HelpTip text={HELP.profiles} label="cluster profiles" /></h4>
      <div className="profile-cluster-picker">
        {boxClusters.map(cluster => (
          <label key={cluster} className="checkbox-label">
            <input type="checkbox" checked={profileClusters.includes(cluster)} onChange={() => toggleProfileCluster(cluster)} />
            <span className="color-swatch" style={{ backgroundColor: colorFor(cluster) }} />
            {clusterLabel(cluster)}
          </label>
        ))}
        <button type="button" className="link-button" onClick={() => setProfileClusters([])}>Clear</button>
      </div>
      {data.profiles.truncated && (
        <p className="compare-note">Profiles are available for the {boxClusters.length} largest groups.</p>
      )}
      {profileTraces.length > 0 ? (
        <Plot
          data={profileTraces}
          layout={{ autosize: true, height: 420, margin: { l: 50, r: 10, t: 20, b: 80 }, hovermode: 'closest',
            xaxis: { type: 'category' }, yaxis: { title: { text: data.value_label } }, legend: { orientation: 'h', y: -0.25 }, ...PLOT_COLORS }}
          config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: `run_${run.number}_cluster_profiles` } }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      ) : <p>Select clusters above to plot their profiles.</p>}

      {realClusters.length > 0 && selectedEntry && (
        <>
          <h4>Features that distinguish a cluster<HelpTip text={HELP.cliffsDelta} label="Cliff's delta" /></h4>
          <div className="compare-toggle">
            <label>
              Cluster{' '}
              <select value={featureCluster} onChange={(e) => setFeatureCluster(Number(e.target.value))}>
                {realClusters.map(entry => (
                  <option key={entry.cluster} value={entry.cluster}>{clusterLabel(entry.cluster)} (n={entry.size})</option>
                ))}
              </select>
            </label>
          </div>
          <p className="compare-note">{data.cliffs_delta.description}. Positive: higher in this cluster.</p>
          <div className="characterization-two-columns">
            <Plot
              data={[{
                type: 'bar', orientation: 'h',
                y: selectedEntry.top_features.map(f => f.feature).reverse(),
                x: selectedEntry.top_features.map(f => f.cliffs_delta).reverse(),
                marker: { color: selectedEntry.top_features.map(f => (f.cliffs_delta > 0 ? '#c0392b' : '#2471a3')).reverse() },
                hovertemplate: '%{y}: δ = %{x:.3f}<extra></extra>',
              }]}
              layout={{ autosize: true, height: 420, margin: { l: 90, r: 10, t: 10, b: 40 },
                xaxis: { title: { text: "Cliff's δ" }, range: [-1, 1] }, yaxis: { type: 'category' }, ...PLOT_COLORS }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: '100%' }}
            />
            <table className="compare-table">
              <thead><tr><th>Feature</th><th>Cliff's δ</th><th>Median in cluster</th><th>Median of the rest</th></tr></thead>
              <tbody>
                {selectedEntry.top_features.map(f => (
                  <tr key={f.feature}>
                    <td>
                      <button type="button" className="link-button" onClick={() => setBoxFeature(f.feature)}>{f.feature}</button>
                    </td>
                    <td>{formatNumber(f.cliffs_delta)}</td>
                    <td>{formatNumber(f.median_in)}</td>
                    <td>{formatNumber(f.median_rest)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h4>Feature distribution by cluster<HelpTip text={HELP.boxplot} label="feature distribution" /></h4>
      <div className="compare-toggle">
        <label>
          Feature{' '}
          <select value={boxFeature} onChange={(e) => setBoxFeature(e.target.value)}>
            {data.columns.map(column => <option key={column} value={column}>{column}</option>)}
          </select>
        </label>
      </div>
      {featureIndex >= 0 && (
        <Plot
          data={[{
            type: 'box',
            x: boxClusters.map(clusterLabel),
            q1: boxClusters.map((_, i) => data.profiles.q1[i][featureIndex]),
            median: boxClusters.map((_, i) => data.profiles.median[i][featureIndex]),
            q3: boxClusters.map((_, i) => data.profiles.q3[i][featureIndex]),
            lowerfence: boxClusters.map((_, i) => data.profiles.lower[i][featureIndex]),
            upperfence: boxClusters.map((_, i) => data.profiles.upper[i][featureIndex]),
            mean: boxClusters.map((_, i) => data.profiles.mean[i][featureIndex]),
            marker: { color: '#555' },
            fillcolor: 'rgba(40, 167, 69, 0.25)',
          }]}
          layout={{ autosize: true, height: 360, margin: { l: 50, r: 10, t: 20, b: 60 }, showlegend: false,
            yaxis: { title: { text: `${boxFeature} (${data.value_label})` } }, ...PLOT_COLORS }}
          config={{ ...PLOT_CONFIG, toImageButtonOptions: { format: 'svg', filename: `run_${run.number}_${boxFeature}_by_cluster` } }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      )}

      {tests && (
        <>
          <h4>Differences between clusters per feature ({tests.test})<HelpTip text={HELP.kruskal} label="feature tests" /></h4>
          <div className="characterization-warning">{tests.note}</div>
          <p className="compare-note">
            {tests.n_groups} clusters, {tests.n.toLocaleString()} rows{tests.noise_excluded ? ' (noise excluded)' : ''}.
            Effect size: {tests.effect_size}; p-values adjusted with {tests.correction}. Sorted by effect size.
          </p>
          <table className="compare-table">
            <thead><tr><th>Feature</th><th>ε²</th><th>H</th><th>p</th><th>q (BH)</th></tr></thead>
            <tbody>
              {(showAllTests ? tests.rows : tests.rows.slice(0, ROW_PREVIEW)).map(row => (
                <tr key={row.feature}>
                  <td>
                    <button type="button" className="link-button" onClick={() => setBoxFeature(row.feature)}>{row.feature}</button>
                  </td>
                  <td>{formatNumber(row.epsilon_squared)}</td>
                  <td>{formatNumber(row.H, 1)}</td>
                  <td>{formatP(row.p)}</td>
                  <td>{formatP(row.q)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {tests.rows.length > ROW_PREVIEW && (
            <button type="button" className="link-button" onClick={() => setShowAllTests(show => !show)}>
              {showAllTests ? 'Show fewer' : `Show all ${tests.rows.length} features`}
            </button>
          )}
        </>
      )}

      <h4>Outliers and noise<HelpTip text={HELP.outliers} label="outliers" /></h4>
      <p className="compare-note">
        Rows furthest from their cluster's median profile (robust z-score within the cluster; above about 3.5 is unusual).
      </p>
      {data.outliers.length > 0 ? (
        <>
          <table className="compare-table">
            <thead><tr><th>Name</th><th>Cluster</th><th>Distance to cluster median</th><th>Robust z</th></tr></thead>
            <tbody>
              {(showAllOutliers ? data.outliers : data.outliers.slice(0, ROW_PREVIEW)).map(row => (
                <tr key={`${row.name}-${row.cluster}`}>
                  <td>{row.name}</td><td>{clusterLabel(row.cluster)}</td><td>{formatNumber(row.distance)}</td>
                  <td>{formatNumber(row.robust_z, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.outliers.length > ROW_PREVIEW && (
            <button type="button" className="link-button" onClick={() => setShowAllOutliers(show => !show)}>
              {showAllOutliers ? 'Show fewer' : `Show top ${data.outliers.length}`}
            </button>
          )}
        </>
      ) : <p>No clusters with at least 5 members.</p>}
      <p>
        <strong>Noise points:</strong> {data.noise.count.toLocaleString()}
        {data.noise.count > 0 && (
          <>
            {' — '}
            {data.noise.names.slice(0, 30).join(', ')}{data.noise.count > 30 ? ' …' : ''}{' '}
            <a href={`${apiBase}${run.links.members}?cluster=-1`} download>Download all</a>
          </>
        )}
      </p>
    </div>
  );
}

export default ClusterCharacterization;
