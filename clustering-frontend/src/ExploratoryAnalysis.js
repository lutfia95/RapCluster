import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const PLOT_CONFIG = { responsive: true, displaylogo: false };
const PLOT_COLORS = { plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc' };
const TABS = [
  ['missing', 'Missing values'],
  ['pca', 'PCA variance'],
  ['distributions', 'Distributions'],
  ['correlation', 'Correlation'],
  ['outliers', 'Outliers'],
];

const percent = (value, digits = 1) => `${(100 * value).toFixed(digits)}%`;

function PreprocessingSummary({ summary }) {
  const removedColumns = [
    ...summary.columns_removed_low_variance.map(column => `${column} (low variance)`),
    ...summary.columns_removed_correlated.map(item => `${item.column} (r with ${item.correlated_with})`),
  ];
  return (
    <ul className="explore-summary">
      <li><strong>Rows:</strong> {summary.rows_used.toLocaleString()} of {summary.rows_in.toLocaleString()} used
        {summary.rows_removed_missing > 0 && `; ${summary.rows_removed_missing} removed for missing values`}
        {summary.rows_removed_zero > 0 && `; ${summary.rows_removed_zero} all-zero rows removed`}
      </li>
      <li><strong>Columns:</strong> {summary.columns_used} of {summary.columns_in} used
        {removedColumns.length > 0 && `; removed: ${removedColumns.slice(0, 10).join(', ')}${removedColumns.length > 10 ? ' …' : ''}`}
      </li>
      <li><strong>Missing cells:</strong> {summary.missing_cells.toLocaleString()} ({summary.cells_imputed.toLocaleString()} imputed)
        {summary.columns_all_missing.length > 0 && `; columns without any value: ${summary.columns_all_missing.join(', ')}`}
      </li>
    </ul>
  );
}

function ExploratoryAnalysis({ data, onClose }) {
  const [tab, setTab] = useState('pca');
  const [scaled, setScaled] = useState(true);
  if (!data) return null;

  const { missing, pca, distributions, correlation, outliers } = data;
  const boxes = scaled ? distributions.after_scaling : distributions.before_scaling;
  const components = pca.explained_variance_ratio.map((_, i) => `PC${i + 1}`);
  const histogramCenters = outliers.histogram.counts.map((_, i) => (
    (outliers.histogram.edges[i] + outliers.histogram.edges[i + 1]) / 2
  ));

  return (
    <section className="explore-section">
      <div className="explore-header">
        <h2>Exploratory Analysis — {data.dataset.filename}</h2>
        <button type="button" className="details-toggle" onClick={onClose}>Close ✕</button>
      </div>
      <p className="explore-values">Values after preprocessing: <strong>{data.value_label}</strong></p>
      <PreprocessingSummary summary={data.summary} />
      {data.sampled_rows < data.summary.rows_used && (
        <p className="compare-note">
          PCA and correlation use a random sample of {data.sampled_rows.toLocaleString()} rows.
        </p>
      )}

      <div className="tab-buttons">
        {TABS.map(([key, label]) => (
          <button key={key} type="button" className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>
        ))}
      </div>

      {tab === 'missing' && (
        missing.missing_cells === 0 ? <p>The data has no missing values.</p> : (
          <>
            <p>
              {missing.missing_cells.toLocaleString()} of {missing.total_cells.toLocaleString()} cells are missing
              ({percent(missing.missing_cells / missing.total_cells, 2)}), counted before preprocessing.
            </p>
            <Plot
              data={[{ type: 'bar', x: missing.columns, y: missing.per_column, marker: { color: '#e67e22' } }]}
              layout={{ title: { text: 'Missing values per column' }, autosize: true, height: 320,
                margin: { l: 50, r: 10, t: 40, b: 80 }, xaxis: { type: 'category' }, yaxis: { title: { text: 'Missing cells' } }, ...PLOT_COLORS }}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: '100%' }}
            />
            <table className="compare-table">
              <thead><tr><th>Missing values per row</th><th>Rows</th></tr></thead>
              <tbody>
                {missing.rows_by_missing_fraction.map(bin => (
                  <tr key={bin.label}><td>{bin.label}</td><td>{bin.rows.toLocaleString()}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )
      )}

      {tab === 'pca' && (
        <>
          <Plot
            data={[
              { type: 'bar', x: components, y: pca.explained_variance_ratio.map(v => 100 * v), name: 'Per component', marker: { color: '#28a745' } },
              { type: 'scatter', mode: 'lines+markers', x: components, y: pca.cumulative.map(v => 100 * v), name: 'Cumulative', line: { color: '#333' } },
            ]}
            layout={{ title: { text: 'Variance explained by principal components' }, autosize: true, height: 360,
              margin: { l: 50, r: 10, t: 40, b: 40 }, yaxis: { title: { text: '% of variance' }, range: [0, 100] },
              legend: { orientation: 'h', y: -0.15 }, ...PLOT_COLORS }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
          <p className="compare-note">
            The first two components explain {percent(pca.cumulative[Math.min(1, pca.cumulative.length - 1)])} of the variance.
            A low value means a 2D PCA view hides much of the structure; UMAP or clustering on all features may be better.
          </p>
        </>
      )}

      {tab === 'distributions' && (
        <>
          <div className="compare-toggle">
            <label className="checkbox-label">
              <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} />
              After scaling (unchecked: after transformation, before scaling)
            </label>
          </div>
          <Plot
            data={[{
              type: 'box', x: distributions.columns, q1: boxes.q1, median: boxes.median, q3: boxes.q3,
              lowerfence: boxes.lower, upperfence: boxes.upper, mean: boxes.mean,
              marker: { color: '#1f77b4' }, hoverinfo: 'x+y',
            }]}
            layout={{ title: { text: `Value distribution per column (${scaled ? 'after' : 'before'} scaling)` }, autosize: true,
              height: 380, margin: { l: 50, r: 10, t: 40, b: 80 }, showlegend: false, xaxis: { type: 'category' }, ...PLOT_COLORS }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </>
      )}

      {tab === 'correlation' && (
        <>
          <Plot
            data={[{ type: 'heatmap', x: correlation.columns, y: correlation.columns, z: correlation.values,
              colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1, hovertemplate: '%{x} × %{y}: r = %{z:.2f}<extra></extra>' }]}
            layout={{ title: { text: 'Pearson correlation between columns (after preprocessing)' }, autosize: true, height: 560,
              margin: { l: 80, r: 10, t: 40, b: 80 }, xaxis: { type: 'category' },
              yaxis: { type: 'category', autorange: 'reversed' }, ...PLOT_COLORS }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
          {correlation.subset && (
            <p className="compare-note">Showing {correlation.columns.length} evenly spaced columns of {distributions.columns.length}.</p>
          )}
        </>
      )}

      {tab === 'outliers' && (
        <>
          <Plot
            data={[{ type: 'bar', x: histogramCenters, y: outliers.histogram.counts, marker: { color: '#6c757d' } }]}
            layout={{ title: { text: 'Distance of each row from the median profile' }, autosize: true, height: 300,
              margin: { l: 50, r: 10, t: 40, b: 40 }, bargap: 0.05, xaxis: { title: { text: 'Euclidean distance' } },
              yaxis: { title: { text: 'Rows' } }, ...PLOT_COLORS }}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
          <p className="compare-note">
            Rows furthest from the median profile of all rows. A robust z-score above about 3.5 is unusual; check these rows
            before clustering.
          </p>
          <table className="compare-table">
            <thead><tr><th>Name</th><th>Data row</th><th>Distance</th><th>Robust z</th></tr></thead>
            <tbody>
              {outliers.rows.map(row => (
                <tr key={`${row.name}-${row.row}`}>
                  <td>{row.name}</td><td>{row.row}</td><td>{row.distance.toFixed(3)}</td><td>{row.robust_z.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

export default ExploratoryAnalysis;
