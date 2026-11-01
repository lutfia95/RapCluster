import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { fetchJson, formatApiError } from './api';
import HelpTip from './HelpTip';
import { HELP, paramHelp } from './helpTexts';

const MAX_TRIALS = 30;
const MAX_VALUES = 50;
const PLOT_CONFIG = { responsive: true, displaylogo: false };
const PLOT_COLORS = { plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc' };
const METRICS = {
  silhouette_score: { label: 'Silhouette', direction: 'max' },
  calinski_harabasz_score: { label: 'Calinski–Harabasz', direction: 'max' },
  davies_bouldin_score: { label: 'Davies–Bouldin', direction: 'min' },
  dbcv: { label: 'DBCV', direction: 'max' },
  n_clusters: { label: 'Number of clusters', direction: null },
  noise_fraction: { label: 'Noise fraction', direction: null },
  bic: { label: 'BIC', direction: 'min' },
  aic: { label: 'AIC', direction: 'min' },
};
const SCREENING_METRICS = { silhouette: 'Silhouette', calinski_harabasz: 'Calinski–Harabasz', davies_bouldin: 'Davies–Bouldin', dbcv: 'DBCV', bic: 'BIC (GaussianMixture)' };
const STATE_LABELS = {
  pending: 'waiting', running: 'running', screened: 'screened', success: 'done', failed: 'failed',
  not_confirmed: 'not confirmed', skipped: 'skipped (time limit)', cancelled: 'cancelled', not_run: 'not run',
};

const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
const formatValue = (value) => (value === null ? 'None' : String(value));

// Evenly spaced values from..to (inclusive); floats are rounded to the step's precision.
function rangeValues(definition, from, to, step) {
  const start = Number(from);
  const end = Number(to);
  const increment = Number(step);
  if (![start, end, increment].every(Number.isFinite) || increment <= 0 || end < start) return [];
  const decimals = definition.type === 'int' ? 0 : Math.min(6, (String(step).split('.')[1] || '').length);
  const values = [];
  for (let i = 0; values.length <= MAX_VALUES; i += 1) {
    const value = Number((start + i * increment).toFixed(decimals));
    if (value > end + 1e-9) break;
    values.push(value);
  }
  return values;
}

function initialEntry(definition) {
  if (!definition) return null;
  if (definition.type === 'select' || definition.type === 'boolean') {
    const options = definition.type === 'boolean' ? [true, false] : definition.options;
    return { name: definition.name, selected: options, includeNone: false };
  }
  const isInt = definition.type === 'int';
  const low = definition.min ?? 0;
  const from = definition.name === 'n_clusters' || definition.name === 'n_components' ? Math.max(2, low) : low;
  const step = definition.step ?? (isInt ? 1 : 0.1);
  const to = isInt ? Math.min(definition.max ?? from + 10, from + 10 * step) : Number((from + 9 * step).toFixed(6));
  return { name: definition.name, from: String(from), to: String(to), step: String(step), includeNone: false };
}

function entryValues(definition, entry) {
  if (!definition || !entry) return [];
  const base = (definition.type === 'select' || definition.type === 'boolean')
    ? entry.selected
    : rangeValues(definition, entry.from, entry.to, entry.step);
  return definition.nullable && entry.includeNone ? [...base, null] : base;
}

function ParameterSweep({ algorithm, paramDefs, reductionMethod, canRun, apiBase, runs, loading, onStart, onRefreshRuns, onView }) {
  const sweepable = useMemo(() => (paramDefs || []).filter(def => def.name !== 'random_state'), [paramDefs]);
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState([]);
  const [maxTrials, setMaxTrials] = useState(20);
  const [screening, setScreening] = useState({ enabled: false, fraction: 20, top_k: 3, metric: 'silhouette' });
  const [sweep, setSweep] = useState(null);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);
  const [metric, setMetric] = useState('silhouette_score');
  const [minClusters, setMinClusters] = useState('2');
  const [maxNoise, setMaxNoise] = useState('');
  const pollRef = useRef(0);

  // A new algorithm starts with its first parameter (usually the number of clusters).
  useEffect(() => {
    setEntries(sweepable.length > 0 ? [initialEntry(sweepable[0])] : []);
  }, [algorithm, sweepable]);

  useEffect(() => () => { pollRef.current += 1; }, []);

  const definitionOf = (name) => sweepable.find(def => def.name === name);
  const valueLists = entries.map(entry => entryValues(definitionOf(entry.name), entry));
  const gridSize = valueLists.reduce((product, values) => product * values.length, entries.length > 0 ? 1 : 0);
  const tooMany = valueLists.some(values => values.length > MAX_VALUES);
  const running = sweep?.job?.state === 'running';

  const updateEntry = (index, changes) => setEntries(previous => previous.map((entry, i) => (i === index ? { ...entry, ...changes } : entry)));
  const changeParameter = (index, name) => setEntries(previous => previous.map((entry, i) => (i === index ? initialEntry(definitionOf(name)) : entry)));
  const addParameter = () => {
    const unused = sweepable.find(def => !entries.some(entry => entry.name === def.name));
    if (unused) setEntries(previous => [...previous, initialEntry(unused)]);
  };

  const poll = async (initial) => {
    const token = pollRef.current + 1;
    pollRef.current = token;
    let current = initial;
    let finished = current.trials.filter(trial => trial.state === 'success').length;
    let failures = 0;
    while (current.job.state === 'running' && pollRef.current === token) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      try {
        current = await fetchJson(`${apiBase}${current.links.self}`);
        failures = 0;
      } catch (e) {
        failures += 1;
        if (failures >= 5) {
          setError(formatApiError(e));
          return;
        }
        continue;
      }
      if (pollRef.current !== token) return;
      setSweep(current);
      const done = current.trials.filter(trial => trial.state === 'success').length;
      if (done !== finished) {
        finished = done;
        onRefreshRuns();  // finished trials appear in Run History and the Leaderboard right away
      }
    }
    onRefreshRuns();
  };

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      const config = {
        parameters: entries.map((entry, i) => ({ name: entry.name, values: valueLists[i] })),
        max_trials: Number(maxTrials),
        screening: screening.enabled
          ? { enabled: true, fraction: Number(screening.fraction) / 100, top_k: Number(screening.top_k), metric: screening.metric }
          : null,
      };
      const created = await onStart(config);
      if (!created) return;
      setSweep(created);
      poll(created);
    } catch (e) {
      setError(formatApiError(e));
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    try {
      setSweep(await fetchJson(`${apiBase}${sweep.links.cancel}`, { method: 'POST' }));
      onRefreshRuns();
    } catch (e) {
      setError(formatApiError(e));
    }
  };

  const viewTrial = (trial) => {
    const run = runs.find(item => item.run_id === trial.run_id);
    if (run) onView(run);
  };

  const sweptNames = sweep ? sweep.parameters.map(parameter => parameter.name) : [];
  const successful = sweep ? sweep.trials.filter(trial => trial.state === 'success') : [];
  const metricInfo = METRICS[metric];
  const constraintFailures = (trial) => {
    const reasons = [];
    if (minClusters !== '' && trial.metrics.n_clusters < Number(minClusters)) reasons.push(`< ${minClusters} clusters`);
    if (maxNoise !== '' && 100 * (trial.metrics.noise_fraction ?? 0) > Number(maxNoise)) reasons.push(`noise > ${maxNoise}%`);
    return reasons;
  };
  const eligible = successful.filter(trial => constraintFailures(trial).length === 0 && Number.isFinite(trial.metrics[metric]));
  const best = metricInfo.direction && eligible.length > 0
    ? eligible.reduce((a, b) => {
      const better = metricInfo.direction === 'max' ? b.metrics[metric] > a.metrics[metric] : b.metrics[metric] < a.metrics[metric];
      return better ? b : a;
    })
    : null;
  const availableMetrics = Object.keys(METRICS).filter(key => successful.some(trial => Number.isFinite(trial.metrics[key])));

  const curve = () => {
    if (successful.length === 0) return null;
    const [first, second] = sweptNames;
    const categorical = (name) => successful.some(trial => typeof trial.values[name] !== 'number');
    const title = { text: `${metricInfo.label} by ${sweptNames.join(' and ')}` };
    if (!second) {
      const points = [...successful].sort((a, b) => (
        categorical(first) ? 0 : a.values[first] - b.values[first]
      ));
      return (
        <Plot
          data={[{
            type: 'scatter', mode: 'lines+markers',
            x: points.map(trial => formatValue(trial.values[first])), y: points.map(trial => trial.metrics[metric]),
            text: points.map(trial => `Run #${trial.run_number}`), hovertemplate: `${first}=%{x}: %{y:.3f} (%{text})<extra></extra>`,
            marker: { size: 9, color: points.map(trial => (trial === best ? '#28a745' : '#1f77b4')) }, line: { color: '#1f77b4' },
          }]}
          layout={{
            title, autosize: true, height: 360, margin: { l: 60, r: 20, t: 40, b: 50 },
            xaxis: { title: { text: first }, type: categorical(first) ? 'category' : 'linear' },
            yaxis: { title: { text: metricInfo.label } }, ...PLOT_COLORS,
          }}
          config={PLOT_CONFIG}
          useResizeHandler
          style={{ width: '100%' }}
        />
      );
    }
    const sortValues = (name) => {
      const values = [...new Set(successful.map(trial => formatValue(trial.values[name])))];
      return categorical(name) ? values : values.sort((a, b) => Number(a) - Number(b));
    };
    const xs = sortValues(first);
    const ys = sortValues(second);
    const z = ys.map(y => xs.map(x => {
      const trial = successful.find(item => formatValue(item.values[first]) === x && formatValue(item.values[second]) === y);
      return trial && Number.isFinite(trial.metrics[metric]) ? trial.metrics[metric] : null;
    }));
    return (
      <Plot
        data={[{
          type: 'heatmap', x: xs, y: ys, z, colorscale: 'Viridis', reversescale: metricInfo.direction === 'min',
          hovertemplate: `${first}=%{x}, ${second}=%{y}: %{z:.3f}<extra></extra>`, hoverongaps: false,
        }]}
        layout={{
          title, autosize: true, height: 380, margin: { l: 90, r: 20, t: 40, b: 60 },
          xaxis: { title: { text: first }, type: 'category' }, yaxis: { title: { text: second }, type: 'category' }, ...PLOT_COLORS,
        }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />
    );
  };

  if (!algorithm) return null;

  return (
    <section className="sweep-section">
      <div className="explore-header">
        <h2>Parameter Sweep<HelpTip text={HELP.sweep} label="parameter sweep" /></h2>
        <button type="button" className="details-toggle" onClick={() => setOpen(value => !value)} aria-expanded={open}>
          {open ? 'Hide ▲' : 'Show ▼'}
        </button>
      </div>
      {open && (
        <>
          <p className="compare-note">
            Runs {algorithm} several times with different values of one or two parameters. Preprocessing, dimensionality
            reduction ({reductionMethod}) and all other parameters are taken from the settings above and stay fixed, so the
            metrics of the trials are comparable. Every trial becomes a normal run in Run History. The whole sweep is one
            job with the usual time and memory limits.
          </p>

          {entries.map((entry, index) => {
            const definition = definitionOf(entry.name);
            const values = valueLists[index];
            return (
              <div className="sweep-parameter" key={index}>
                <label>
                  Parameter {index + 1}
                  {definition && <HelpTip text={paramHelp(algorithm, definition)} label={definition.name} />}
                  <select className="sweep-parameter-name" value={entry.name} disabled={running} onChange={(e) => changeParameter(index, e.target.value)}>
                    {sweepable.filter(def => def.name === entry.name || !entries.some(other => other.name === def.name)).map(def => (
                      <option key={def.name} value={def.name}>{def.name}</option>
                    ))}
                  </select>
                </label>
                {definition && (definition.type === 'select' || definition.type === 'boolean') ? (
                  <span className="sweep-options">
                    {(definition.type === 'boolean' ? [true, false] : definition.options).map(option => (
                      <label key={String(option)} className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={entry.selected.includes(option)}
                          disabled={running}
                          onChange={(e) => updateEntry(index, {
                            selected: e.target.checked ? [...entry.selected, option] : entry.selected.filter(item => item !== option),
                          })}
                        />
                        {String(option)}
                      </label>
                    ))}
                  </span>
                ) : definition && (
                  <span className="sweep-range">
                    <HelpTip text={HELP.sweepRange} label="value range" />
                    <label>from<input type="number" className="sweep-from" value={entry.from} step={definition.step} disabled={running} onChange={(e) => updateEntry(index, { from: e.target.value })} /></label>
                    <label>to<input type="number" className="sweep-to" value={entry.to} step={definition.step} disabled={running} onChange={(e) => updateEntry(index, { to: e.target.value })} /></label>
                    <label>step<input type="number" className="sweep-step" value={entry.step} min="0" step={definition.step} disabled={running} onChange={(e) => updateEntry(index, { step: e.target.value })} /></label>
                  </span>
                )}
                {definition?.nullable && (
                  <label className="checkbox-label">
                    <input type="checkbox" checked={entry.includeNone} disabled={running} onChange={(e) => updateEntry(index, { includeNone: e.target.checked })} />
                    also None<HelpTip text={HELP.sweepNone} label="None value" />
                  </label>
                )}
                <span className="sweep-values">
                  {values.length === 0 ? 'no values' : `${values.length} value(s): ${values.slice(0, 12).map(formatValue).join(', ')}${values.length > 12 ? ' …' : ''}`}
                  {definition && definition.min !== undefined && definition.type !== 'select' && ` (allowed ${definition.min}–${definition.max})`}
                </span>
                {entries.length > 1 && (
                  <button type="button" className="link-button" disabled={running} onClick={() => setEntries(previous => previous.filter((_, i) => i !== index))}>Remove</button>
                )}
              </div>
            );
          })}
          {entries.length === 1 && sweepable.length > 1 && (
            <button type="button" className="link-button" disabled={running} onClick={addParameter}>+ Second parameter</button>
          )}

          <div className="sweep-settings">
            <label>
              Max. trials<HelpTip text={HELP.maxTrials} label="maximum trials" />
              <input id="sweep-max-trials" type="number" min="1" max={MAX_TRIALS} value={maxTrials} disabled={running} onChange={(e) => setMaxTrials(e.target.value)} />
            </label>
            <label className="checkbox-label">
              <input id="sweep-screening" type="checkbox" checked={screening.enabled} disabled={running} onChange={(e) => setScreening(previous => ({ ...previous, enabled: e.target.checked }))} />
              Screen on a subsample first<HelpTip text={HELP.screening} label="screening" />
            </label>
            {screening.enabled && (
              <>
                <label>Subsample (%)<HelpTip text={HELP.screeningFraction} label="subsample size" /><input type="number" min="5" max="90" value={screening.fraction} disabled={running} onChange={(e) => setScreening(previous => ({ ...previous, fraction: e.target.value }))} /></label>
                <label>Confirm best<HelpTip text={HELP.screeningTopK} label="confirmed trials" /><input type="number" min="1" max={MAX_TRIALS} value={screening.top_k} disabled={running} onChange={(e) => setScreening(previous => ({ ...previous, top_k: e.target.value }))} /></label>
                <label>
                  by<HelpTip text={HELP.screeningMetric} label="screening metric" />
                  <select value={screening.metric} disabled={running} onChange={(e) => setScreening(previous => ({ ...previous, metric: e.target.value }))}>
                    {Object.entries(SCREENING_METRICS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                  </select>
                </label>
              </>
            )}
          </div>
          <p className="compare-note sweep-plan">
            {gridSize === 0 ? 'Choose at least one value per parameter.' : tooMany ? `At most ${MAX_VALUES} values per parameter.`
              : gridSize <= Number(maxTrials)
                ? `Grid search: all ${gridSize} combination(s) will be tried.`
                : `Random search: ${maxTrials} of ${gridSize} combinations, chosen at random with a fixed seed.`}
            {screening.enabled && gridSize > 0 && ` All trials run on ${screening.fraction}% of the rows; the best ${screening.top_k} are then run on all rows and saved.`}
          </p>
          {running ? (
            <div className="job-progress" role="status">
              <span className="job-progress-spinner" aria-hidden="true" />
              <span className="job-progress-text">
                <strong>Sweep running:</strong> {sweep.job.message || sweep.job.stage}
                {Number.isFinite(sweep.job.elapsed_seconds) && ` (${sweep.job.elapsed_seconds.toFixed(0)} s)`}
              </span>
              <button type="button" onClick={cancel}>Cancel sweep</button>
            </div>
          ) : (
            <button
              type="button"
              className="sweep-start"
              onClick={start}
              disabled={!canRun || loading || starting || gridSize === 0 || tooMany || !(Number(maxTrials) >= 1 && Number(maxTrials) <= MAX_TRIALS)}
            >
              {starting ? 'Starting…' : 'Start sweep'}
            </button>
          )}
          {error && <div className="error-message">{error}</div>}

          {sweep && (
            <div className="sweep-results">
              <h3>
                Sweep of {sweep.algorithm}: {sweptNames.join(' × ')} ({sweep.search === 'grid' ? 'grid' : `random ${sweep.trials.length} of ${sweep.grid_size}`})
              </h3>
              {sweep.job.state !== 'running' && sweep.job.state !== 'success' && (
                <div className="error-message">
                  {sweep.job.state === 'cancelled' ? 'The sweep was cancelled; finished trials are kept as runs.' : formatApiError({ message: sweep.job.error, hint: sweep.job.hint })}
                </div>
              )}
              {sweep.stopped_reason && <div className="sampling-note">{sweep.stopped_reason}</div>}

              {successful.length > 0 && (
                <>
                  <div className="sweep-settings">
                    <label>
                      Metric
                      <select id="sweep-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
                        {availableMetrics.map(key => <option key={key} value={key}>{METRICS[key].label}</option>)}
                      </select>
                    </label>
                    <label>Min. clusters<input type="number" min="1" value={minClusters} onChange={(e) => setMinClusters(e.target.value)} /></label>
                    <label>Max. noise (%)<input type="number" min="0" max="100" value={maxNoise} onChange={(e) => setMaxNoise(e.target.value)} /></label>
                  </div>
                  {metricInfo.direction && (
                    <div className="best-run-banner sweep-best">
                      {best ? (
                        <div>
                          <strong>
                            Best according to {metricInfo.label} ({metricInfo.direction === 'max' ? 'higher' : 'lower'} is better) under the selected constraints:
                          </strong>{' '}
                          Run #{best.run_number} ({Object.entries(best.values).map(([k, v]) => `${k}=${formatValue(v)}`).join(', ')};{' '}
                          {formatNumber(best.metrics[metric])}).
                          <div className="best-run-note">
                            Another metric or other constraints can favour another trial; internal metrics prefer certain
                            cluster shapes, so this is not the universally best clustering. Check stability before relying on it.
                          </div>
                        </div>
                      ) : <span>No trial satisfies the selected constraints.</span>}
                      {best && <div className="best-run-actions"><button type="button" onClick={() => viewTrial(best)} disabled={loading}>View</button></div>}
                    </div>
                  )}
                  {curve()}
                </>
              )}

              <div className="run-history-table-wrapper">
                <table className="compare-table sweep-trials">
                  <thead>
                    <tr>
                      <th>Trial</th>{sweptNames.map(name => <th key={name}>{name}</th>)}<th>State</th><th>Run</th>
                      <th>Clusters</th><th>Noise</th><th>Silhouette ↑</th><th>CH ↑</th><th>DB ↓</th><th>DBCV ↑</th>
                      {sweep.algorithm === 'GaussianMixture' && <><th>BIC ↓</th><th>AIC ↓</th></>}
                      {sweep.screening && <th>Screening {SCREENING_METRICS[sweep.screening.metric]}</th>}
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {sweep.trials.map(trial => {
                      const m = trial.metrics || {};
                      const screeningKey = sweep.screening && { silhouette: 'silhouette_score', calinski_harabasz: 'calinski_harabasz_score', davies_bouldin: 'davies_bouldin_score', dbcv: 'dbcv', bic: 'bic' }[sweep.screening.metric];
                      return (
                        <tr key={trial.trial} className={trial === best ? 'best-run' : (trial.state === 'success' ? '' : 'excluded-run')}>
                          <td>{trial.trial}</td>
                          {sweptNames.map(name => <td key={name}>{formatValue(trial.values[name])}</td>)}
                          <td title={trial.error || ''}>
                            {STATE_LABELS[trial.state] || trial.state}
                            {trial.state === 'failed' && ` (${trial.error_code}${trial.phase ? `, ${trial.phase}` : ''})`}
                          </td>
                          <td>{trial.run_number ? `#${trial.run_number}` : '–'}</td>
                          <td>{m.n_clusters ?? '–'}</td>
                          <td>{Number.isFinite(m.noise_fraction) ? `${(100 * m.noise_fraction).toFixed(1)}%` : '–'}</td>
                          <td>{formatNumber(m.silhouette_score)}</td>
                          <td>{formatNumber(m.calinski_harabasz_score, 1)}</td>
                          <td>{formatNumber(m.davies_bouldin_score)}</td>
                          <td>{formatNumber(m.dbcv)}</td>
                          {sweep.algorithm === 'GaussianMixture' && <><td>{formatNumber(m.bic, 1)}</td><td>{formatNumber(m.aic, 1)}</td></>}
                          {sweep.screening && <td>{formatNumber(trial.screening_metrics?.[screeningKey])}</td>}
                          <td>{trial.state === 'success' && <button type="button" onClick={() => viewTrial(trial)} disabled={loading}>View</button>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {sweep.trials.some(trial => trial.state === 'failed') && (
                <div className="sweep-failures">
                  <h4>Failed trials</h4>
                  <ul>
                    {sweep.trials.filter(trial => trial.state === 'failed').map(trial => (
                      <li key={trial.trial}>
                        Trial {trial.trial} ({Object.entries(trial.values).map(([k, v]) => `${k}=${formatValue(v)}`).join(', ')}):{' '}
                        <code>{trial.error_code}</code> — {trial.error}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {sweep.screening && (
                <p className="compare-note">
                  Screening ran every trial on {sweep.screening.n_rows?.toLocaleString() ?? `${100 * sweep.screening.fraction}% of the`} rows and
                  confirmed the best {sweep.screening.top_k} by {SCREENING_METRICS[sweep.screening.metric]} on all rows. Screening metrics
                  come from a subsample and are only used for this selection.
                </p>
              )}
              <a className="download-all-button" href={`${apiBase}${sweep.links.config}`}>Download sweep configuration (JSON)</a>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default ParameterSweep;
