import React, { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { fetchJson, formatApiError, pollJob } from './api';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const CRITERIA = {
  silhouette: {
    label: 'Silhouette score', direction: 'max', short: 'Silhouette',
    value: run => run.metrics?.silhouette_score,
  },
  calinski_harabasz: {
    label: 'Calinski–Harabasz score', direction: 'max', short: 'CH',
    value: run => run.metrics?.calinski_harabasz_score,
  },
  davies_bouldin: {
    label: 'Davies–Bouldin score', direction: 'min', short: 'DB',
    value: run => run.metrics?.davies_bouldin_score,
  },
  dbcv: {
    label: 'DBCV (density-based validation)', direction: 'max', short: 'DBCV',
    value: run => run.metrics?.dbcv,
  },
  stability: {
    label: 'stability (mean ARI of repeated solutions vs. the run)', direction: 'max', short: 'Stability',
    value: run => run.stability?.mean_ari,
  },
  runtime: {
    label: 'runtime', direction: 'min', short: 'Runtime',
    value: run => run.runtime_seconds?.total,
  },
};
const DEFAULT_PARETO_OBJECTIVES = ['silhouette', 'calinski_harabasz', 'davies_bouldin'];
const EMPTY_CONSTRAINTS = {
  minClusters: '', maxClusters: '', maxNoisePct: '', minClusterSize: '', maxRuntime: '', maxImbalance: '', minStability: '',
};
// "Recommended" needs acceptable quality AND stability; the thresholds are shown and editable.
const DEFAULT_RECOMMENDATION = { quality: 'silhouette', minQuality: '0.25', minStability: '0.75' };
const QUALITY_OPTIONS = { silhouette: 'Silhouette', dbcv: 'DBCV' };

function recommendationFailures(run, rule) {
  const quality = CRITERIA[rule.quality].value(run);
  const minQuality = parseLimit(rule.minQuality);
  const minStability = parseLimit(rule.minStability);
  const reasons = [];
  if (run.status !== 'success' || run.n_clusters < 2) return ['fewer than 2 clusters'];
  if (!run.stability) reasons.push('stability not computed');
  else if (minStability !== null && run.stability.mean_ari < minStability) reasons.push(`stability ${formatNumber(run.stability.mean_ari)} < ${minStability}`);
  if (!Number.isFinite(quality)) reasons.push(`${QUALITY_OPTIONS[rule.quality]} not available`);
  else if (minQuality !== null && quality < minQuality) reasons.push(`${QUALITY_OPTIONS[rule.quality]} ${formatNumber(quality)} < ${minQuality}`);
  return reasons;
}

const formatNumber = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
const parseLimit = (text) => (text === '' || text === null ? null : Number(text));

function constraintViolations(run, constraints) {
  const reasons = [];
  const limit = Object.fromEntries(Object.entries(constraints).map(([key, text]) => [key, parseLimit(text)]));
  const noisePct = 100 * (run.metrics?.noise_fraction ?? 0);

  if (limit.minClusters !== null && run.n_clusters < limit.minClusters) reasons.push(`fewer than ${limit.minClusters} clusters`);
  if (limit.maxClusters !== null && run.n_clusters > limit.maxClusters) reasons.push(`more than ${limit.maxClusters} clusters`);
  if (limit.maxNoisePct !== null && noisePct > limit.maxNoisePct) reasons.push(`noise ${noisePct.toFixed(1)}% > ${limit.maxNoisePct}%`);
  if (limit.minClusterSize !== null && (run.metrics?.min_cluster_size ?? 0) < limit.minClusterSize) {
    reasons.push(`smallest cluster ${run.metrics?.min_cluster_size ?? '?'} < ${limit.minClusterSize}`);
  }
  if (limit.maxRuntime !== null && run.runtime_seconds.total > limit.maxRuntime) {
    reasons.push(`runtime ${run.runtime_seconds.total.toFixed(1)} s > ${limit.maxRuntime} s`);
  }
  if (limit.maxImbalance !== null && (run.metrics?.cluster_size_imbalance ?? Infinity) > limit.maxImbalance) {
    reasons.push(`size imbalance ${formatNumber(run.metrics?.cluster_size_imbalance, 1)} > ${limit.maxImbalance}`);
  }
  if (limit.minStability !== null) {
    if (!run.stability) reasons.push('stability not computed yet');
    else if (run.stability.mean_ari < limit.minStability) reasons.push(`stability ${formatNumber(run.stability.mean_ari)} < ${limit.minStability}`);
  }
  return reasons;
}

// a dominates b when it is at least as good on every objective and strictly better on one.
function dominates(a, b, objectives) {
  let strictlyBetter = false;
  for (const key of objectives) {
    const sign = CRITERIA[key].direction === 'max' ? 1 : -1;
    const difference = sign * (CRITERIA[key].value(a) - CRITERIA[key].value(b));
    if (difference < 0) return false;
    if (difference > 0) strictlyBetter = true;
  }
  return strictlyBetter;
}

function paretoFronts(runs, objectives) {
  const fronts = [];
  let remaining = [...runs];
  while (remaining.length > 0) {
    const candidates = remaining;
    const front = candidates.filter(run => !candidates.some(other => other !== run && dominates(other, run, objectives)));
    fronts.push(front);
    remaining = candidates.filter(run => !front.includes(run));
  }
  return fronts;
}

function describeConstraints(constraints) {
  const parts = [];
  if (constraints.minClusters !== '' || constraints.maxClusters !== '') {
    parts.push(`clusters ${constraints.minClusters || '2'}–${constraints.maxClusters || '∞'}`);
  }
  if (constraints.maxNoisePct !== '') parts.push(`noise ≤ ${constraints.maxNoisePct}%`);
  if (constraints.minClusterSize !== '') parts.push(`min. cluster size ≥ ${constraints.minClusterSize}`);
  if (constraints.maxRuntime !== '') parts.push(`runtime ≤ ${constraints.maxRuntime} s`);
  if (constraints.maxImbalance !== '') parts.push(`imbalance ≤ ${constraints.maxImbalance}`);
  if (constraints.minStability !== '') parts.push(`stability ≥ ${constraints.minStability}`);
  return parts.length > 0 ? parts.join(', ') : 'no constraints';
}

function describeRecommendation(rule) {
  return `${QUALITY_OPTIONS[rule.quality]} ≥ ${rule.minQuality || '–'} and stability (mean ARI) ≥ ${rule.minStability || '–'}`;
}

function Leaderboard({ runs, apiBase, loading, onView, onRunUpdated }) {
  const [criterion, setCriterion] = useState('silhouette');
  const [paretoObjectives, setParetoObjectives] = useState(DEFAULT_PARETO_OBJECTIVES);
  const [constraints, setConstraints] = useState(EMPTY_CONSTRAINTS);
  const [showExcluded, setShowExcluded] = useState(true);
  const [pinned, setPinned] = useState(null);
  const [stabilityPending, setStabilityPending] = useState([]);
  const [stabilityError, setStabilityError] = useState(null);
  const [recommendation, setRecommendation] = useState(DEFAULT_RECOMMENDATION);
  const [collapsed, setCollapsed] = useState(false);

  const objectives = criterion === 'pareto' ? paretoObjectives : [criterion];
  const criterionLabel = criterion === 'pareto'
    ? `Pareto ranking of ${paretoObjectives.map(key => CRITERIA[key].short).join(', ')}`
    : CRITERIA[criterion].label;

  const { ranked, excluded, bestRuns } = useMemo(() => {
    const eligible = [];
    const rejected = [];
    runs.forEach(run => {
      if (run.status !== 'success') return;
      const reasons = run.n_clusters < 2 ? ['fewer than 2 clusters (metrics undefined)'] : constraintViolations(run, constraints);
      objectives.forEach(key => {
        if (!Number.isFinite(CRITERIA[key].value(run))) {
          reasons.push(key === 'stability' ? 'stability not computed yet' : `${CRITERIA[key].short} not available`);
        }
      });
      if (reasons.length > 0) rejected.push({ run, reasons });
      else eligible.push(run);
    });

    let rows;
    if (criterion === 'pareto') {
      const first = CRITERIA[objectives[0]];
      const sign = first.direction === 'max' ? -1 : 1;
      rows = paretoFronts(eligible, objectives).flatMap((front, frontIndex) => (
        [...front]
          .sort((a, b) => sign * (first.value(a) - first.value(b)) || a.number - b.number)
          .map(run => ({ run, rank: frontIndex + 1 }))
      ));
    } else {
      const { value, direction } = CRITERIA[criterion];
      const sign = direction === 'max' ? -1 : 1;
      rows = [...eligible]
        .sort((a, b) => sign * (value(a) - value(b)) || a.number - b.number)
        .map((run, index) => ({ run, rank: index + 1 }));
    }

    const best = criterion === 'pareto'
      ? rows.filter(row => row.rank === 1).map(row => row.run)
      : rows.slice(0, 1).map(row => row.run);
    return { ranked: rows, excluded: rejected, bestRuns: best };
    // objectives is derived from criterion and paretoObjectives
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs, criterion, paretoObjectives, constraints]);

  const successfulRuns = runs.filter(run => run.status === 'success');
  if (successfulRuns.length === 0) return null;

  const pinnedRun = pinned && runs.find(run => run.run_id === pinned.runId);
  const runsWithoutStability = successfulRuns.filter(run => !run.stability && run.n_clusters >= 2);
  const recommendedIds = new Set(successfulRuns.filter(run => recommendationFailures(run, recommendation).length === 0).map(run => run.run_id));
  const plottedRuns = successfulRuns.filter(run => run.stability && Number.isFinite(CRITERIA[recommendation.quality].value(run)));
  const updateRecommendation = (key, value) => setRecommendation(previous => ({ ...previous, [key]: value }));

  const computeStability = async (targetRuns) => {
    setStabilityError(null);
    for (const run of targetRuns) {
      setStabilityPending(previous => [...previous, run.run_id]);
      try {
        // Runs as a background job on the server; poll until it finishes.
        // Server defaults: 10 subsamples, plus 5 other seeds for seeded algorithms.
        await fetchJson(`${apiBase}${run.links.stability}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
        });
        const status = await pollJob(apiBase, run.links.status, { kind: 'stability' });
        if (status.run) onRunUpdated(status.run);
        if (status.state !== 'success') {
          setStabilityError(`Stability for run #${run.number} ${status.state === 'cancelled' ? 'was cancelled' : 'failed'}: ${formatApiError({ message: status.error, hint: status.hint })}`);
        }
      } catch (e) {
        setStabilityError(`Stability for run #${run.number} failed: ${formatApiError(e)}`);
      } finally {
        setStabilityPending(previous => previous.filter(id => id !== run.run_id));
      }
    }
  };

  const updateConstraint = (key, value) => setConstraints(previous => ({ ...previous, [key]: value }));
  const toggleObjective = (key) => setParetoObjectives(previous => (
    previous.includes(key) ? previous.filter(item => item !== key) : [...previous, key]
  ));

  const pinBest = () => {
    if (bestRuns.length === 0) return;
    setPinned({ runId: bestRuns[0].run_id, criterionLabel, constraintsText: describeConstraints(constraints) });
  };

  const renderStability = (run) => {
    if (run.stability) {
      const seedsText = run.stability.n_seeds ? ` and ${run.stability.n_seeds} other seeds` : '';
      const clustersText = run.stability.n_unstable_clusters !== undefined
        ? `; ${run.stability.n_stable_clusters} stable / ${run.stability.n_doubtful_clusters} doubtful / ${run.stability.n_unstable_clusters} unstable clusters`
        : '';
      return (
        <span title={`± ${formatNumber(run.stability.std_ari)} over ${run.stability.n_resamples} subsamples${seedsText}${clustersText}`}>
          {formatNumber(run.stability.mean_ari)}
        </span>
      );
    }
    if (stabilityPending.includes(run.run_id)) return 'computing…';
    if (run.n_clusters < 2) return '–';
    return (
      <button type="button" className="link-button" onClick={() => computeStability([run])} disabled={stabilityPending.length > 0}>
        compute
      </button>
    );
  };

  const renderRow = (run, rank, reasons) => {
    const isBest = bestRuns.includes(run);
    const highlight = (key) => (objectives.includes(key) ? 'criterion-cell' : '');
    return (
      <tr key={run.run_id} className={reasons ? 'excluded-run' : (isBest ? 'best-run' : '')}>
        <td>{reasons ? '–' : rank}{isBest && ' ★'}</td>
        <td>
          #{run.number}
          {recommendedIds.has(run.run_id) && (
            <span className="recommended-badge" title={`Meets the recommendation thresholds: ${describeRecommendation(recommendation)}`}>Recommended</span>
          )}
        </td>
        <td>{run.clustering.algorithm}</td>
        <td>{run.reduction.method}</td>
        <td>{run.n_clusters}</td>
        <td>{formatNumber(100 * (run.metrics?.noise_fraction ?? 0), 1)}%</td>
        <td>{run.metrics?.min_cluster_size ?? '–'}</td>
        <td>{formatNumber(run.metrics?.cluster_size_imbalance, 1)}</td>
        <td className={highlight('silhouette')}>{formatNumber(run.metrics?.silhouette_score)}</td>
        <td className={highlight('calinski_harabasz')}>{formatNumber(run.metrics?.calinski_harabasz_score, 1)}</td>
        <td className={highlight('davies_bouldin')}>{formatNumber(run.metrics?.davies_bouldin_score)}</td>
        <td className={highlight('dbcv')}>{formatNumber(run.metrics?.dbcv)}</td>
        <td className={highlight('stability')}>{renderStability(run)}</td>
        <td className={highlight('runtime')}>{formatNumber(run.runtime_seconds.total, 2)}</td>
        <td className="excluded-reason">{reasons ? reasons.join('; ') : ''}</td>
        <td>
          <button type="button" onClick={() => onView(run)} disabled={loading}>View</button>
        </td>
      </tr>
    );
  };

  return (
    <section className="leaderboard">
      <div className="section-header">
        <h2>Run Leaderboard<HelpTip text={HELP.leaderboard} label="leaderboard" /></h2>
        <button type="button" className="details-toggle" onClick={() => setCollapsed(value => !value)} aria-expanded={!collapsed}>
          {collapsed ? 'Show ▼' : 'Hide ▲'}
        </button>
      </div>
      {!collapsed && (<>

      <div className="leaderboard-controls">
        <label>
          Rank by<HelpTip text={HELP.rankBy} label="ranking criterion" />
          <select value={criterion} onChange={(e) => setCriterion(e.target.value)}>
            <option value="silhouette">Maximize Silhouette score</option>
            <option value="calinski_harabasz">Maximize Calinski–Harabasz score</option>
            <option value="davies_bouldin">Minimize Davies–Bouldin score</option>
            <option value="runtime">Minimize runtime</option>
            <option value="stability">Maximize stability</option>
            <option value="dbcv">Maximize DBCV (density-based clustering)</option>
            <option value="pareto">Multi-objective Pareto ranking</option>
          </select>
        </label>

        {criterion === 'pareto' && (
          <fieldset className="pareto-objectives">
            <legend>Objectives<HelpTip text={HELP.pareto} label="Pareto ranking" /></legend>
            {Object.entries(CRITERIA).map(([key, { short, direction }]) => (
              <label key={key}>
                <input type="checkbox" checked={paretoObjectives.includes(key)} onChange={() => toggleObjective(key)} />
                {short} ({direction})
              </label>
            ))}
          </fieldset>
        )}
      </div>

      <fieldset className="leaderboard-constraints">
        <legend>Constraints (leave empty for no limit)<HelpTip text={HELP.constraints} label="constraints" /></legend>
        <label>Min. clusters<input type="number" min="2" value={constraints.minClusters} onChange={(e) => updateConstraint('minClusters', e.target.value)} /></label>
        <label>Max. clusters<input type="number" min="2" value={constraints.maxClusters} onChange={(e) => updateConstraint('maxClusters', e.target.value)} /></label>
        <label>Max. noise (%)<HelpTip text={HELP.maxNoise} label="maximum noise" /><input type="number" min="0" max="100" value={constraints.maxNoisePct} onChange={(e) => updateConstraint('maxNoisePct', e.target.value)} /></label>
        <label>Min. cluster size<HelpTip text={HELP.minClusterSize} label="minimum cluster size" /><input type="number" min="1" value={constraints.minClusterSize} onChange={(e) => updateConstraint('minClusterSize', e.target.value)} /></label>
        <label>Max. runtime (s)<input type="number" min="0" step="0.1" value={constraints.maxRuntime} onChange={(e) => updateConstraint('maxRuntime', e.target.value)} /></label>
        <label>
          Max. size imbalance<HelpTip text={HELP.maxImbalance} label="size imbalance" /><input type="number" min="1" step="0.5" value={constraints.maxImbalance} onChange={(e) => updateConstraint('maxImbalance', e.target.value)} />
        </label>
        <label>
          Min. stability<HelpTip text={HELP.minStability} label="minimum stability" /><input type="number" min="0" max="1" step="0.05" value={constraints.minStability} onChange={(e) => updateConstraint('minStability', e.target.value)} />
        </label>
        <button type="button" className="link-button" onClick={() => setConstraints(EMPTY_CONSTRAINTS)}>Clear</button>
      </fieldset>

      <fieldset className="leaderboard-constraints recommendation-thresholds">
        <legend>"Recommended" badge: quality and stability must both be acceptable<HelpTip text={HELP.recommended} label="Recommended badge" /></legend>
        <label>
          Quality measure
          <select value={recommendation.quality} onChange={(e) => updateRecommendation('quality', e.target.value)}>
            {Object.entries(QUALITY_OPTIONS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select>
        </label>
        <label>Min. {QUALITY_OPTIONS[recommendation.quality]}<input id="recommend-min-quality" type="number" step="0.05" value={recommendation.minQuality} onChange={(e) => updateRecommendation('minQuality', e.target.value)} /></label>
        <label>Min. stability (mean ARI)<input id="recommend-min-stability" type="number" min="0" max="1" step="0.05" value={recommendation.minStability} onChange={(e) => updateRecommendation('minStability', e.target.value)} /></label>
        <button type="button" className="link-button" onClick={() => setRecommendation(DEFAULT_RECOMMENDATION)}>Reset</button>
        <span className="compare-note">
          {recommendedIds.size} run(s) recommended. Runs without a stability analysis cannot be recommended. Silhouette ≥ 0.25 is
          often read as weak but real structure; use DBCV for density-based results with noise.
        </span>
      </fieldset>

      {criterion === 'pareto' && paretoObjectives.length === 0 ? (
        <p className="compare-note">Select at least one objective.</p>
      ) : (
        <div className="best-run-banner">
          {bestRuns.length > 0 ? (
            <>
              <div>
                <strong>
                  {criterion === 'pareto'
                    ? `Pareto-optimal runs (${criterionLabel}) under the selected constraints: `
                    : `Best run according to ${criterionLabel} under the selected constraints: `}
                </strong>
                {bestRuns.map(run => `Run #${run.number} (${run.clustering.algorithm}, ${run.n_clusters} clusters)`).join(', ')}
                <div className="best-run-note">
                  Constraints: {describeConstraints(constraints)}. A different criterion or different constraints can favor
                  another run; no single score finds the best clustering for every purpose.
                </div>
              </div>
              <div className="best-run-actions">
                <button type="button" onClick={() => onView(bestRuns[0])} disabled={loading}>View</button>
                <button type="button" onClick={pinBest} title={HELP.pin}>Pin</button>
              </div>
            </>
          ) : (
            <span>No run satisfies the selected constraints{criterion === 'stability' && runsWithoutStability.length > 0 ? ' (compute stability first)' : ''}.</span>
          )}
        </div>
      )}

      {pinnedRun && (
        <div className="pinned-run-banner">
          <div>
            <strong>Pinned: Run #{pinnedRun.number}</strong> ({pinnedRun.clustering.algorithm}, {pinnedRun.reduction.method},{' '}
            {pinnedRun.n_clusters} clusters, Silhouette {formatNumber(pinnedRun.metrics?.silhouette_score)})
            <div className="best-run-note">
              Pinned as best according to {pinned.criterionLabel} with {pinned.constraintsText}. It stays here while you explore other settings.
            </div>
          </div>
          <div className="best-run-actions">
            <button type="button" onClick={() => onView(pinnedRun)} disabled={loading}>View</button>
            <button type="button" onClick={() => setPinned(null)}>Unpin</button>
          </div>
        </div>
      )}

      <div className="leaderboard-toolbar">
        <label className="compare-checkbox">
          <input type="checkbox" checked={showExcluded} onChange={(e) => setShowExcluded(e.target.checked)} />
          Show excluded runs ({excluded.length})
        </label>
        {runsWithoutStability.length > 0 && (
          <button type="button" onClick={() => computeStability(runsWithoutStability)} disabled={stabilityPending.length > 0}>
            {stabilityPending.length > 0 ? 'Computing stability…' : `Compute stability for ${runsWithoutStability.length} run(s)`}
          </button>
        )}
      </div>
      <p className="compare-note">
        Stability re-clusters 10 random 80% subsamples of a run's embedding (plus 5 runs with other random seeds for seeded
        algorithms) and averages the adjusted Rand index with the original labels (1 = perfectly stable). Cluster- and
        row-level stability are in the Stability tab of each run, where the number of solutions can be changed.
      </p>
      {stabilityError && <div className="error-message">{stabilityError}</div>}

      {plottedRuns.length > 0 && (
        <Plot
          data={[{
            type: 'scatter', mode: 'markers+text',
            x: plottedRuns.map(run => run.stability.mean_ari),
            y: plottedRuns.map(run => CRITERIA[recommendation.quality].value(run)),
            text: plottedRuns.map(run => `#${run.number}`), textposition: 'top center',
            customdata: plottedRuns.map(run => `${run.clustering.algorithm}, ${run.n_clusters} clusters`),
            hovertemplate: 'Run %{text}: %{customdata}<br>stability %{x:.3f}, quality %{y:.3f}<extra></extra>',
            marker: {
              size: 12, line: { width: 1, color: '#333' },
              color: plottedRuns.map(run => (recommendedIds.has(run.run_id) ? '#28a745' : '#adb5bd')),
            },
          }]}
          layout={{
            title: { text: `Stability versus quality (${QUALITY_OPTIONS[recommendation.quality]})` },
            autosize: true, height: 360, margin: { l: 60, r: 20, t: 40, b: 50 },
            xaxis: { title: { text: 'Stability (mean ARI)' }, range: [Math.min(0, ...plottedRuns.map(run => run.stability.mean_ari)) - 0.05, 1.05] },
            // Keep the quality threshold line inside the plot instead of on its edge.
            yaxis: {
              title: { text: QUALITY_OPTIONS[recommendation.quality] },
              range: (() => {
                const values = plottedRuns.map(run => CRITERIA[recommendation.quality].value(run));
                const threshold = parseLimit(recommendation.minQuality);
                if (threshold !== null && Number.isFinite(threshold)) values.push(threshold);
                const low = Math.min(...values);
                const high = Math.max(...values);
                const pad = Math.max(0.05, 0.1 * (high - low));
                return [low - pad, high + pad];
              })(),
            },
            shapes: [
              ...(parseLimit(recommendation.minStability) !== null ? [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: Number(recommendation.minStability), x1: Number(recommendation.minStability), line: { dash: 'dash', color: '#28a745' } }] : []),
              ...(parseLimit(recommendation.minQuality) !== null ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: Number(recommendation.minQuality), y1: Number(recommendation.minQuality), line: { dash: 'dash', color: '#28a745' } }] : []),
            ],
            plot_bgcolor: '#fcfcfc', paper_bgcolor: '#fcfcfc',
          }}
          config={{ responsive: true, displaylogo: false }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      )}
      {plottedRuns.length > 0 && (
        <p className="compare-note">
          Green: recommended (upper right of the dashed lines). Only runs with a computed stability are shown.
        </p>
      )}

      <div className="run-history-table-wrapper">
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th>Rank</th><th>Run</th><th>Algorithm</th><th>Reduction</th><th>Clusters</th><th>Noise</th>
              <th>Min. size</th><th>Imbalance</th><th>Silhouette ↑<HelpTip text={HELP.silhouette} label="Silhouette" /></th><th>CH ↑<HelpTip text={HELP.calinskiHarabasz} label="Calinski-Harabasz" /></th><th>DB ↓<HelpTip text={HELP.daviesBouldin} label="Davies-Bouldin" /></th><th>DBCV ↑<HelpTip text={HELP.dbcv} label="DBCV" /></th>
              <th>Stability ↑<HelpTip text={HELP.stabilityScore} label="stability" /></th><th>Runtime (s) ↓</th><th>Excluded because</th><th />
            </tr>
          </thead>
          <tbody>
            {ranked.map(({ run, rank }) => renderRow(run, criterion === 'pareto' ? `F${rank}` : rank, null))}
            {showExcluded && excluded.map(({ run, reasons }) => renderRow(run, null, reasons))}
          </tbody>
        </table>
      </div>
      {criterion === 'pareto' && (
        <p className="compare-note">
          F1 marks the Pareto front: runs that no other run beats on all selected objectives at once. F2 is the front after removing F1, and so on.
        </p>
      )}
      </>)}
    </section>
  );
}

export default Leaderboard;
