import React from 'react';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

// Must match DEFAULT_OPTIONS in preprocessing.py; the defaults reproduce RapCluster's original pipeline.
export const DEFAULT_PREPROCESSING = {
  max_missing_fraction: 1.0,
  imputation: 'zero',
  transform: 'log2p1',
  remove_zero_rows: true,
  remove_low_variance: false,
  variance_threshold: 0.0,
  correlation_filter: false,
  correlation_threshold: 0.95,
  scaling: 'row_zscore',
};

// 'None' in the Preprocessing menu: values are clustered as uploaded. Missing cells still have to be filled
// (0) because the algorithms cannot handle them; nothing else is changed or removed.
export const NO_PREPROCESSING = {
  ...DEFAULT_PREPROCESSING,
  transform: 'none',
  remove_zero_rows: false,
  scaling: 'none',
};

const sameOptions = (options, reference) => (
  Object.keys(reference).every(key => options?.[key] === reference[key])
);
export const isDefaultPreprocessing = (options) => sameOptions(options, DEFAULT_PREPROCESSING);
export const isNoPreprocessing = (options) => sameOptions(options, NO_PREPROCESSING);

const IMPUTATION = [
  { value: 'zero', label: 'Replace with 0 (default)' },
  { value: 'min', label: 'Column minimum' },
  { value: 'half_min', label: 'Half of the column minimum' },
  { value: 'median', label: 'Column median' },
];

const TRANSFORM = [
  { value: 'log2p1', label: 'log2(x + 1) (default)' },
  { value: 'log10p1', label: 'log10(x + 1)' },
  { value: 'signed_log2', label: 'Signed log: sign(x) · log2(|x| + 1)' },
  { value: 'none', label: 'None' },
];

const SCALING = [
  {
    value: 'row_zscore',
    label: 'Per-row z-score (default)',
    hint: 'Compares the shape of each row. Suits ordered measurements of the same quantity, e.g. fractionation or time-course profiles.',
  },
  {
    value: 'row_minmax',
    label: 'Per-row min–max (0–1)',
    hint: 'Scales each row to 0–1 (maximum = 1). Keeps the profile shape; common for elution profiles.',
  },
  {
    value: 'column_zscore',
    label: 'Per-column z-score',
    hint: 'Gives every column mean 0 and standard deviation 1. Use when the columns are different measurements (e.g. the iris example).',
  },
  {
    value: 'column_robust',
    label: 'Per-column robust (median/IQR)',
    hint: 'Like the per-column z-score, but uses the median and interquartile range, so outliers have less influence.',
  },
  {
    value: 'column_minmax',
    label: 'Per-column min–max (0–1)',
    hint: 'Scales every column to 0–1. Sensitive to extreme values.',
  },
  {
    value: 'none',
    label: 'None',
    hint: 'Values are clustered as transformed. Columns with a large spread dominate the distances.',
  },
];

// mode 'none': no preprocessing, the settings are hidden. mode 'active': the settings are shown; with
// useDefaults they stay at their defaults and locked, unticking the box makes them editable.
function PreprocessingOptions({ options, onChange, mode, onModeChange, useDefaults, onUseDefaultsChange, onExplore, disabled }) {
  const set = (key, value) => onChange({ ...options, [key]: value });
  const scalingHint = SCALING.find(item => item.value === options.scaling)?.hint;
  const setNumber = (key, value, factor = 1) => {
    const number = parseFloat(value);
    if (Number.isFinite(number)) set(key, number / factor);
  };

  return (
    <section className="preprocessing-section">
      <h2>2. Preprocessing<HelpTip text={HELP.preprocessing} label="preprocessing" /></h2>
      <p className="preprocessing-intro">
        Every step is applied in this order and recorded with the run. The defaults reproduce RapCluster's original
        pipeline.
      </p>

      <div className="form-group preprocessing-mode">
        <label htmlFor="prep-mode">Preprocessing:<HelpTip text={HELP.preprocessingMode} label="preprocessing mode" /></label>
        <select id="prep-mode" value={mode} onChange={(e) => onModeChange(e.target.value)}>
          <option value="none">None</option>
          <option value="active">Active</option>
        </select>
      </div>

      {mode === 'active' && (<>
      <label className="checkbox-label use-default-preprocessing">
        <input type="checkbox" checked={useDefaults} onChange={(e) => onUseDefaultsChange(e.target.checked)} />
        Use default preprocessing<HelpTip text={HELP.useDefaultPreprocessing} label="default preprocessing" />
      </label>
      {useDefaults && <small className="locked-note">Untick to change the settings below.</small>}

      {/* A disabled fieldset locks every input inside it while still showing the values. */}
      <fieldset className="preprocessing-fields" disabled={useDefaults}>
      <div className="preprocessing-grid">
        <div className="form-group">
          <label htmlFor="prep-max-missing">Remove rows with more than … % missing values:<HelpTip text={HELP.maxMissing} label="missing values filter" /></label>
          <input
            id="prep-max-missing"
            type="number"
            min="0"
            max="100"
            step="5"
            value={Math.round(options.max_missing_fraction * 100)}
            onChange={(e) => setNumber('max_missing_fraction', e.target.value, 100)}
          />
          <small>100 keeps every row.</small>
        </div>

        <div className="form-group">
          <label htmlFor="prep-imputation">Fill remaining missing values with:<HelpTip text={HELP.imputation} label="imputation" /></label>
          <select id="prep-imputation" value={options.imputation} onChange={(e) => set('imputation', e.target.value)}>
            {IMPUTATION.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
          <small>In proteomics, missing often means "below detection" rather than zero.</small>
        </div>

        <div className="form-group">
          <label htmlFor="prep-transform">Transformation:<HelpTip text={HELP.transform} label="transformation" /></label>
          <select id="prep-transform" value={options.transform} onChange={(e) => set('transform', e.target.value)}>
            {TRANSFORM.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
          <small>log(x + 1) needs values above −1; use the signed log or none for data with negative values.</small>
        </div>

        <div className="form-group">
          <label htmlFor="prep-scaling">Scaling:<HelpTip text={HELP.scaling} label="scaling" /></label>
          <select id="prep-scaling" value={options.scaling} onChange={(e) => set('scaling', e.target.value)}>
            {SCALING.map(item => <option key={item.value} value={item.value} title={item.hint}>{item.label}</option>)}
          </select>
          {scalingHint && <small>{scalingHint}</small>}
        </div>
      </div>

      <div className="preprocessing-filters">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={options.remove_zero_rows}
            onChange={(e) => set('remove_zero_rows', e.target.checked)}
          />
          Remove all-zero rows<HelpTip text={HELP.removeZeroRows} label="all-zero rows" />
        </label>

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={options.remove_low_variance}
            onChange={(e) => set('remove_low_variance', e.target.checked)}
          />
          Remove low-variance columns<HelpTip text={HELP.removeLowVariance} label="low-variance filter" />
        </label>
        {options.remove_low_variance && (
          <span className="inline-option">
            standard deviation ≤
            <input
              type="number"
              min="0"
              step="0.01"
              value={options.variance_threshold}
              onChange={(e) => setNumber('variance_threshold', e.target.value)}
              aria-label="Variance threshold (standard deviation after transformation)"
            />
            <small>(after transformation; 0 = constant columns only)</small>
          </span>
        )}

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={options.correlation_filter}
            onChange={(e) => set('correlation_filter', e.target.checked)}
          />
          Remove highly correlated columns<HelpTip text={HELP.correlationFilter} label="correlation filter" />
        </label>
        {options.correlation_filter && (
          <span className="inline-option">
            |r| &gt;
            <input
              type="number"
              min="0.5"
              max="1"
              step="0.01"
              value={options.correlation_threshold}
              onChange={(e) => setNumber('correlation_threshold', e.target.value)}
              aria-label="Correlation threshold"
            />
            <small>Not recommended for profile data: neighbouring fractions or time points are correlated by nature.</small>
          </span>
        )}
      </div>
      </fieldset>
      </>)}

      <button type="button" className="explore-button" onClick={onExplore} disabled={disabled} title={HELP.explore}>
        Explore data with these settings
      </button>
    </section>
  );
}

export default PreprocessingOptions;
