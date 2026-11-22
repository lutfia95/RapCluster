// Moves the complete settings of an analysis in and out of RapCluster: as the "Method Details" text shown with the
// results and as a JSON file with the same layout as config.json in a run's ZIP download (so reproduce.py reads
// it too). Both can be pasted or loaded under "1. Upload Data" to repeat the analysis with exactly these settings.
import { DEFAULT_PREPROCESSING } from './PreprocessingOptions';

export const SETTINGS_FORMAT_VERSION = 1;

const DATA_LABELS = {
  filename: 'File',
  sha256: 'SHA-256',
  sheet_name: 'Sheet',
  name_column: 'Name column',
  intensity_start_index: 'Data start column index (0-based)',
};

const formatValue = (value) => (value === null || value === undefined ? 'None' : String(value));

// The settings of a run record (or run_record.json) in the config.json layout.
export const configFromRun = (run) => {
  if (!run?.clustering || !run?.reduction || !run?.preprocessing) return null;
  const { preprocessing } = run;
  const config = {
    rapcluster_settings: SETTINGS_FORMAT_VERSION,
    rapcluster_run: run.number ?? null,
    created_at: run.created_at ?? null,
    dataset: {
      filename: run.dataset?.filename ?? null,
      sha256: run.dataset?.sha256 ?? null,
      size_bytes: run.dataset?.size_bytes ?? null,
    },
    preprocessing: {
      name_column: preprocessing.name_column,
      intensity_start_index: preprocessing.intensity_start_index,
      sheet_name: preprocessing.sheet_name ?? null,
      options: { ...DEFAULT_PREPROCESSING, ...(preprocessing.options || {}) },
    },
    reduction: { ...run.reduction, params: { ...(run.reduction.params || {}) } },
    clustering: { algorithm: run.clustering.algorithm, params: { ...(run.clustering.params || {}) } },
  };
  if (run.random_seed) config.random_seed = run.random_seed;
  return config;
};

// Human-readable and complete: every line can be read back by parseMethodDetails.
export const formatMethodDetails = (config, { algoParams = {}, reductionParamsDef = {}, descriptions = {} } = {}) => {
  if (!config) return '';
  const { preprocessing, reduction, clustering, dataset = {} } = config;
  const paramLines = (definitions, values) => {
    const names = [...definitions.map(p => p.name), ...Object.keys(values).filter(n => !definitions.some(p => p.name === n))];
    return names.map(name => {
      const definition = definitions.find(p => p.name === name);
      return `- ${name}: ${formatValue(name in values ? values[name] : definition?.default)}`;
    });
  };

  const lines = [
    'RapCluster settings (paste this text into "1. Upload Data → Reuse settings" to repeat the analysis)',
    '',
    'Data:',
  ];
  if (dataset.filename) lines.push(`- ${DATA_LABELS.filename}: ${dataset.filename}`);
  if (dataset.sha256) lines.push(`- ${DATA_LABELS.sha256}: ${dataset.sha256}`);
  if (preprocessing.sheet_name) lines.push(`- ${DATA_LABELS.sheet_name}: ${preprocessing.sheet_name}`);
  lines.push(`- ${DATA_LABELS.name_column}: ${preprocessing.name_column}`);
  lines.push(`- ${DATA_LABELS.intensity_start_index}: ${preprocessing.intensity_start_index}`);

  lines.push('', 'Preprocessing:');
  const options = { ...DEFAULT_PREPROCESSING, ...(preprocessing.options || {}) };
  Object.keys(options).forEach(key => lines.push(`- ${key}: ${formatValue(options[key])}`));

  lines.push('', `Dimensionality Reduction: ${reduction.method}`);
  lines.push(`  → ${descriptions.reduction?.[reduction.method] || 'No description.'}`);
  if (reduction.method !== 'None') {
    lines.push(...paramLines(reductionParamsDef[reduction.method] || [], reduction.params || {}));
  }

  lines.push('', `Clustering Algorithm: ${clustering.algorithm}`);
  lines.push(`  → ${descriptions.clustering?.[clustering.algorithm] || 'No description.'}`);
  lines.push(...paramLines(algoParams[clustering.algorithm] || [], clustering.params || {}));
  return `${lines.join('\n')}\n`;
};

export const settingsJson = (config) => `${JSON.stringify(config, null, 2)}\n`;

// --- Reading settings back -------------------------------------------------------------------------------

const parseBoolean = (text) => {
  const lowered = String(text).trim().toLowerCase();
  if (['true', 'yes', '1'].includes(lowered)) return true;
  if (['false', 'no', '0'].includes(lowered)) return false;
  return undefined;
};

const isNone = (value) => value === null || value === undefined || (typeof value === 'string' && value.trim().toLowerCase() === 'none') || value === '';

// Converts a value (text from Method Details or a JSON value) to the type of a parameter definition.
const coerceParam = (definition, raw, problems, where) => {
  if (isNone(raw)) {
    if (definition.nullable || definition.default === null) return null;
    problems.push(`${where}: ${definition.name} cannot be None; the default ${formatValue(definition.default)} is used.`);
    return definition.default;
  }
  if (definition.type === 'int' || definition.type === 'float') {
    const number = Number(raw);
    if (!Number.isFinite(number) || (definition.type === 'int' && !Number.isInteger(number))) {
      problems.push(`${where}: ${definition.name} = "${raw}" is not a valid ${definition.type}; the default is used.`);
      return definition.default;
    }
    return number;
  }
  if (definition.type === 'boolean') {
    const value = typeof raw === 'boolean' ? raw : parseBoolean(raw);
    if (value === undefined) {
      problems.push(`${where}: ${definition.name} = "${raw}" is not true/false; the default is used.`);
      return definition.default;
    }
    return value;
  }
  if (definition.type === 'select') {
    const option = (definition.options || []).find(o => String(o) === String(raw).trim());
    if (option === undefined) {
      problems.push(`${where}: ${definition.name} = "${raw}" is not one of ${(definition.options || []).join(', ')}; the default is used.`);
      return definition.default;
    }
    return option;
  }
  return raw;
};

const coerceParams = (definitions, rawParams, problems, where) => {
  const params = {};
  definitions.forEach(definition => {
    params[definition.name] = definition.name in rawParams
      ? coerceParam(definition, rawParams[definition.name], problems, where)
      : definition.default;
    if (!(definition.name in rawParams)) {
      problems.push(`${where}: ${definition.name} was not given; the default ${formatValue(definition.default)} is used.`);
    }
  });
  Object.keys(rawParams)
    .filter(name => !definitions.some(p => p.name === name))
    .forEach(name => problems.push(`${where}: unknown parameter "${name}" was ignored.`));
  return params;
};

const coerceOptions = (rawOptions, problems) => {
  const options = { ...DEFAULT_PREPROCESSING };
  Object.entries(rawOptions).forEach(([key, raw]) => {
    if (!(key in DEFAULT_PREPROCESSING)) {
      problems.push(`Preprocessing: unknown option "${key}" was ignored.`);
      return;
    }
    const reference = DEFAULT_PREPROCESSING[key];
    let value = raw;
    if (typeof reference === 'number') value = Number(raw);
    else if (typeof reference === 'boolean') value = typeof raw === 'boolean' ? raw : parseBoolean(raw);
    else value = String(raw).trim();
    if (value === undefined || (typeof reference === 'number' && !Number.isFinite(value))) {
      problems.push(`Preprocessing: ${key} = "${raw}" is not valid; the default ${formatValue(reference)} is used.`);
      return;
    }
    options[key] = value;
  });
  return options;
};

// Checks a config (from JSON or Method Details) against the server's algorithm and parameter definitions.
// Returns { settings, warnings } or throws an Error with a message for the user.
export const normalizeConfig = (raw, { algoParams = {}, reductionParamsDef = {} } = {}) => {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('The settings must be a JSON object.');
  }
  const warnings = [];
  const clustering = raw.clustering || {};
  const reduction = raw.reduction || {};
  const preprocessing = raw.preprocessing || {};

  const algorithm = clustering.algorithm;
  if (!algorithm) throw new Error('No clustering algorithm found in the settings.');
  if (!algoParams[algorithm]) {
    throw new Error(`Unknown clustering algorithm "${algorithm}". Available: ${Object.keys(algoParams).join(', ')}.`);
  }
  const reductionMethod = reduction.method || 'None';
  if (reductionMethod !== 'None' && !reductionParamsDef[reductionMethod]) {
    throw new Error(`Unknown dimensionality reduction "${reductionMethod}". Available: None, ${Object.keys(reductionParamsDef).join(', ')}.`);
  }

  // A parameter-sweep configuration keeps the fixed clustering parameters in "fixed_params".
  const rawClusteringParams = clustering.params || clustering.fixed_params || {};
  if (clustering.fixed_params && raw.sweep) {
    warnings.push('This is a parameter-sweep configuration: its fixed settings were loaded, the swept values were not.');
  }

  // config.json/run records keep the options under "options"; without them RapCluster's defaults apply.
  // Method Details text from older versions has no preprocessing section: the current preprocessing is kept.
  let rawOptions = preprocessing.options;
  let keepPreprocessing = false;
  if (!rawOptions) {
    rawOptions = Object.fromEntries(Object.entries(preprocessing).filter(([key]) => key in DEFAULT_PREPROCESSING));
    if (!Object.keys(rawOptions).length && raw.keepCurrentPreprocessing) {
      keepPreprocessing = true;
      warnings.push('The text has no preprocessing section; the current preprocessing settings are kept.');
    } else if (!Object.keys(rawOptions).length) {
      warnings.push('No preprocessing options were given; the default preprocessing is used.');
    }
  }

  const startIndex = preprocessing.intensity_start_index;
  const settings = {
    dataset: { filename: raw.dataset?.filename || null, sha256: raw.dataset?.sha256 || null },
    nameColumn: preprocessing.name_column ?? null,
    intensityStartIndex: startIndex === undefined || startIndex === null || startIndex === '' ? null : Number(startIndex),
    sheetName: preprocessing.sheet_name || null,
    preprocessingOptions: keepPreprocessing ? null : coerceOptions(rawOptions, warnings),
    reduction: {
      method: reductionMethod,
      params: reductionMethod === 'None'
        ? {}
        : coerceParams(reductionParamsDef[reductionMethod], reduction.params || {}, warnings, `Reduction (${reductionMethod})`),
    },
    clustering: {
      algorithm,
      params: coerceParams(algoParams[algorithm], rawClusteringParams, warnings, `Clustering (${algorithm})`),
    },
  };
  if (settings.intensityStartIndex !== null
      && (!Number.isInteger(settings.intensityStartIndex) || settings.intensityStartIndex < 0)) {
    warnings.push(`Data start column index "${startIndex}" is not a whole number ≥ 0 and was not changed.`);
    settings.intensityStartIndex = null;
  }
  if (settings.nameColumn === null) warnings.push('No name column was given; the current one is kept.');
  if (settings.intensityStartIndex === null && startIndex === undefined) {
    warnings.push('No data start column index was given; the current one is kept.');
  }
  return { settings, warnings };
};

export const parseSettingsJson = (text) => {
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`This is not valid JSON: ${e.message}`);
  }
};

// Reads the Method Details text (as shown and downloaded with the results) into the config.json layout.
// Values stay text here; normalizeConfig converts them with the parameter definitions.
export const parseMethodDetails = (text) => {
  const config = { dataset: {}, preprocessing: { options: {} }, reduction: { params: {} }, clustering: { params: {} } };
  const dataKeys = Object.fromEntries(Object.entries(DATA_LABELS).map(([key, label]) => [label.toLowerCase(), key]));
  let section = null;
  let sawReduction = false;

  String(text).replace(/\r\n?/g, '\n').split('\n').forEach(rawLine => {
    const line = rawLine.trim();
    if (!line || line.startsWith('→') || line.startsWith('->')) return;
    let match = line.match(/^Clustering Algorithm:\s*(.+)$/i);
    if (match) {
      section = 'clustering';
      config.clustering.algorithm = match[1].trim();
      return;
    }
    match = line.match(/^Dimensionality Reduction:\s*(.+)$/i);
    if (match) {
      section = 'reduction';
      sawReduction = true;
      config.reduction.method = match[1].trim();
      return;
    }
    if (/^Data:$/i.test(line)) { section = 'data'; return; }
    if (/^Preprocessing:$/i.test(line)) { section = 'preprocessing'; return; }

    match = line.match(/^[-•*]\s*([^:]+?):\s?(.*)$/);
    if (!match || !section) return;
    const [, key, value] = match;
    if (section === 'data') {
      const dataKey = dataKeys[key.trim().toLowerCase()];
      if (!dataKey) return;
      if (dataKey === 'filename' || dataKey === 'sha256') config.dataset[dataKey] = value.trim();
      else config.preprocessing[dataKey] = value.trim();
    } else if (section === 'preprocessing') {
      config.preprocessing.options[key.trim()] = value.trim();
    } else {
      config[section].params[key.trim()] = value.trim();
    }
  });

  if (!config.clustering.algorithm) {
    throw new Error('No "Clustering Algorithm:" line found. Paste the complete Method Details text from RapCluster.');
  }
  if (!sawReduction) config.reduction.method = 'None';
  // Method Details from older RapCluster versions had no data and preprocessing sections.
  if (!Object.keys(config.preprocessing.options).length) {
    delete config.preprocessing.options;
    config.keepCurrentPreprocessing = true;
  }
  return config;
};

// Settings suggested by the results assistant name only what changes; this puts them on top of the run's
// settings. A new algorithm or reduction method starts from its own defaults instead of the old parameters.
export const mergeSuggestion = (base, suggestion, { algoParams = {}, reductionParamsDef = {} } = {}) => {
  if (!suggestion || typeof suggestion !== 'object') throw new Error('The suggestion is not a settings object.');
  const merged = JSON.parse(JSON.stringify(base));
  const defaults = (definitions) => Object.fromEntries((definitions || []).map(p => [p.name, p.default]));

  if (suggestion.preprocessing?.options) {
    merged.preprocessing.options = { ...merged.preprocessing.options, ...suggestion.preprocessing.options };
  }
  if (suggestion.reduction) {
    const method = suggestion.reduction.method ?? merged.reduction.method;
    const start = method === base.reduction.method ? base.reduction.params : defaults(reductionParamsDef[method]);
    merged.reduction = { method, params: { ...start, ...(suggestion.reduction.params || {}) } };
  }
  if (suggestion.clustering) {
    const algorithm = suggestion.clustering.algorithm ?? merged.clustering.algorithm;
    const start = algorithm === base.clustering.algorithm ? base.clustering.params : defaults(algoParams[algorithm]);
    merged.clustering = { algorithm, params: { ...start, ...(suggestion.clustering.params || {}) } };
  }
  return merged;
};
