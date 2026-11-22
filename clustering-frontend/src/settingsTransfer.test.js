import {
  configFromRun, formatMethodDetails, normalizeConfig, parseMethodDetails, parseSettingsJson, settingsJson,
} from './settingsTransfer';
import { DEFAULT_PREPROCESSING } from './PreprocessingOptions';

const algoParams = {
  KMeans: [
    { name: 'n_clusters', type: 'int', default: 8 },
    { name: 'random_state', type: 'int', default: 42 },
    { name: 'init', type: 'select', default: 'k-means++', options: ['k-means++', 'random'] },
  ],
  HDBSCAN: [
    { name: 'min_samples', type: 'int', default: 15, nullable: true },
    { name: 'cluster_selection_epsilon', type: 'float', default: 0.0 },
    { name: 'allow_single_cluster', type: 'boolean', default: false },
  ],
};
const reductionParamsDef = {
  UMAP: [
    { name: 'n_neighbors', type: 'int', default: 15 },
    { name: 'min_dist', type: 'float', default: 0.1 },
    { name: 'metric', type: 'select', default: 'euclidean', options: ['euclidean', 'cosine'] },
  ],
};
const defs = { algoParams, reductionParamsDef };

const run = {
  number: 3,
  created_at: '2026-11-22T10:00:00Z',
  dataset: { filename: 'data: v2.xlsx', sha256: 'abc', size_bytes: 10 },
  preprocessing: {
    name_column: 'Gene name', intensity_start_index: 2, sheet_name: 'Sheet 2',
    options: { ...DEFAULT_PREPROCESSING, transform: 'none', variance_threshold: 0.25, remove_low_variance: true },
  },
  reduction: { method: 'UMAP', params: { n_neighbors: 30, min_dist: 0.05, metric: 'cosine' }, clustering_dims: 2 },
  clustering: { algorithm: 'HDBSCAN', params: { min_samples: null, cluster_selection_epsilon: 0.3, allow_single_cluster: true } },
  random_seed: { reduction: null, clustering: null },
};

const expected = (settings) => {
  expect(settings.nameColumn).toBe('Gene name');
  expect(settings.intensityStartIndex).toBe(2);
  expect(settings.sheetName).toBe('Sheet 2');
  expect(settings.dataset.filename).toBe('data: v2.xlsx');
  expect(settings.preprocessingOptions).toEqual(run.preprocessing.options);
  expect(settings.reduction).toEqual({ method: 'UMAP', params: { n_neighbors: 30, min_dist: 0.05, metric: 'cosine' } });
  expect(settings.clustering).toEqual(run.clustering);
};

test('Method Details text round-trips every setting', () => {
  const text = formatMethodDetails(configFromRun(run), defs);
  const { settings, warnings } = normalizeConfig(parseMethodDetails(text), defs);
  expect(warnings).toEqual([]);
  expected(settings);
});

test('settings JSON round-trips every setting', () => {
  const { settings, warnings } = normalizeConfig(parseSettingsJson(settingsJson(configFromRun(run))), defs);
  expect(warnings).toEqual([]);
  expected(settings);
});

test('old Method Details text keeps current preprocessing and reports problems', () => {
  const text = 'Clustering Algorithm: KMeans\n  → K-Means\n- n_clusters: 4\n- init: bogus\n\nDimensionality Reduction: None\n';
  const { settings, warnings } = normalizeConfig(parseMethodDetails(text), defs);
  expect(settings.clustering.params).toEqual({ n_clusters: 4, random_state: 42, init: 'k-means++' });
  expect(settings.reduction.method).toBe('None');
  expect(settings.preprocessingOptions).toBeNull();
  expect(warnings.join(' ')).toMatch(/init = "bogus"/);
});

test('unknown algorithm and invalid JSON are rejected', () => {
  expect(() => normalizeConfig({ clustering: { algorithm: 'Foo' } }, defs)).toThrow(/Unknown clustering algorithm/);
  expect(() => parseSettingsJson('{nope')).toThrow(/not valid JSON/);
  expect(() => parseMethodDetails('hello')).toThrow(/Clustering Algorithm/);
});

test('sweep configuration uses the fixed parameters', () => {
  const { settings } = normalizeConfig({
    preprocessing: { name_column: 'Name', intensity_start_index: 1 },
    reduction: { method: 'None', params: {} },
    clustering: { algorithm: 'KMeans', fixed_params: { n_clusters: 5, random_state: 7, init: 'random' } },
    sweep: {},
  }, defs);
  expect(settings.clustering.params).toEqual({ n_clusters: 5, random_state: 7, init: 'random' });
});
