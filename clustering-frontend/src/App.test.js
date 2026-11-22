import { fireEvent, render, screen } from '@testing-library/react';

// Plotly needs a real browser (canvas, WebGL); the page itself is what is tested here.
jest.mock('react-plotly.js', () => () => <div data-testid="plot" />);

const API = {
  '/algorithms': {
    clustering_algorithms: ['KMeans', 'HDBSCAN'],
    dimensionality_reduction_methods: ['PCA', 'UMAP'],
    algorithm_params: {
      KMeans: [
        { name: 'n_clusters', type: 'int', default: 8, min: 2, max: 50, step: 1 },
        { name: 'random_state', type: 'int', default: 42, min: 1, max: 999 },
      ],
      HDBSCAN: [
        { name: 'min_cluster_size', type: 'int', default: 5, min: 2, max: 100, step: 1 },
        { name: 'min_samples', type: 'int', default: 15, min: 1, max: 20, step: 1, nullable: true },
      ],
    },
    reduction_params: {
      PCA: [{ name: 'n_components', type: 'int', default: 2, min: 2, max: 10, step: 1 }],
      UMAP: [{ name: 'n_neighbors', type: 'int', default: 15, min: 2, max: 200, step: 1 }],
    },
  },
  '/examples': { examples: [] },
  '/limits': { max_upload_mb: 50 },
};

let App;

beforeAll(() => {
  // App.js points react-modal at #root when it is imported.
  const root = document.createElement('div');
  root.id = 'root';
  document.body.appendChild(root);
  App = require('./App').default;
});

beforeEach(() => {
  global.fetch = jest.fn(async (url) => {
    const path = Object.keys(API).find(key => String(url).endsWith(key));
    return {
      ok: Boolean(path),
      status: path ? 200 : 404,
      json: async () => (path ? API[path] : { error: 'not found' }),
      text: async () => JSON.stringify(path ? API[path] : { error: 'not found' }),
      headers: { get: () => 'application/json' },
    };
  });
});

const renderApp = async () => {
  render(<App />);
  // The algorithm list arrives from the (mocked) server.
  await screen.findByRole('option', { name: 'HDBSCAN' });
};

test('renders the input sections with the algorithms from the server', async () => {
  await renderApp();
  expect(screen.getByRole('heading', { name: /1\. Upload Data/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /3\. Dimensionality Reduction/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /4\. Clustering Algorithm/ })).toBeInTheDocument();
  expect(document.getElementById('clustering-algorithm')).toHaveValue('KMeans');
});

test('pasting Method Details applies all settings to the form', async () => {
  await renderApp();
  fireEvent.click(screen.getByRole('button', { name: 'Paste Method Details' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Paste Method Details' }), {
    target: {
      value: [
        'Data:',
        '- Name column: Gene',
        '- Data start column index (0-based): 3',
        '',
        'Preprocessing:',
        '- transform: none',
        '- scaling: column_zscore',
        '',
        'Dimensionality Reduction: UMAP',
        '- n_neighbors: 30',
        '',
        'Clustering Algorithm: HDBSCAN',
        '- min_cluster_size: 12',
        '- min_samples: None',
      ].join('\n'),
    },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }));

  expect(await screen.findByText('Settings applied:')).toBeInTheDocument();
  expect(document.getElementById('clustering-algorithm')).toHaveValue('HDBSCAN');
  expect(document.getElementById('HDBSCAN-min_cluster_size')).toHaveValue(12);
  expect(document.getElementById('reduction-method')).toHaveValue('UMAP');
  expect(document.getElementById('UMAP-n_neighbors')).toHaveValue(30);
  expect(document.getElementById('name-column')).toHaveValue('Gene');
  expect(document.getElementById('intensity-start-index')).toHaveValue(3);
});

test('invalid settings JSON shows an error and leaves the form unchanged', async () => {
  await renderApp();
  fireEvent.click(screen.getByRole('button', { name: 'Load from JSON' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Load from JSON' }), { target: { value: '{"clustering": ' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }));

  expect(await screen.findByText(/not valid JSON/)).toBeInTheDocument();
  expect(document.getElementById('clustering-algorithm')).toHaveValue('KMeans');
});
