import React, { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { TextDecoder, TextEncoder } from 'util';
import RunAssistant, { settingsSuggestions } from './RunAssistant';

global.TextDecoder = global.TextDecoder || TextDecoder;

const run = { run_id: 'r1', number: 4 };
const status = { available: true, model: 'qwen2.5:7b', reason: null };

// A fetch response whose body streams the given NDJSON lines in separate chunks.
const streamingResponse = (lines) => {
  const encoder = new TextEncoder();
  const chunks = lines.map(line => encoder.encode(`${JSON.stringify(line)}\n`));
  return {
    ok: true,
    status: 200,
    body: { getReader: () => ({ read: async () => (chunks.length ? { value: chunks.shift(), done: false } : { done: true }) }) },
  };
};

// RunAssistant keeps no conversation itself; App passes it in, as here.
function Harness({ onApplySettings = jest.fn(), panelStatus = status }) {
  const [messages, setMessages] = useState([]);
  return (
    <RunAssistant run={run} url="/api/x/assistant" status={panelStatus} messages={messages}
      onMessagesChange={setMessages} onApplySettings={onApplySettings} onClose={() => {}} />
  );
}

test('a suggested question is answered word by word', async () => {
  global.fetch = jest.fn(async () => streamingResponse([
    { delta: 'Run #4 found ' }, { delta: '**3 clusters**.' }, { done: true },
  ]));
  render(<Harness />);
  fireEvent.click(screen.getByRole('button', { name: 'Summarize these results in plain words.' }));

  expect(await screen.findByText('3 clusters')).toBeInTheDocument();
  expect(screen.getByText(/Run #4 found/)).toBeInTheDocument();
  const [, request] = global.fetch.mock.calls[0];
  expect(JSON.parse(request.body).messages).toEqual([{ role: 'user', content: 'Summarize these results in plain words.' }]);
});

test('settings in an answer can be applied to the form', async () => {
  const suggestion = { clustering: { algorithm: 'HDBSCAN', params: { min_cluster_size: 20 } } };
  global.fetch = jest.fn(async () => streamingResponse([
    { delta: 'Try HDBSCAN:\n```json\n' }, { delta: `${JSON.stringify(suggestion)}\n\`\`\`` }, { done: true },
  ]));
  const onApplySettings = jest.fn(() => ({ summary: 'no reduction → HDBSCAN.', warnings: [] }));
  render(<Harness onApplySettings={onApplySettings} />);
  fireEvent.change(screen.getByRole('textbox', { name: 'Question about the results' }), { target: { value: 'What next?' } });
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

  fireEvent.click(await screen.findByRole('button', { name: 'Apply these settings to the form' }));
  expect(onApplySettings).toHaveBeenCalledWith(suggestion);
  expect(screen.getByText(/Applied: no reduction → HDBSCAN\./)).toBeInTheDocument();
});

test('a server error is shown and the empty answer is dropped', async () => {
  global.fetch = jest.fn(async () => streamingResponse([{ error: 'The assistant model server is not reachable.' }]));
  render(<Harness />);
  fireEvent.click(screen.getByRole('button', { name: 'How stable are the clusters?' }));
  expect(await screen.findByText(/not reachable/)).toBeInTheDocument();
  expect(screen.queryByText(/Reading the results/)).not.toBeInTheDocument();
});

test('an unavailable assistant explains why and disables the input', () => {
  render(<Harness panelStatus={{ available: false, model: 'qwen2.5:7b', reason: 'The model server is not running.' }} />);
  expect(screen.getByText(/not running/)).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: 'Question about the results' })).toBeDisabled();
});

test('only settings-shaped json blocks that change something become suggestions, once each', () => {
  const kmeansRun = { clustering: { algorithm: 'MiniBatchKMeans' }, reduction: { method: 'PCA' } };
  const block = (value) => `\`\`\`json\n${JSON.stringify(value)}\n\`\`\``;
  const change = { clustering: { algorithm: 'MiniBatchKMeans', params: { n_clusters: 12 } } };
  const noChange = { clustering: { algorithm: 'MiniBatchKMeans', params: {} } };
  expect(settingsSuggestions(block({ clustering: { algorithm: 'KMeans' } }), kmeansRun)).toHaveLength(1);
  expect(settingsSuggestions(`${block(change)}\nagain:\n${block(change)}`, kmeansRun)).toEqual([change]);
  expect(settingsSuggestions(`${block(noChange)}\n${block(noChange)}`, kmeansRun)).toHaveLength(0);
  expect(settingsSuggestions(block({ other: 1 }), kmeansRun)).toHaveLength(0);
  expect(settingsSuggestions('```json\n{"clustering": \n', kmeansRun)).toHaveLength(0);
});

test('an answer the server stopped for repeating itself is replaced by the shortened answer', async () => {
  global.fetch = jest.fn(async () => streamingResponse([
    { delta: 'Loop. Loop. ' }, { replace: 'Loop.\n\n(The answer started to repeat itself and was stopped here.)' }, { done: true },
  ]));
  render(<Harness />);
  fireEvent.click(screen.getByRole('button', { name: 'What makes each cluster different?' }));
  expect(await screen.findByText(/started to repeat itself/)).toBeInTheDocument();
  expect(screen.queryByText(/Loop\. Loop\./)).not.toBeInTheDocument();
});
