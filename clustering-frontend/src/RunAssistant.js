import React, { useEffect, useRef, useState } from 'react';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';
import { formatApiError, readJson } from './api';

const SUGGESTED_QUESTIONS = [
  'Summarize these results in plain words.',
  'How good is this clustering? Are the clusters well separated?',
  'What makes each cluster different?',
  'How stable are the clusters?',
  'Which rows are in each cluster?',
  'Compare this run with the other runs of the session.',
  'Which settings should I try next, and why?',
  'Write a short results paragraph for a paper.',
];

// Whether a suggestion would change anything compared with the run (a different algorithm or method, or at
// least one parameter or option value).
const changesSomething = (suggestion, run) => {
  const { clustering, reduction, preprocessing } = suggestion;
  const hasValues = (object) => Boolean(object && typeof object === 'object' && Object.keys(object).length);
  return (clustering?.algorithm && clustering.algorithm !== run?.clustering?.algorithm)
    || hasValues(clustering?.params)
    || (reduction?.method && reduction.method !== run?.reduction?.method)
    || hasValues(reduction?.params)
    || hasValues(preprocessing?.options);
};

// Fenced ```json blocks of an answer that look like settings (see the assistant's system prompt); repeated
// blocks and blocks that change nothing compared with the run are left out.
export const settingsSuggestions = (text, run) => {
  const suggestions = [];
  const seen = new Set();
  const fence = /```(?:json)?\s*([\s\S]*?)```/g;
  let match;
  while ((match = fence.exec(text)) !== null) {
    try {
      const value = JSON.parse(match[1]);
      const key = JSON.stringify(value);
      if (value && typeof value === 'object' && !Array.isArray(value)
          && (value.clustering || value.reduction || value.preprocessing)
          && !seen.has(key) && changesSomething(value, run)) {
        seen.add(key);
        suggestions.push(value);
      }
    } catch (e) {
      // not (yet) valid JSON, e.g. while the answer is still being written
    }
  }
  return suggestions;
};

// Plain text with **bold** and fenced code blocks; everything else is shown as written.
const renderAnswer = (text) => text.split(/(```[\s\S]*?(?:```|$))/g).map((part, index) => {
  if (part.startsWith('```')) {
    return <pre key={index} className="assistant-code">{part.replace(/^```\w*\n?/, '').replace(/```$/, '')}</pre>;
  }
  return (
    <span key={index}>
      {part.split(/(\*\*[^*]+\*\*)/g).map((piece, j) => (
        piece.startsWith('**') && piece.endsWith('**') && piece.length > 4
          ? <strong key={j}>{piece.slice(2, -2)}</strong>
          : piece
      ))}
    </span>
  );
});

// Streams the answer (NDJSON lines {"delta"} / {"tool"} / {"replace"} / {"done"} / {"error"}); onText(text, replace)
// gets each piece, or the whole shortened answer when the server stopped a repeating answer; onTool(note) gets what
// the assistant looked up (another run, a cluster's members, ...).
const streamAnswer = async (url, messages, onText, onTool, signal) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })) }),
    signal,
  });
  if (!response.ok) {
    await readJson(response);  // throws an ApiError with the server's message
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const chunk = JSON.parse(line);
      if (chunk.error) throw new Error(chunk.error);
      if (chunk.replace !== undefined) onText(chunk.replace, true);
      else if (chunk.tool) onTool(chunk.tool);
      else if (chunk.delta) onText(chunk.delta, false);
    }
    if (done) return;
  }
};

// "Clustering Assistant — Run #N" panel: a conversation with the local language model about one run's results.
function RunAssistant({ run, url, status, messages, onMessagesChange, onApplySettings, onClose }) {
  const [question, setQuestion] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [applied, setApplied] = useState({});
  const abortRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [messages]);

  // A different run (or closing the panel) stops an answer that is still being written.
  useEffect(() => () => abortRef.current?.abort(), [run?.run_id]);

  if (!run) return null;
  const available = status?.available;

  const ask = async (text) => {
    const content = text.trim();
    if (!content || pending) return;
    const history = [...messages, { role: 'user', content }];
    let answer = '';
    let steps = [];
    const show = (content) => onMessagesChange([...history, { role: 'assistant', content, steps }]);
    show('');
    setQuestion('');
    setError(null);
    setPending(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamAnswer(url, history, (text, replace) => {
        answer = replace ? text : answer + text;
        show(answer);
      }, (note) => {
        steps = [...steps, note];
        show(answer);
      }, controller.signal);
      if (!answer.trim()) throw new Error('The assistant returned an empty answer. Please ask again.');
    } catch (e) {
      if (e.name !== 'AbortError') setError(formatApiError(e));
      // Keep a partial answer; drop an empty one so the conversation stays user/assistant alternating.
      onMessagesChange(answer.trim() ? [...history, { role: 'assistant', content: `${answer} …`, steps }] : history);
    } finally {
      setPending(false);
      abortRef.current = null;
    }
  };

  const applySuggestion = (key, suggestion) => {
    try {
      const result = onApplySettings(suggestion);
      setApplied(current => ({ ...current, [key]: { ok: true, ...result } }));
    } catch (e) {
      setApplied(current => ({ ...current, [key]: { ok: false, error: e.message } }));
    }
  };

  return (
    <section className="assistant-panel">
      <div className="assistant-header">
        <h2>
          Clustering Assistant — Run #{run.number}
          <HelpTip text={HELP.assistant} label="clustering assistant" />
        </h2>
        <button type="button" className="details-toggle" onClick={onClose}>Close ✕</button>
      </div>
      <p className="assistant-note">
        {status?.model ? `${status.model} ` : ''}runs locally on the RapCluster server. It knows all results of this
        run (metrics, clusters and their members, characterization, stability, settings, warnings) and can look up
        every other run of this session: any cluster with all its members, where a row ended up in each run, how two
        runs agree. Name runs by number (e.g. &quot;compare run 2 with run 5&quot;, &quot;cluster 3 of run 4&quot;).
        It is a small language model: check important statements against the numbers in the results.
      </p>

      {status && !available && (
        <div className="assistant-message error">
          The assistant is not available: {status.reason}
        </div>
      )}

      <div className="assistant-conversation" aria-live="polite">
        {messages.length === 0 && available && (
          <div className="assistant-suggestions">
            <p>Ask anything about this run, for example:</p>
            {SUGGESTED_QUESTIONS.map(text => (
              <button type="button" key={text} onClick={() => ask(text)} disabled={pending}>{text}</button>
            ))}
          </div>
        )}
        {messages.map((message, index) => {
          const isLast = index === messages.length - 1;
          const suggestions = message.role === 'assistant' && !(pending && isLast) ? settingsSuggestions(message.content, run) : [];
          return (
            <div key={index} className={`assistant-turn ${message.role}`}>
              <div className="assistant-turn-label">{message.role === 'user' ? 'You' : 'Assistant'}</div>
              {message.steps?.length > 0 && (
                <ul className="assistant-steps">
                  {message.steps.map((step, j) => <li key={j}>{step}</li>)}
                </ul>
              )}
              <div className="assistant-turn-text">
                {message.content
                  ? renderAnswer(message.content)
                  : <span className="assistant-thinking">
                    {message.steps?.length ? 'Reading what it looked up…' : `Reading the results of Run #${run.number}…`}
                  </span>}
              </div>
              {suggestions.map((suggestion, j) => {
                const key = `${index}-${j}`;
                const outcome = applied[key];
                return (
                  <div key={key} className="assistant-apply">
                    <button type="button" onClick={() => applySuggestion(key, suggestion)}>
                      Apply these settings to the form
                    </button>
                    {outcome && (
                      <span className={outcome.ok ? 'assistant-apply-ok' : 'assistant-apply-error'}>
                        {outcome.ok
                          ? `Applied: ${outcome.summary}${outcome.warnings.length ? ` (${outcome.warnings.join(' ')})` : ''} Run the clustering to try them.`
                          : outcome.error}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
        <div ref={endRef} />
      </div>

      {error && <div className="assistant-message error">{error}</div>}

      <form
        className="assistant-input"
        onSubmit={(e) => { e.preventDefault(); ask(question); }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              ask(question);
            }
          }}
          placeholder={available ? `Ask about Run #${run.number} or any other run of this session…` : 'The assistant is not available.'}
          rows={2}
          maxLength={2000}
          disabled={!available}
          aria-label="Question about the results"
        />
        {pending ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" disabled={!available || !question.trim()}>Ask</button>
        )}
        {messages.length > 0 && !pending && (
          <button type="button" className="secondary" onClick={() => { onMessagesChange([]); setApplied({}); setError(null); }}>
            New conversation
          </button>
        )}
      </form>
    </section>
  );
}

export default RunAssistant;
