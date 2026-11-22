import React, { useState } from 'react';
import HelpTip from './HelpTip';
import { HELP } from './helpTexts';

const MODES = {
  text: {
    button: 'Paste Method Details',
    placeholder: 'Paste the complete "Method Details" text of a RapCluster analysis here…',
  },
  json: {
    button: 'Load from JSON',
    placeholder: 'Paste the settings JSON here, or choose a .json file above (settings JSON or config.json from a run ZIP)…',
  },
};

// "Reuse settings" in 1. Upload Data: applies the settings of an earlier analysis to the form.
// onApply(kind, text) returns { warnings, summary } or throws an Error whose message is shown.
function SettingsImport({ onApply, disabled }) {
  const [mode, setMode] = useState(null);
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const toggleMode = (nextMode) => {
    setMode(current => (current === nextMode ? null : nextMode));
    setText('');
    setResult(null);
  };

  const apply = (content = text) => {
    try {
      setResult({ ok: true, ...onApply(mode, content) });
    } catch (e) {
      setResult({ ok: false, error: e.message });
    }
  };

  const handleJsonFile = async (event) => {
    const selected = event.target.files[0];
    event.target.value = '';
    if (!selected) return;
    const content = await selected.text();
    setText(content);
    apply(content);
  };

  return (
    <div className="form-group settings-import">
      <label>Reuse settings from a previous analysis:<HelpTip text={HELP.settingsImport} label="reuse settings" /></label>
      <div className="settings-import-buttons">
        {Object.entries(MODES).map(([key, { button }]) => (
          <button
            type="button"
            key={key}
            className={mode === key ? 'active' : ''}
            onClick={() => toggleMode(key)}
            aria-expanded={mode === key}
          >
            {button}
          </button>
        ))}
      </div>

      {mode && (
        <div className="settings-import-panel">
          {mode === 'json' && (
            <div className="settings-import-file">
              <input
                type="file"
                accept=".json,application/json"
                onChange={handleJsonFile}
                className="file-input-hidden"
                id="settings-json-file"
              />
              <label htmlFor="settings-json-file" className="file-picker-button">Choose JSON file…</label>
              <span className="settings-import-or">or paste below</span>
            </div>
          )}
          <textarea
            value={text}
            onChange={(e) => { setText(e.target.value); setResult(null); }}
            placeholder={MODES[mode].placeholder}
            rows={8}
            spellCheck={false}
            aria-label={MODES[mode].button}
          />
          <div className="settings-import-actions">
            <button type="button" onClick={() => apply()} disabled={disabled || !text.trim()}>Apply settings</button>
            <button type="button" className="secondary" onClick={() => toggleMode(mode)}>Close</button>
          </div>
          {result && !result.ok && <div className="settings-import-message error">{result.error}</div>}
          {result?.ok && (
            <div className={`settings-import-message ${result.warnings.length ? 'warning' : 'success'}`}>
              <strong>Settings applied:</strong> {result.summary}
              {result.warnings.length > 0 && (
                <ul>{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SettingsImport;
