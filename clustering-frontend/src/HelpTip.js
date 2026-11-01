import React, { useRef, useState } from 'react';

const BUBBLE_WIDTH = 340;
const MARGIN = 8;

// "?" marker that explains the element next to it on mouse hover or keyboard focus (tap on touch screens).
// The bubble uses position: fixed, so tables and panels with overflow do not clip it.
function HelpTip({ text, label }) {
  const anchor = useRef(null);
  const [position, setPosition] = useState(null);
  if (!text) return null;

  const show = () => {
    const rect = anchor.current.getBoundingClientRect();
    const width = Math.min(BUBBLE_WIDTH, window.innerWidth - 2 * MARGIN);
    const left = Math.max(MARGIN, Math.min(rect.left + rect.width / 2 - width / 2, window.innerWidth - width - MARGIN));
    const below = rect.top < 200;  // not enough room above
    setPosition({ left, width, top: below ? rect.bottom + MARGIN : rect.top - MARGIN, below });
  };
  const hide = () => setPosition(null);
  // Inside a <label> a click would otherwise toggle the checkbox or open the select.
  const swallowClick = (e) => { e.preventDefault(); e.stopPropagation(); };

  return (
    <span
      className="help-tip"
      ref={anchor}
      tabIndex={0}
      role="button"
      aria-label={label ? `About ${label}` : 'More information'}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={(e) => { if (e.key === 'Escape') hide(); }}
      onClick={swallowClick}
    >
      ?
      {position && (
        <span
          role="tooltip"
          className={`help-tip-bubble${position.below ? ' below' : ''}`}
          style={{ left: position.left, top: position.top, width: position.width }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

export default HelpTip;
