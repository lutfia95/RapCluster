// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// jsdom has no object URLs; plotly.js (via mapbox-gl) calls URL.createObjectURL when it is imported.
if (typeof window.URL.createObjectURL !== 'function') {
  window.URL.createObjectURL = () => 'blob:jsdom';
  window.URL.revokeObjectURL = () => {};
}
