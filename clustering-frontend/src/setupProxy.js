// Dev server only (react-scripts loads this file automatically; it is not part of the build).
// Forwards /rapcluster/api/* to the local Flask backend, the same way nginx does in Docker.
// This way the browser only talks to port 3000, so the dev app also works from other
// devices on the network, and Flask can stay bound to 127.0.0.1.
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function setupProxy(app) {
  app.use(
    createProxyMiddleware('/rapcluster/api', {
      target: process.env.RAPCLUSTER_BACKEND || 'http://127.0.0.1:5000',
      pathRewrite: { '^/rapcluster/api': '/api' },
      changeOrigin: true,
    })
  );
};
