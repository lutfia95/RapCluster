// Shared helpers for talking to the RapCluster API: readable error messages (including which
// server imposed a limit) and polling of background jobs.

export class ApiError extends Error {
  constructor(message, { status = null, code = null, hint = null, data = null } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.hint = hint;
    this.data = data;
  }
}

// Describes an error response that did not come from RapCluster as JSON, e.g. an HTML page
// from nginx or from a proxy/gateway in front of the RapCluster server.
const describeNonJsonFailure = (response, uploadLimitMb) => {
  const server = response.headers.get('Server');
  const origin = server ? `a server in front of RapCluster (it identifies itself as "${server}")` : 'a server in front of RapCluster';
  switch (response.status) {
    case 413:
      return {
        message: `The upload was rejected as too large (HTTP 413) by ${origin} before it reached RapCluster.`,
        hint: `RapCluster itself accepts files up to ${uploadLimitMb ?? 50} MB; the network or proxy you are using has a lower limit. Try a smaller or compressed (xlsx) file, or another network.`,
        code: 'upload_too_large_proxy',
      };
    case 502:
    case 503:
      return {
        message: `The RapCluster server is not reachable right now (HTTP ${response.status} from ${origin}).`,
        hint: 'The service may be restarting. Please try again in a minute.',
        code: 'gateway_error',
      };
    case 504:
      return {
        message: `The request took too long and was stopped by ${origin} (HTTP 504 gateway timeout).`,
        hint: 'Please try again. Analyses run in the background, so this usually affects only uploads or downloads.',
        code: 'gateway_timeout',
      };
    default:
      return { message: `The server answered with HTTP ${response.status}.`, hint: null, code: 'http_error' };
  }
};

// Parses a JSON response; non-2xx responses throw an ApiError with a user-facing message.
export const readJson = async (response, { uploadLimitMb } = {}) => {
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (e) {
    data = null;
  }
  if (response.ok && data !== null) return data;
  if (data && data.error) {
    throw new ApiError(data.error, { status: response.status, code: data.code, hint: data.hint, data });
  }
  if (response.ok) {
    throw new ApiError('The server sent an unreadable response.', { status: response.status, code: 'bad_response' });
  }
  const { message, hint, code } = describeNonJsonFailure(response, uploadLimitMb);
  throw new ApiError(message, { status: response.status, code, hint, data });
};

// fetch() rejects with a TypeError when the connection fails or is closed without a response.
export const networkError = (error, { uploading = false, uploadLimitMb } = {}) => {
  if (!(error instanceof TypeError)) return error;
  if (uploading) {
    return new ApiError('The connection was closed while the file was being uploaded, without an answer from RapCluster.', {
      code: 'upload_connection_closed',
      hint: `RapCluster accepts files up to ${uploadLimitMb ?? 50} MB. A proxy or gateway between you and RapCluster may have a lower upload limit or dropped the connection; try a smaller file or another network.`,
    });
  }
  return new ApiError('Could not reach the RapCluster server.', {
    code: 'network_error',
    hint: 'Check your internet connection and try again.',
  });
};

export const fetchJson = async (url, options = {}, context = {}) => {
  let response;
  try {
    response = await fetch(url, options);
  } catch (e) {
    throw networkError(e, context);
  }
  return readJson(response, context);
};

// "message — hint" for display.
export const formatApiError = (error) => (
  error && error.hint ? `${error.message} ${error.hint}` : (error?.message || String(error))
);

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Polls a job status link until the job is no longer running. Short network hiccups are retried.
export const pollJob = async (apiBase, statusLink, { kind = 'cluster', intervalMs = 1500, onProgress, isCancelled } = {}) => {
  let failures = 0;
  for (;;) {
    if (isCancelled?.()) return null;
    try {
      const status = await fetchJson(`${apiBase}${statusLink}?kind=${kind}`);
      failures = 0;
      if (status.state !== 'running') return status;
      onProgress?.(status);
    } catch (e) {
      failures += 1;
      if (failures >= 5 || (e instanceof ApiError && e.status && e.status < 500)) throw e;
    }
    await sleep(intervalMs);
  }
};
