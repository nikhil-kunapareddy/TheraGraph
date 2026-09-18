import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api } from '@/lib/api';

function mockFetch(response: {
  ok?: boolean;
  status?: number;
  json?: unknown;
  headers?: Record<string, string>;
}) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: response.ok ?? true,
    status: response.status ?? 200,
    statusText: 'OK',
    headers: new Headers(response.headers ?? {}),
    json: async () => response.json ?? {},
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** The arguments of a recorded fetch call, with "did it get called?" asserted once. */
function callArgs(fetchMock: ReturnType<typeof mockFetch>, index = 0) {
  const call = fetchMock.mock.calls[index];
  if (!call) throw new Error(`fetch was not called ${index + 1} time(s)`);
  return { url: call[0] as string, init: call[1] as RequestInit | undefined };
}

describe('api client', () => {
  beforeEach(() => vi.unstubAllGlobals());

  it('builds query strings and drops empty parameters', async () => {
    const fetchMock = mockFetch({ json: { hits: [] } });
    await api.search('alzheimer', { limit: 5, entityTypes: ['disease', 'target'] });
    const { url } = callArgs(fetchMock);
    expect(url).toContain('q=alzheimer');
    expect(url).toContain('limit=5');
    // Repeated parameters, not a comma-joined string, so FastAPI parses a list.
    expect(url.match(/entity_types=/g)).toHaveLength(2);
  });

  it('unwraps the error envelope into a typed error', async () => {
    mockFetch({
      ok: false,
      status: 404,
      json: { error: { code: 'not_found', message: "target 'NOPE' is not in the local corpus" } },
      headers: { 'X-Request-ID': 'req_abc' },
    });
    await expect(api.target('NOPE')).rejects.toThrowError(ApiClientError);
    try {
      await api.target('NOPE');
    } catch (error) {
      const typed = error as ApiClientError;
      expect(typed.code).toBe('not_found');
      expect(typed.isNotFound).toBe(true);
      expect(typed.requestId).toBe('req_abc');
      // The hint is chosen by code, so the UI can act on it.
      expect(typed.hint).toContain('ingested corpus');
    }
  });

  it('marks retryable errors as retryable', async () => {
    mockFetch({
      ok: false,
      status: 503,
      json: {
        error: { code: 'upstream_unavailable', message: 'ChEMBL unreachable', retryable: true },
      },
    });
    try {
      await api.corpus();
    } catch (error) {
      expect((error as ApiClientError).retryable).toBe(true);
    }
  });

  it('survives a non-JSON error body', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      headers: new Headers(),
      json: async () => {
        throw new Error('not json');
      },
    });
    vi.stubGlobal('fetch', fetchMock);
    try {
      await api.corpus();
    } catch (error) {
      const typed = error as ApiClientError;
      expect(typed.status).toBe(502);
      expect(typed.retryable).toBe(true);
    }
  });

  it('sends JSON bodies with the right content type', async () => {
    const fetchMock = mockFetch({ json: { hits: [] } });
    await api.similarity({ smiles: 'CCO', threshold: 0.7 });
    const init = callArgs(fetchMock).init as RequestInit;
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body as string)).toEqual({ smiles: 'CCO', threshold: 0.7 });
  });

  it('encodes identifiers that need it', async () => {
    const fetchMock = mockFetch({ json: {} });
    await api.compound('CHEMBL25/weird id');
    expect(callArgs(fetchMock).url).toContain('CHEMBL25%2Fweird%20id');
  });
});
