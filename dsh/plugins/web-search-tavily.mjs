/**
 * bubble-watch: Tavily-backed `WebSearchProvider` for DeepSeek Harness, which ships DeepSeek, Exa and
 * Perplexity search backends but no Tavily one. Modeled on `@deepseek-ai/dsh-web-search-exa`, and kept
 * dependency-free so dsh can load it by absolute path from a patch layer (see dsh/web-search.patch.yml).
 *
 * Registers under provider id `tavily`; select it with the web service's `searchProvider: tavily`.
 */

export const name = 'web-search-tavily'

/** The web seam this provider registers into. */
export const inject = ['web']

export const TAVILY_PROVIDER_ID = 'tavily'
export const TAVILY_DEFAULT_BASE_URL = 'https://api.tavily.com'

/** Same shape dsh's `WebError` carries (name + code), so tool results read like the built-in backends'. */
class WebError extends Error {
  constructor(message, code, options) {
    super(message, options)
    this.name = 'WebError'
    this.code = code
  }
}

const isAbortError = error => error instanceof DOMException && error.name === 'AbortError'

/**
 * Map a Tavily `/search` response to dsh's normalized search result. An entry without non-blank
 * `content` has no portable snippet and is dropped (inventing one would lie). Tavily's generated
 * answer is not requested, so `content` is omitted; the web service owns `maxResults` truncation.
 */
export function mapTavilyResponse(payload) {
  const sources = (payload?.results ?? [])
    .filter(r => r && typeof r.url === 'string' && typeof r.content === 'string' && r.content.trim().length > 0)
    .map(r => ({
      url: r.url,
      ...(typeof r.title === 'string' && r.title.length > 0 ? { title: r.title } : {}),
      snippet: r.content.trim(),
      ...(typeof r.published_date === 'string' && r.published_date.length > 0 ? { publishedAt: r.published_date } : {}),
    }))
  return { sources, truncated: false }
}

export class TavilySearchProvider {
  id = TAVILY_PROVIDER_ID

  constructor(options) {
    this.options = options
  }

  available() {
    return this.options.apiKey.length > 0 && URL.canParse(this.options.baseURL)
  }

  async search(request, signal) {
    let response
    try {
      response = await fetch(`${this.options.baseURL}/search`, {
        method: 'POST',
        redirect: 'error',
        headers: {
          authorization: `Bearer ${this.options.apiKey}`,
          'content-type': 'application/json',
          accept: 'application/json',
        },
        body: JSON.stringify({
          query: request.query,
          max_results: request.maxResults ?? this.options.maxResults,
          search_depth: this.options.searchDepth,
          topic: this.options.topic,
          include_answer: false,
        }),
        ...(signal !== undefined ? { signal } : {}),
      })
    } catch (error) {
      if (isAbortError(error)) throw new WebError('Tavily search aborted', 'WEB_ABORTED', { cause: error })
      throw new WebError(`Tavily search request failed: ${String(error)}`, 'WEB_PROVIDER_ERROR', { cause: error })
    }

    if (!response.ok) {
      let message = `Tavily API error (HTTP ${response.status})`
      try {
        const body = await response.json()
        const detail = body?.detail?.error ?? body?.detail ?? body?.error ?? body?.message
        if (typeof detail === 'string' && detail.length > 0) message = detail
      } catch (error) {
        if (isAbortError(error)) throw new WebError('Tavily search aborted', 'WEB_ABORTED', { cause: error })
        // A non-JSON error body (common for gateway 5xx/429) only costs the richer message.
      }
      throw new WebError(message, 'WEB_PROVIDER_ERROR')
    }

    try {
      return mapTavilyResponse(await response.json())
    } catch (error) {
      if (isAbortError(error)) throw new WebError('Tavily search aborted', 'WEB_ABORTED', { cause: error })
      throw new WebError(`Tavily returned an unprocessable response body: ${String(error)}`, 'WEB_PROVIDER_ERROR', { cause: error })
    }
  }
}

/** Register the Tavily search provider with `ctx.web`. The API key falls back to `$TAVILY_API_KEY`. */
export function apply(ctx, config = {}) {
  ctx.web.registerSearchProvider(new TavilySearchProvider({
    apiKey: config.apiKey ?? process.env.TAVILY_API_KEY ?? '',
    baseURL: config.baseURL ?? TAVILY_DEFAULT_BASE_URL,
    searchDepth: config.searchDepth ?? 'basic',
    topic: config.topic ?? 'general',
    maxResults: config.maxResults ?? 5,
  }))
}
