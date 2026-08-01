// server.js - OpenAI to NVIDIA NIM API Proxy (dual-key load balanced)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

// Safety net: log instead of letting the whole process die on an unexpected
// throw/rejection somewhere in the codebase (e.g. a future serialization bug).
// The real fix for known cases should still happen at the source — this is
// just a backstop so one bad request can't take down every other request.
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased payload limit for large context requests

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output with <think> tags
const SHOW_REASONING = true; // TEMP: debugging whether reasoning_content is actually returned

// 🔥 DEFAULT THINKING MODE TOGGLE (if not provided by client)
const ENABLE_THINKING_MODE = false;

// ---------------------------------------------------------------------------
// 🔀 MULTI-KEY LOAD BALANCER
// ---------------------------------------------------------------------------
// Any number of devs/accounts, each capped at ~40 requests/minute by NVIDIA.
// All keys are pooled behind this one proxy and every request is routed to
// whichever key currently has the most headroom left in a rolling 60s
// window ("least loaded"), not blind round robin — so one slow/heavy burst
// on one key doesn't leave the others idle, and none of them gets pushed
// past its own limit.
//
// If ALL keys are momentarily maxed out, requests don't just fail — they
// wait (checking again as old requests roll out of the 60s window) up to
// NIM_QUEUE_TIMEOUT_MS, then return a proper 429 with Retry-After if nothing
// freed up in time. That's the "juggle continuously without crashing"
// behavior: smooth over short bursts, degrade gracefully under sustained
// overload instead of hammering NVIDIA or hanging forever.
//
// Set these in Railway → your service → Variables. Add one indexed variable
// per key — any count works, just keep numbering sequentially:
//   NIM_API_KEY_1          - first key  (required)
//   NIM_API_KEY_2          - second key
//   NIM_API_KEY_3          - third key, and so on as more people join
//   NIM_RPM_LIMIT_PER_KEY  - requests/minute allowed per key (default 40)
//   NIM_QUEUE_TIMEOUT_MS   - max wait for a free slot before replying 429 (default 30000)
//
// Back-compat: if you still have the old single NIM_API_KEY var and no
// NIM_API_KEY_1/2/3..., it's used automatically as the only key.
// ---------------------------------------------------------------------------

const RPM_LIMIT_PER_KEY = parseInt(process.env.NIM_RPM_LIMIT_PER_KEY || '40', 10);
const QUEUE_TIMEOUT_MS = parseInt(process.env.NIM_QUEUE_TIMEOUT_MS || '30000', 10);

// Scans for NIM_API_KEY_1, NIM_API_KEY_2, NIM_API_KEY_3, ... (any count, any
// order) so adding another dev's key is just "add a Railway variable" —
// no code changes needed. Also accepts the old single NIM_API_KEY var, and
// an optional comma-separated NIM_API_KEYS for convenience.
function loadApiKeys() {
  const keys = [];

  const indexed = Object.keys(process.env)
    .filter((k) => /^NIM_API_KEY_\d+$/.test(k))
    .sort((a, b) => parseInt(a.split('_').pop(), 10) - parseInt(b.split('_').pop(), 10))
    .map((k) => process.env[k]);
  keys.push(...indexed);

  if (process.env.NIM_API_KEYS) {
    keys.push(...process.env.NIM_API_KEYS.split(',').map((s) => s.trim()));
  }

  if (indexed.length === 0 && process.env.NIM_API_KEY) {
    keys.push(process.env.NIM_API_KEY); // back-compat, single-key era
  }

  return [...new Set(keys.filter(Boolean))];
}

const RAW_KEYS = loadApiKeys();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class KeyPool {
  constructor(keys, limitPerMinute) {
    this.limit = limitPerMinute;
    this.windowMs = 60 * 1000;
    this.keys = keys.map((key, idx) => ({
      id: idx + 1,
      key,
      label: `key_${idx + 1}`,
      timestamps: [] // start-of-request timestamps within the trailing 60s window
    }));
  }

  _prune(keyState, now) {
    const cutoff = now - this.windowMs;
    while (keyState.timestamps.length && keyState.timestamps[0] <= cutoff) {
      keyState.timestamps.shift();
    }
  }

  // Picks the key with the most headroom left in the current 60s window.
  _bestOption(now) {
    let best = null;
    for (const keyState of this.keys) {
      this._prune(keyState, now);
      const used = keyState.timestamps.length;
      const headroom = this.limit - used;
      if (!best || headroom > best.headroom) {
        best = { keyState, headroom, used };
      }
    }
    return best;
  }

  // Reserves a slot on the least-loaded key. If every key is saturated,
  // waits (re-checking as slots roll off the window) instead of failing
  // immediately, up to timeoutMs. Throws RATE_LIMIT_TIMEOUT if nothing frees
  // up in time, so a request can never hang forever.
  async acquire(timeoutMs = QUEUE_TIMEOUT_MS) {
    if (this.keys.length === 0) {
      const err = new Error('No NIM API keys configured.');
      err.code = 'NO_KEYS';
      throw err;
    }

    const deadline = Date.now() + timeoutMs;

    while (true) {
      const now = Date.now();
      const best = this._bestOption(now);

      if (best.headroom > 0) {
        best.keyState.timestamps.push(now);
        return best.keyState;
      }

      if (now >= deadline) {
        const err = new Error('All API keys are at their per-minute rate limit; timed out waiting for a free slot.');
        err.code = 'RATE_LIMIT_TIMEOUT';
        throw err;
      }

      // Every key is full — figure out when the oldest reservation on any
      // key falls out of the 60s window and sleep roughly until then.
      let earliest = Infinity;
      for (const keyState of this.keys) {
        this._prune(keyState, now);
        if (keyState.timestamps.length) {
          earliest = Math.min(earliest, keyState.timestamps[0]);
        }
      }
      const rawWait = (earliest + this.windowMs) - now + 25;
      const waitMs = Math.min(Math.max(rawWait, 50), 5000, Math.max(deadline - now, 0));
      await sleep(waitMs);
    }
  }

  // Called when NIM itself returns a 429 for a key we thought had headroom —
  // meaning real usage on that key is higher than our local count (e.g.
  // something outside this proxy is also drawing on it). Pad the window so
  // we stop routing to it until it naturally rolls off, instead of
  // repeatedly re-discovering the same 429.
  markSaturated(keyState) {
    const now = Date.now();
    while (keyState.timestamps.length < this.limit) {
      keyState.timestamps.push(now);
    }
  }

  stats() {
    const now = Date.now();
    return this.keys.map((keyState) => {
      this._prune(keyState, now);
      const used = keyState.timestamps.length;
      const oldest = keyState.timestamps[0];
      return {
        label: keyState.label,
        used,
        limit: this.limit,
        remaining: Math.max(this.limit - used, 0),
        resets_in_ms: oldest ? Math.max((oldest + this.windowMs) - now, 0) : 0
      };
    });
  }
}

const keyPool = new KeyPool(RAW_KEYS, RPM_LIMIT_PER_KEY);

if (RAW_KEYS.length === 0) {
  console.warn('⚠️  No NIM API keys configured (set NIM_API_KEY_1, NIM_API_KEY_2, ... in Railway). Requests will fail until set.');
} else {
  console.log(`✅ Loaded ${RAW_KEYS.length} NIM API key(s) · ${RPM_LIMIT_PER_KEY} req/min each · queue timeout ${QUEUE_TIMEOUT_MS}ms`);
}

// High-context Model mapping
const MODEL_MAPPING = {
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimaxai/minimax-m3': 'minimaxai/minimax-m3'
};

// 🔥 Per-model "max reasoning" defaults.
// Each of these models exposes thinking through a different chat_template_kwargs
// shape on NIM, so they can't share one generic { thinking: true } flag.
// Applied automatically unless the client sends its own extra_body.
const MAX_REASONING_BY_MODEL = {
  'z-ai/glm-5.2': { chat_template_kwargs: { reasoning_effort: 'max' } },
  'deepseek-ai/deepseek-v4-pro': { chat_template_kwargs: { thinking: true, reasoning_effort: 'max' } },
  'deepseek-ai/deepseek-v4-flash': { chat_template_kwargs: { thinking: true, reasoning_effort: 'high' } },
  'minimaxai/minimax-m3': { chat_template_kwargs: { thinking_mode: 'enabled' } }
};

// 1. Root Endpoint (Fixes Railway "/" 404 Error)
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    service: 'OpenAI to NVIDIA NIM Proxy',
    message: 'Proxy is running successfully on Railway!',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions',
      proxy_stats: '/v1/proxy-stats'
    }
  });
});

// 2. Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    keys_configured: RAW_KEYS.length,
    rpm_limit_per_key: RPM_LIMIT_PER_KEY,
    key_usage: keyPool.stats()
  });
});

// 2b. Load-balancer stats endpoint (usage per key, no key values exposed)
app.get('/v1/proxy-stats', (req, res) => {
  res.json({
    rpm_limit_per_key: RPM_LIMIT_PER_KEY,
    queue_timeout_ms: QUEUE_TIMEOUT_MS,
    keys: keyPool.stats()
  });
});

// 3. List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// 4. Chat completions endpoint (Main Proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      model,
      messages,
      temperature,
      top_p,
      max_tokens,
      stream,
      extra_body,       // legacy/back-compat: some SDKs nest reasoning config here
      chat_template_kwargs, // NIM's actual native field — sent directly by well-behaved clients
      seed,
      ...rest           // pass through anything else the client sends (tools, response_format, etc.)
    } = req.body;

    if (keyPool.keys.length === 0) {
      return res.status(500).json({
        error: {
          message: 'No NIM API key configured. Set NIM_API_KEY_1 (and NIM_API_KEY_2, NIM_API_KEY_3, ... for more) in Railway environment variables.',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model] || model;

    // MAX_ALLOWED_TOKENS is an upper bound only (1,000,000), for clients that explicitly
    // want to push near a model's real limit. DEFAULT_MAX_TOKENS is what's used when the
    // client omits max_tokens entirely — kept at 131072 since that's GLM-5.2's confirmed
    // real output cap on NVIDIA's hosted endpoint; defaulting the omitted case to 1,000,000
    // caused NIM to silently return an empty {choices: [], usage: zeros} response instead
    // of a clean error. Other models may have different real caps — if a specific explicit
    // max_tokens value causes the same empty-response symptom, that model's real ceiling is
    // lower than requested.
    const MAX_ALLOWED_TOKENS = 1000000;
    const DEFAULT_MAX_TOKENS = 131072;
    const selectedMaxTokens = max_tokens ? Math.min(max_tokens, MAX_ALLOWED_TOKENS) : DEFAULT_MAX_TOKENS;

    // Determine reasoning config for this request.
    // Priority: client's own top-level chat_template_kwargs > legacy extra_body.chat_template_kwargs
    //           > per-model max-reasoning default > global ENABLE_THINKING_MODE toggle.
    // IMPORTANT: NIM's real API takes chat_template_kwargs at the TOP LEVEL of the request body.
    // "extra_body" is only a client-SDK-side convention (OpenAI/Together SDKs flatten it before
    // sending); NIM itself rejects a literal "extra_body" field with a 400 Validation error.
    const finalChatTemplateKwargs =
      chat_template_kwargs ||
      extra_body?.chat_template_kwargs ||
      MAX_REASONING_BY_MODEL[nimModel]?.chat_template_kwargs ||
      (ENABLE_THINKING_MODE ? { thinking: true } : undefined);

    // Transform OpenAI request to NVIDIA NIM format.
    // Spread ...rest first so any other NIM-supported fields (tools, tool_choice,
    // response_format, etc.) pass straight through, then set the normalized fields on top.
    const nimRequest = {
      ...rest,
      model: nimModel,
      messages: messages,
      temperature: temperature !== undefined ? temperature : 0.7,
      top_p: top_p !== undefined ? top_p : 0.95,
      max_tokens: selectedMaxTokens,
      stream: stream || false
    };

    // Forward seed for reproducible generations (e.g. seed=42 from client scripts)
    if (seed !== undefined) {
      nimRequest.seed = seed;
    }

    if (finalChatTemplateKwargs) {
      nimRequest.chat_template_kwargs = finalChatTemplateKwargs;
    }

    // 🔀 Acquire the least-loaded key and make the request. If a key comes
    // back 429/5xx, mark it and retry on the other key (bounded by how many
    // keys are configured) before giving up. Non-retryable errors (bad
    // request, auth, etc.) bubble straight out.
    const attempts = Math.max(keyPool.keys.length, 1);
    let response;
    let lastError;

    for (let i = 0; i < attempts; i++) {
      let keyState;
      try {
        keyState = await keyPool.acquire();
      } catch (acquireErr) {
        if (acquireErr.code === 'RATE_LIMIT_TIMEOUT') {
          res.setHeader('Retry-After', '5');
          return res.status(429).json({
            error: {
              message: acquireErr.message,
              type: 'rate_limit_error',
              code: 429
            }
          });
        }
        throw acquireErr;
      }

      try {
        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${keyState.key}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json'
        });
        lastError = null;
        break; // success — stop trying further keys
      } catch (error) {
        lastError = error;
        const status = error.response?.status;

        if (status === 429) {
          console.warn(`⚠️  ${keyState.label} returned 429 from NIM — marking saturated, trying next key if available.`);
          keyPool.markSaturated(keyState);
          continue;
        }
        if (status >= 500 && status < 600) {
          console.warn(`⚠️  ${keyState.label} returned ${status} from NIM — trying next key if available.`);
          continue;
        }
        throw error; // not retryable — handled by outer catch below
      }
    }

    if (!response) {
      throw lastError || new Error('All configured NIM API keys failed.');
    }

    if (stream) {
      // Handle streaming response with reasoning option
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content ?? data.choices[0].delta.reasoning;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                    delete data.choices[0].delta.reasoning;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

    } else {
      // Transform NIM non-streaming response to OpenAI standard format
      const openaiResponse = {
        id: response.data.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: response.data.created || Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map(choice => {
          let fullContent = choice.message?.content || '';

          const reasoningText = choice.message?.reasoning_content ?? choice.message?.reasoning;
          if (SHOW_REASONING && reasoningText) {
            fullContent = '<think>\n' + reasoningText + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index || 0,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    // When responseType was 'stream' (client sent stream: true), axios puts
    // the raw Node HTTP stream on error.response.data instead of parsed
    // JSON. That object is circular (socket -> parser -> incoming -> ...),
    // so it must NEVER be handed to res.json()/JSON.stringify — doing so
    // throws "Converting circular structure to JSON", which (since this is
    // inside an async handler with no outer .catch) becomes an unhandled
    // rejection and kills the whole process, not just this one request.
    const rawData = error.response?.data;
    const isStream = rawData && typeof rawData.pipe === 'function';

    console.error('Proxy error:', isStream ? error.message : (rawData || error.message));

    const statusCode = error.response?.status || 500;
    const errorDetails = (!isStream && (rawData?.error || rawData)) || {
      message: error.message || 'Internal server error',
      type: 'proxy_error',
      code: statusCode
    };

    if (!res.headersSent) {
      res.status(statusCode).json({ error: errorDetails });
    }
  }
});

// Catch-all for non-existing routes
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Use /v1/chat/completions for completions.`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
