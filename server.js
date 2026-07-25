// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased payload limit for large context requests

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output with <think> tags
const SHOW_REASONING = false; 

// 🔥 DEFAULT THINKING MODE TOGGLE (if not provided by client)
const ENABLE_THINKING_MODE = false; 

// High-context Model mapping
const MODEL_MAPPING = {
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
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
      chat: '/v1/chat/completions'
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
    api_key_configured: Boolean(NIM_API_KEY)
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
      extra_body,
      seed
    } = req.body;

    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY is not set in Railway environment variables.',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model] || model;
    
    // Default High Max Tokens (Default 131,072 for high context support, unless client explicitly requests lower/higher)
    const MAX_ALLOWED_TOKENS = 131072; // 128k tokens
    const selectedMaxTokens = max_tokens ? Math.min(max_tokens, MAX_ALLOWED_TOKENS) : 131072;

    // Handle extra_body for thinking parameters
    // Priority: client-supplied extra_body > per-model max-reasoning default > global toggle
    const finalExtraBody =
      extra_body ||
      MAX_REASONING_BY_MODEL[nimModel] ||
      (ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined);

    // Transform OpenAI request to NVIDIA NIM format
    const nimRequest = {
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

    if (finalExtraBody) {
      nimRequest.extra_body = finalExtraBody;
    }

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

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
                const reasoning = data.choices[0].delta.reasoning_content;
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
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
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

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
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
    console.error('Proxy error:', error.response?.data || error.message);

    const statusCode = error.response?.status || 500;
    const errorDetails = error.response?.data?.error || error.response?.data || {
      message: error.message || 'Internal server error',
      type: 'proxy_error',
      code: statusCode
    };

    res.status(statusCode).json({ error: errorDetails });
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
