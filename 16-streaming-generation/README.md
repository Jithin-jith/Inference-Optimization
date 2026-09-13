# Module 16: Streaming Generation (Perceived Latency Reduction)

## 1. Technical Explanation & Mathematical Intuition

In non-streaming HTTP REST API endpoints, the client sends a prompt and waits passively while the server generates the **entire response** (e.g. 500 tokens).

If the server generates tokens at 30 tokens/sec, the user waits:
$$\text{Total User Wait Time} = \text{TTFT} + \frac{500 \text{ tokens}}{30 \text{ tokens/sec}} \approx 1.5 \text{s} + 16.6 \text{s} = 18.1 \text{ seconds!}$$

**Streaming Generation** utilizes **Server-Sent Events (SSE)** or WebSockets to transmit generated tokens to the user interface token-by-token, as soon as each individual token is sampled from the softmax distribution.

### Perceived Latency (TTFT) Improvement
$$\text{Perceived User Wait Time} = \text{TTFT} \approx 100\text{ms} - 300\text{ms}$$
While total execution time remains identical, perceived responsiveness improves by **90% to 98%**, creating an instantaneous human-like typing experience.

---

## 2. Architecture & Execution Flow

```
Non-Streaming HTTP Request:
Client ---> Prompt ---> Server ---> Generate Token 1..500 ---> [Wait 18 Seconds] ---> Send Complete Payload

Streaming HTTP Response (Server-Sent Events / Chunked Transfer):
Client ---> Prompt ---> Server
                         |---> Token 1 -> Streamed over SSE -> Rendered on Screen (Time: 150ms!)
                         |---> Token 2 -> Streamed over SSE -> Rendered on Screen
                         |---> Token 3 -> Streamed over SSE -> Rendered on Screen
                         ...
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Perceived TTFT** | Cuts user wait time from ~15s to ~200ms | Maintains persistent HTTP connection per active client |
| **UX Responsiveness** | Creates smooth, real-time typing interface | SSE stream handling required in web frontend (JavaScript EventSource) |
| **User Drop-off** | Reduces user bounce rate by >80% | Middleware proxies (like NGINX) must disable buffering |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (Ollama / vLLM / FastAPI StreamingResponse)**:
  - **Ollama**: Python API supports `ollama.chat(stream=True)`.
  - **vLLM / FastAPI**: Serves OpenAI-compatible `/v1/chat/completions` with `"stream": true` yielding SSE `data: {...}` lines.
- **Proprietary (Google Gemini Streaming SDK)**:
  - Gemini provides `client.models.generate_content_stream(...)` which streams `GenerateContentResponse` chunks in real-time.

---

## 5. AWS Deep Dive

- **Amazon Bedrock Streaming API**: Bedrock provides `InvokeModelWithResponseStream` API action, emitting `ResponseStream` event chunks over AWS SDK.
- **AWS API Gateway**: Supports HTTP Streaming & WebSockets. Set `ResponseBuffering` to disabled to prevent API Gateway from buffering streamed tokens.
- **SageMaker Streaming Endpoints**: SageMaker supports response streaming via `InvokeEndpointWithResponseStream`.
