package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// OpenAICompatExecutor implements a stateless executor for OpenAI-compatible providers.
// It performs request/response translation and executes against the provider base URL
// using per-auth credentials (API key) and per-auth HTTP transport (proxy) from context.
type OpenAICompatExecutor struct {
	provider string
	cfg      *config.Config
}

// copilotResponsesOnlyModels lists Copilot models that only support the /responses
// endpoint (not /chat/completions). These are newer OpenAI reasoning models.
var copilotResponsesOnlyModels = map[string]bool{
	"gpt-5.5":       true,
	"gpt-5.4-mini":  true,
	"gpt-5.3-codex": true,
}

// isCopilotResponsesModel returns true if the model requires the /responses endpoint
// when used through GitHub Copilot.
func (e *OpenAICompatExecutor) isCopilotResponsesModel(model string) bool {
	if e.provider != "github-copilot" {
		return false
	}
	// Strip copilot- prefix if present
	bare := strings.TrimPrefix(model, "copilot-")
	return copilotResponsesOnlyModels[bare]
}

// convertChatCompletionsToResponses converts an OpenAI Chat Completions request
// into an OpenAI Responses API request. This is used for Copilot models that
// only support /responses (gpt-5.5, gpt-5.4-mini, gpt-5.3-codex).
func convertChatCompletionsToResponses(payload []byte, stream bool) []byte {
	root := gjson.ParseBytes(payload)

	out := []byte(`{}`)
	// Model
	if v := root.Get("model"); v.Exists() {
		out, _ = sjson.SetBytes(out, "model", v.String())
	}
	// Stream
	out, _ = sjson.SetBytes(out, "stream", stream)

	// max_tokens / max_completion_tokens → max_output_tokens
	if v := root.Get("max_completion_tokens"); v.Exists() {
		out, _ = sjson.SetBytes(out, "max_output_tokens", v.Int())
	} else if v := root.Get("max_tokens"); v.Exists() {
		out, _ = sjson.SetBytes(out, "max_output_tokens", v.Int())
	}

	// Build input array from messages
	var input []any
	var instructions string
	messages := root.Get("messages")
	messages.ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		switch role {
		case "system":
			// System messages become instructions
			content := msg.Get("content").String()
			if instructions != "" {
				instructions += "\n\n"
			}
			instructions += content
		case "user":
			content := msg.Get("content")
			if content.IsArray() {
				// Multi-part content: extract text parts
				var textParts []string
				content.ForEach(func(_, part gjson.Result) bool {
					if part.Get("type").String() == "text" {
						textParts = append(textParts, part.Get("text").String())
					}
					return true
				})
				input = append(input, map[string]any{
					"role":    "user",
					"content": strings.Join(textParts, "\n"),
					"type":    "message",
				})
			} else {
				input = append(input, map[string]any{
					"role":    "user",
					"content": content.String(),
					"type":    "message",
				})
			}
		case "assistant":
			content := msg.Get("content")
			if content.IsArray() {
				// Tool calls in assistant message
				var textContent string
				var toolCalls []any
				content.ForEach(func(_, part gjson.Result) bool {
					switch part.Get("type").String() {
					case "text":
						textContent = part.Get("text").String()
					}
					return true
				})
				if textContent != "" {
					input = append(input, map[string]any{
						"role":    "assistant",
						"content": textContent,
						"type":    "message",
					})
				}
				_ = toolCalls
			} else {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": content.String(),
					"type":    "message",
				})
			}
			// Handle tool_calls
			if tc := msg.Get("tool_calls"); tc.Exists() && tc.IsArray() {
				tc.ForEach(func(_, call gjson.Result) bool {
					input = append(input, map[string]any{
						"type":      "function_call",
						"call_id":   call.Get("id").String(),
						"name":      call.Get("function.name").String(),
						"arguments": call.Get("function.arguments").String(),
					})
					return true
				})
			}
		case "tool":
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": msg.Get("tool_call_id").String(),
				"output":  msg.Get("content").String(),
			})
		}
		return true
	})

	if instructions != "" {
		out, _ = sjson.SetBytes(out, "instructions", instructions)
	}
	if len(input) > 0 {
		out, _ = sjson.SetBytes(out, "input", input)
	} else {
		out, _ = sjson.SetRawBytes(out, "input", []byte("[]"))
	}

	// Tools: convert from chat completions format to responses format
	if tools := root.Get("tools"); tools.Exists() && tools.IsArray() {
		var responsesTools []any
		tools.ForEach(func(_, tool gjson.Result) bool {
			if tool.Get("type").String() == "function" {
				fn := tool.Get("function")
				t := map[string]any{
					"type":        "function",
					"name":        fn.Get("name").String(),
					"description": fn.Get("description").String(),
				}
				if params := fn.Get("parameters"); params.Exists() {
					t["parameters"] = json.RawMessage(params.Raw)
				}
				responsesTools = append(responsesTools, t)
			}
			return true
		})
		if len(responsesTools) > 0 {
			out, _ = sjson.SetBytes(out, "tools", responsesTools)
		}
	}

	// Tool choice
	if tc := root.Get("tool_choice"); tc.Exists() {
		out, _ = sjson.SetRawBytes(out, "tool_choice", []byte(tc.Raw))
	}

	// Temperature
	if v := root.Get("temperature"); v.Exists() {
		out, _ = sjson.SetBytes(out, "temperature", v.Float())
	}

	// Top-p
	if v := root.Get("top_p"); v.Exists() {
		out, _ = sjson.SetBytes(out, "top_p", v.Float())
	}

	return out
}

// NewOpenAICompatExecutor creates an executor bound to a provider key (e.g., "openrouter").
func NewOpenAICompatExecutor(provider string, cfg *config.Config) *OpenAICompatExecutor {
	return &OpenAICompatExecutor{provider: provider, cfg: cfg}
}

// Identifier implements cliproxyauth.ProviderExecutor.
func (e *OpenAICompatExecutor) Identifier() string { return e.provider }

// PrepareRequest injects OpenAI-compatible credentials into the outgoing HTTP request.
func (e *OpenAICompatExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	_, apiKey := e.resolveCredentials(auth)
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects OpenAI-compatible credentials into the request and executes it.
func (e *OpenAICompatExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("openai compat executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

func (e *OpenAICompatExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName


	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "missing provider baseURL"}
		return
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	endpoint := "/chat/completions"
	useResponsesAPI := false
	if opts.Alt == "responses/compact" {
		to = sdktranslator.FromString("openai-response")
		endpoint = "/responses/compact"
	} else if e.isCopilotResponsesModel(baseModel) {
		useResponsesAPI = true
		endpoint = "/responses"
	}
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, opts.Stream)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, opts.Stream)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel, requestPath)
	if opts.Alt == "responses/compact" {
		if updated, errDelete := sjson.DeleteBytes(translated, "stream"); errDelete == nil {
			translated = updated
		}
	}

	// Copilot GPT models require max_completion_tokens instead of max_tokens
	if e.provider == "github-copilot" && !useResponsesAPI {
		if v := gjson.GetBytes(translated, "max_tokens"); v.Exists() {
			translated, _ = sjson.SetBytes(translated, "max_completion_tokens", v.Value())
			translated, _ = sjson.DeleteBytes(translated, "max_tokens")
		}
		translated, _ = sjson.DeleteBytes(translated, "reasoning_effort")
	}

	// For /responses-only models: convert the chat completions payload to Responses API format
	if useResponsesAPI {
		translated = convertChatCompletionsToResponses(translated, opts.Stream)
	}

	url := strings.TrimSuffix(baseURL, "/") + endpoint
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	httpReq.Header.Set("User-Agent", "cli-proxy-openai-compat")
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("openai compat executor: close response body error: %v", errClose)
		}
	}()
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, body)
	reporter.Publish(ctx, helps.ParseOpenAIUsage(body))
	// Ensure we at least record the request even if upstream doesn't return usage
	reporter.EnsurePublished(ctx)
	// Translate response back to source format when needed
	var param any
	// For /responses models, convert the Responses API response to Chat Completions format
	// so the existing openai→claude translator can handle it.
	responseBody := body
	if useResponsesAPI {
		responseBody = convertResponsesResponseToChatCompletions(body)
	}
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, responseBody, &param)
	resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
	return resp, nil
}

func (e *OpenAICompatExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName


	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "missing provider baseURL"}
		return nil, err
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	endpoint := "/chat/completions"
	useResponsesAPI := e.isCopilotResponsesModel(baseModel)
	if useResponsesAPI {
		endpoint = "/responses"
	}
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, true)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel, requestPath)

	// Request usage data in the final streaming chunk so that token statistics
	// are captured even when the upstream is an OpenAI-compatible provider.
	if !useResponsesAPI {
		translated, _ = sjson.SetBytes(translated, "stream_options.include_usage", true)
	}

	// Copilot GPT models require max_completion_tokens instead of max_tokens
	if e.provider == "github-copilot" && !useResponsesAPI {
		if v := gjson.GetBytes(translated, "max_tokens"); v.Exists() {
			translated, _ = sjson.SetBytes(translated, "max_completion_tokens", v.Value())
			translated, _ = sjson.DeleteBytes(translated, "max_tokens")
		}
		// Copilot /chat/completions doesn't support reasoning_effort with tools
		translated, _ = sjson.DeleteBytes(translated, "reasoning_effort")
	}

	// For /responses endpoint, ensure stream is set and remove stream_options
	if useResponsesAPI {
		translated = convertChatCompletionsToResponses(translated, true)
	}

	url := strings.TrimSuffix(baseURL, "/") + endpoint
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	httpReq.Header.Set("User-Agent", "cli-proxy-openai-compat")
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("openai compat executor: close response body error: %v", errClose)
		}
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("openai compat executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := helps.ParseOpenAIStreamUsage(line); ok {
				reporter.Publish(ctx, detail)
			}
			trimmedLine := bytes.TrimSpace(line)
			if len(trimmedLine) == 0 {
				continue
			}

			if !bytes.HasPrefix(trimmedLine, []byte("data:")) {
				if bytes.HasPrefix(trimmedLine, []byte(":")) || bytes.HasPrefix(trimmedLine, []byte("event:")) ||
					bytes.HasPrefix(trimmedLine, []byte("id:")) || bytes.HasPrefix(trimmedLine, []byte("retry:")) {
					continue
				}
				if bytes.HasPrefix(trimmedLine, []byte("{")) || bytes.HasPrefix(trimmedLine, []byte("[")) {
					streamErr := statusErr{code: http.StatusBadGateway, msg: string(trimmedLine)}
					helps.RecordAPIResponseError(ctx, e.cfg, streamErr)
					reporter.PublishFailure(ctx, streamErr)
					select {
					case out <- cliproxyexecutor.StreamChunk{Err: streamErr}:
					case <-ctx.Done():
					}
					return
				}
				continue
			}

			// For /responses API, convert SSE lines to chat completions format first
			processLine := bytes.Clone(trimmedLine)
			if useResponsesAPI {
				converted := convertResponsesStreamLine(processLine)
				if converted == nil {
					continue
				}
				processLine = converted
			}

			// OpenAI-compatible streams must use SSE data lines.
			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, processLine, &param)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					return
				}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			reporter.PublishFailure(ctx, errScan)
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errScan}:
			case <-ctx.Done():
			}
		} else {
			// In case the upstream close the stream without a terminal [DONE] marker.
			// Feed a synthetic done marker through the translator so pending
			// response.completed events are still emitted exactly once.
			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte("data: [DONE]"), &param)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					return
				}
			}
		}
		// Ensure we record the request if no usage chunk was ever seen
		reporter.EnsurePublished(ctx)
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

func (e *OpenAICompatExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	modelForCounting := baseModel

	translated, err := thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	enc, err := helps.TokenizerForModel(modelForCounting)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("openai compat executor: tokenizer init failed: %w", err)
	}

	count, err := helps.CountOpenAIChatTokens(enc, translated)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("openai compat executor: token counting failed: %w", err)
	}

	usageJSON := helps.BuildOpenAIUsageJSON(count)
	translatedUsage := sdktranslator.TranslateTokenCount(ctx, to, from, count, usageJSON)
	return cliproxyexecutor.Response{Payload: translatedUsage}, nil
}

// Refresh is a no-op for API-key based compatibility providers.
func (e *OpenAICompatExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("openai compat executor: refresh called")
	if refreshed, handled, err := helps.RefreshAuthViaHome(ctx, e.cfg, auth); handled {
		return refreshed, err
	}
	return auth, nil
}

func (e *OpenAICompatExecutor) resolveCredentials(auth *cliproxyauth.Auth) (baseURL, apiKey string) {
	if auth == nil {
		return "", ""
	}
	if auth.Attributes != nil {
		baseURL = strings.TrimSpace(auth.Attributes["base_url"])
		apiKey = strings.TrimSpace(auth.Attributes["api_key"])
	}
	// For github-copilot, exchange GitHub OAuth token for Copilot API token
	if e.provider == "github-copilot" && apiKey != "" && baseURL != "" {
		if copilotToken, err := exchangeCopilotToken(apiKey); err == nil {
			apiKey = copilotToken
		} else {
			log.Debugf("github-copilot token exchange failed: %v", err)
		}
	}
	return
}

func (e *OpenAICompatExecutor) resolveCompatConfig(auth *cliproxyauth.Auth) *config.OpenAICompatibility {
	if auth == nil || e.cfg == nil {
		return nil
	}
	candidates := make([]string, 0, 3)
	if auth.Attributes != nil {
		if v := strings.TrimSpace(auth.Attributes["compat_name"]); v != "" {
			candidates = append(candidates, v)
		}
		if v := strings.TrimSpace(auth.Attributes["provider_key"]); v != "" {
			candidates = append(candidates, v)
		}
	}
	if v := strings.TrimSpace(auth.Provider); v != "" {
		candidates = append(candidates, v)
	}
	for i := range e.cfg.OpenAICompatibility {
		compat := &e.cfg.OpenAICompatibility[i]
		if compat.Disabled {
			continue
		}
		for _, candidate := range candidates {
			if candidate != "" && strings.EqualFold(strings.TrimSpace(candidate), compat.Name) {
				return compat
			}
		}
	}
	return nil
}

func (e *OpenAICompatExecutor) overrideModel(payload []byte, model string) []byte {
	if len(payload) == 0 || model == "" {
		return payload
	}
	payload, _ = sjson.SetBytes(payload, "model", model)
	return payload
}

// convertResponsesResponseToChatCompletions converts an OpenAI Responses API response
// to a Chat Completions response so the existing openai→claude translator can handle it.
func convertResponsesResponseToChatCompletions(body []byte) []byte {
	root := gjson.ParseBytes(body)

	out := []byte(`{"id":"","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`)

	// ID
	if v := root.Get("id"); v.Exists() {
		out, _ = sjson.SetBytes(out, "id", v.String())
	}
	// Model
	if v := root.Get("model"); v.Exists() {
		out, _ = sjson.SetBytes(out, "model", v.String())
	}
	// Created
	if v := root.Get("created_at"); v.Exists() {
		out, _ = sjson.SetBytes(out, "created", v.Int())
	}

	// Extract content and tool calls from output
	var textParts []string
	var toolCalls []map[string]any
	toolCallIdx := 0

	output := root.Get("output")
	if output.Exists() && output.IsArray() {
		output.ForEach(func(_, item gjson.Result) bool {
			switch item.Get("type").String() {
			case "message":
				item.Get("content").ForEach(func(_, content gjson.Result) bool {
					if content.Get("type").String() == "output_text" {
						textParts = append(textParts, content.Get("text").String())
					}
					return true
				})
			case "function_call":
				tc := map[string]any{
					"id":   item.Get("call_id").String(),
					"type": "function",
					"function": map[string]any{
						"name":      item.Get("name").String(),
						"arguments": item.Get("arguments").String(),
					},
					"index": toolCallIdx,
				}
				toolCalls = append(toolCalls, tc)
				toolCallIdx++
			}
			return true
		})
	}

	// Set content
	content := strings.Join(textParts, "")
	if content != "" {
		out, _ = sjson.SetBytes(out, "choices.0.message.content", content)
	}

	// Set tool calls
	if len(toolCalls) > 0 {
		out, _ = sjson.SetBytes(out, "choices.0.message.tool_calls", toolCalls)
		out, _ = sjson.SetBytes(out, "choices.0.finish_reason", "tool_calls")
	}

	// Usage
	if usage := root.Get("usage"); usage.Exists() {
		if v := usage.Get("input_tokens"); v.Exists() {
			out, _ = sjson.SetBytes(out, "usage.prompt_tokens", v.Int())
		}
		if v := usage.Get("output_tokens"); v.Exists() {
			out, _ = sjson.SetBytes(out, "usage.completion_tokens", v.Int())
		}
		total := usage.Get("input_tokens").Int() + usage.Get("output_tokens").Int()
		out, _ = sjson.SetBytes(out, "usage.total_tokens", total)
	}

	return out
}

// convertResponsesStreamLine converts a streaming SSE line from the Responses API
// into an OpenAI Chat Completions streaming format line.
func convertResponsesStreamLine(line []byte) []byte {
	if !bytes.HasPrefix(line, []byte("data:")) {
		return nil
	}
	data := bytes.TrimSpace(line[5:])
	if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
		return line
	}

	eventType := gjson.GetBytes(data, "type").String()
	switch eventType {
	case "response.output_text.delta":
		delta := gjson.GetBytes(data, "delta").String()
		chunk := fmt.Sprintf(`{"choices":[{"index":0,"delta":{"content":"%s"},"finish_reason":null}]}`,
			strings.ReplaceAll(strings.ReplaceAll(delta, `\`, `\\`), `"`, `\"`))
		return []byte("data: " + chunk)

	case "response.function_call_arguments.delta":
		delta := gjson.GetBytes(data, "delta").String()
		callID := gjson.GetBytes(data, "call_id").String()
		name := gjson.GetBytes(data, "name").String()
		idx := gjson.GetBytes(data, "output_index").Int()
		var chunk []byte
		if name != "" {
			// First chunk of function call with name
			chunk, _ = json.Marshal(map[string]any{
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []map[string]any{{
							"index": idx,
							"id":    callID,
							"type":  "function",
							"function": map[string]any{
								"name":      name,
								"arguments": delta,
							},
						}},
					},
					"finish_reason": nil,
				}},
			})
		} else {
			chunk, _ = json.Marshal(map[string]any{
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []map[string]any{{
							"index":    idx,
							"function": map[string]any{"arguments": delta},
						}},
					},
					"finish_reason": nil,
				}},
			})
		}
		return append([]byte("data: "), chunk...)

	case "response.completed":
		// Convert to final chunk with stop reason and usage
		finishReason := "stop"
		if output := gjson.GetBytes(data, "response.output"); output.Exists() {
			output.ForEach(func(_, item gjson.Result) bool {
				if item.Get("type").String() == "function_call" {
					finishReason = "tool_calls"
					return false
				}
				return true
			})
		}
		chunk, _ := json.Marshal(map[string]any{
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": finishReason,
			}},
		})
		result := append([]byte("data: "), chunk...)
		result = append(result, []byte("\ndata: [DONE]")...)
		return result

	case "response.output_item.added":
		// When a function_call item is added, emit the initial tool_call chunk with id and name
		itemType := gjson.GetBytes(data, "item.type").String()
		if itemType == "function_call" {
			callID := gjson.GetBytes(data, "item.call_id").String()
			name := gjson.GetBytes(data, "item.name").String()
			idx := gjson.GetBytes(data, "output_index").Int()
			chunk, _ := json.Marshal(map[string]any{
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []map[string]any{{
							"index": idx,
							"id":    callID,
							"type":  "function",
							"function": map[string]any{
								"name":      name,
								"arguments": "",
							},
						}},
					},
					"finish_reason": nil,
				}},
			})
			return append([]byte("data: "), chunk...)
		}
		return nil

	case "response.output_item.done",
		"response.content_part.added", "response.content_part.done",
		"response.created", "response.in_progress",
		"response.function_call_arguments.done":
		// Skip these events — they don't need direct mapping to chat completions streaming
		return nil

	default:
		return nil
	}
}

type statusErr struct {
	code       int
	msg        string
	retryAfter *time.Duration
}

func (e statusErr) Error() string {
	if e.msg != "" {
		return e.msg
	}
	return fmt.Sprintf("status %d", e.code)
}
func (e statusErr) StatusCode() int            { return e.code }
func (e statusErr) RetryAfter() *time.Duration { return e.retryAfter }

// exchangeCopilotToken exchanges a GitHub OAuth token for a short-lived Copilot API token.
func exchangeCopilotToken(githubToken string) (string, error) {
	req, err := http.NewRequest("GET", "https://api.github.com/copilot_internal/v2/token", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "token "+githubToken)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "GithubCopilot/1.300.0")
	req.Header.Set("Editor-Version", "vscode/1.100.0")
	req.Header.Set("Editor-Plugin-Version", "copilot/1.300.0")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("copilot token exchange failed: %d", resp.StatusCode)
	}

	var result struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if result.Token == "" {
		return "", fmt.Errorf("copilot token exchange returned empty token")
	}
	return result.Token, nil
}
