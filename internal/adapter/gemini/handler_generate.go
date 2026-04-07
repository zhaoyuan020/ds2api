package gemini

import (
	"bytes"
	"ds2api/internal/toolcall"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/go-chi/chi/v5"

	"ds2api/internal/sse"
	"ds2api/internal/translatorcliproxy"
	"ds2api/internal/util"

	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
)

func (h *Handler) handleGenerateContent(w http.ResponseWriter, r *http.Request, stream bool) {
	if h.OpenAI == nil {
		writeGeminiError(w, http.StatusInternalServerError, "OpenAI proxy backend unavailable.")
		return
	}
	if h.proxyViaOpenAI(w, r, stream) {
		return
	}
	writeGeminiError(w, http.StatusBadGateway, "Failed to proxy Gemini request.")
}

func (h *Handler) proxyViaOpenAI(w http.ResponseWriter, r *http.Request, stream bool) bool {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, "invalid body")
		return true
	}
	routeModel := strings.TrimSpace(chi.URLParam(r, "model"))
	translatedReq := translatorcliproxy.ToOpenAI(sdktranslator.FormatGemini, routeModel, raw, stream)
	if !strings.Contains(string(translatedReq), `"stream"`) {
		var reqMap map[string]any
		if json.Unmarshal(translatedReq, &reqMap) == nil {
			reqMap["stream"] = stream
			if b, e := json.Marshal(reqMap); e == nil {
				translatedReq = b
			}
		}
	}

	isVercelPrepare := strings.TrimSpace(r.URL.Query().Get("__stream_prepare")) == "1"
	isVercelRelease := strings.TrimSpace(r.URL.Query().Get("__stream_release")) == "1"

	if isVercelRelease {
		proxyReq := r.Clone(r.Context())
		proxyReq.URL.Path = "/v1/chat/completions"
		proxyReq.Body = io.NopCloser(bytes.NewReader(raw))
		proxyReq.ContentLength = int64(len(raw))
		rec := httptest.NewRecorder()
		h.OpenAI.ChatCompletions(rec, proxyReq)
		res := rec.Result()
		defer func() { _ = res.Body.Close() }()
		body, _ := io.ReadAll(res.Body)
		for k, vv := range res.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(res.StatusCode)
		_, _ = w.Write(body)
		return true
	}

	proxyReq := r.Clone(r.Context())
	proxyReq.URL.Path = "/v1/chat/completions"
	proxyReq.Body = io.NopCloser(bytes.NewReader(translatedReq))
	proxyReq.ContentLength = int64(len(translatedReq))

	if stream && !isVercelPrepare {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache, no-transform")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		streamWriter := translatorcliproxy.NewOpenAIStreamTranslatorWriter(w, sdktranslator.FormatGemini, routeModel, raw, translatedReq)
		h.OpenAI.ChatCompletions(streamWriter, proxyReq)
		return true
	}

	rec := httptest.NewRecorder()
	h.OpenAI.ChatCompletions(rec, proxyReq)
	res := rec.Result()
	defer func() { _ = res.Body.Close() }()
	body, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		for k, vv := range res.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		writeGeminiErrorFromOpenAI(w, res.StatusCode, body)
		return true
	}
	if isVercelPrepare {
		for k, vv := range res.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(res.StatusCode)
		_, _ = w.Write(body)
		return true
	}
	converted := translatorcliproxy.FromOpenAINonStream(sdktranslator.FormatGemini, routeModel, raw, translatedReq, body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(converted)
	return true
}

func writeGeminiErrorFromOpenAI(w http.ResponseWriter, status int, raw []byte) {
	message := strings.TrimSpace(string(raw))
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err == nil {
		if errObj, ok := parsed["error"].(map[string]any); ok {
			if msg, ok := errObj["message"].(string); ok && strings.TrimSpace(msg) != "" {
				message = strings.TrimSpace(msg)
			}
		}
	}
	if message == "" {
		message = http.StatusText(status)
	}
	writeGeminiError(w, status, message)
}

//nolint:unused // retained for native Gemini non-stream handling path.
func (h *Handler) handleNonStreamGenerateContent(w http.ResponseWriter, resp *http.Response, model, finalPrompt string, thinkingEnabled bool, toolNames []string) {
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		writeGeminiError(w, resp.StatusCode, strings.TrimSpace(string(body)))
		return
	}

	result := sse.CollectStream(resp, thinkingEnabled, true)
	stripReferenceMarkers := h.compatStripReferenceMarkers()
	writeJSON(w, http.StatusOK, buildGeminiGenerateContentResponse(
		model,
		finalPrompt,
		cleanVisibleOutput(result.Thinking, stripReferenceMarkers),
		cleanVisibleOutput(result.Text, stripReferenceMarkers),
		toolNames,
	))
}

//nolint:unused // retained for native Gemini non-stream handling path.
func buildGeminiGenerateContentResponse(model, finalPrompt, finalThinking, finalText string, toolNames []string) map[string]any {
	parts := buildGeminiPartsFromFinal(finalText, finalThinking, toolNames)
	usage := buildGeminiUsage(finalPrompt, finalThinking, finalText)
	return map[string]any{
		"candidates": []map[string]any{
			{
				"index": 0,
				"content": map[string]any{
					"role":  "model",
					"parts": parts,
				},
				"finishReason": "STOP",
			},
		},
		"modelVersion":  model,
		"usageMetadata": usage,
	}
}

//nolint:unused // retained for native Gemini non-stream handling path.
func buildGeminiUsage(finalPrompt, finalThinking, finalText string) map[string]any {
	promptTokens := util.EstimateTokens(finalPrompt)
	reasoningTokens := util.EstimateTokens(finalThinking)
	completionTokens := util.EstimateTokens(finalText)
	return map[string]any{
		"promptTokenCount":     promptTokens,
		"candidatesTokenCount": reasoningTokens + completionTokens,
		"totalTokenCount":      promptTokens + reasoningTokens + completionTokens,
	}
}

//nolint:unused // retained for native Gemini non-stream handling path.
func buildGeminiPartsFromFinal(finalText, finalThinking string, toolNames []string) []map[string]any {
	detected := toolcall.ParseToolCalls(finalText, toolNames)
	if len(detected) == 0 && finalThinking != "" {
		detected = toolcall.ParseToolCalls(finalThinking, toolNames)
	}
	if len(detected) > 0 {
		parts := make([]map[string]any, 0, len(detected))
		for _, tc := range detected {
			parts = append(parts, map[string]any{
				"functionCall": map[string]any{
					"name": tc.Name,
					"args": tc.Input,
				},
			})
		}
		return parts
	}

	text := finalText
	if text == "" {
		text = finalThinking
	}
	return []map[string]any{{"text": text}}
}
