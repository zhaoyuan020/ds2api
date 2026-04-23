package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"ds2api/internal/auth"
	"ds2api/internal/util"
)

func historySplitTestMessages() []any {
	toolCalls := []any{
		map[string]any{
			"name":      "search",
			"arguments": map[string]any{"query": "docs"},
		},
	}
	return []any{
		map[string]any{"role": "system", "content": "system instructions"},
		map[string]any{"role": "user", "content": "first user turn"},
		map[string]any{
			"role":              "assistant",
			"content":           "",
			"reasoning_content": "hidden reasoning",
			"tool_calls":        toolCalls,
		},
		map[string]any{
			"role":         "tool",
			"name":         "search",
			"tool_call_id": "call-1",
			"content":      "tool result",
		},
		map[string]any{"role": "user", "content": "latest user turn"},
	}
}

func TestBuildOpenAIHistoryTranscriptPreservesOrderAndToolHistory(t *testing.T) {
	promptMessages, historyMessages := splitOpenAIHistoryMessages(historySplitTestMessages(), 1)
	if len(promptMessages) != 2 {
		t.Fatalf("expected 2 prompt messages, got %d", len(promptMessages))
	}
	if len(historyMessages) != 3 {
		t.Fatalf("expected 3 history messages, got %d", len(historyMessages))
	}

	transcript := buildOpenAIHistoryTranscript(historyMessages)
	if !strings.Contains(transcript, "first user turn") {
		t.Fatalf("expected user history in transcript, got %s", transcript)
	}
	if !strings.Contains(transcript, "<tool_calls>") {
		t.Fatalf("expected assistant tool_calls in transcript, got %s", transcript)
	}
	if !strings.Contains(transcript, "tool_call_id=call-1") {
		t.Fatalf("expected tool call id in transcript, got %s", transcript)
	}
	if !strings.Contains(transcript, "[reasoning_content]") {
		t.Fatalf("expected reasoning block in HISTORY.txt, got %s", transcript)
	}
	if !strings.Contains(transcript, "hidden reasoning") {
		t.Fatalf("expected reasoning text in HISTORY.txt, got %s", transcript)
	}

	userIdx := strings.Index(transcript, "=== 1. USER ===")
	assistantIdx := strings.Index(transcript, "=== 2. ASSISTANT ===")
	toolIdx := strings.Index(transcript, "=== 3. TOOL ===")
	if userIdx < 0 || assistantIdx < 0 || toolIdx < 0 {
		t.Fatalf("expected ordered role sections, got %s", transcript)
	}
	if userIdx >= assistantIdx || assistantIdx >= toolIdx {
		t.Fatalf("expected USER -> ASSISTANT -> TOOL order, got %s", transcript)
	}
	if reasoningIdx := strings.Index(transcript, "[reasoning_content]"); reasoningIdx < 0 || reasoningIdx > strings.Index(transcript, "<tool_calls>") {
		t.Fatalf("expected reasoning block before tool calls, got %s", transcript)
	}
	reasoning := extractHistorySplitReasoningContent(historyMessages)
	if reasoning != "hidden reasoning" {
		t.Fatalf("expected latest assistant reasoning to be extracted, got %q", reasoning)
	}

	finalPrompt, _ := buildHistorySplitPrompt(promptMessages, reasoning, nil, util.DefaultToolChoicePolicy(), false)
	if !strings.Contains(finalPrompt, "latest user turn") {
		t.Fatalf("expected latest user turn in final prompt, got %s", finalPrompt)
	}
	if strings.Contains(finalPrompt, "first user turn") {
		t.Fatalf("expected earlier history to be removed from final prompt, got %s", finalPrompt)
	}
	if !strings.Contains(finalPrompt, "[reasoning_content]") || !strings.Contains(finalPrompt, "hidden reasoning") {
		t.Fatalf("expected latest assistant reasoning to be attached to prompt, got %s", finalPrompt)
	}
	if !strings.Contains(finalPrompt, "HISTORY.txt") {
		t.Fatalf("expected history instruction in final prompt, got %s", finalPrompt)
	}
	if !strings.Contains(finalPrompt, "Follow the instructions in this prompt first") {
		t.Fatalf("expected stronger prompt override in final prompt, got %s", finalPrompt)
	}
	if strings.Index(finalPrompt, "Follow the instructions in this prompt first") > strings.Index(finalPrompt, "Continue the conversation") {
		t.Fatalf("expected history split instruction before continuity instructions, got %s", finalPrompt)
	}
}

func TestSplitOpenAIHistoryMessagesUsesLatestUserTurn(t *testing.T) {
	toolCalls := []any{
		map[string]any{
			"name":      "search",
			"arguments": map[string]any{"query": "docs"},
		},
	}
	messages := []any{
		map[string]any{"role": "system", "content": "system instructions"},
		map[string]any{"role": "user", "content": "first user turn"},
		map[string]any{
			"role":       "assistant",
			"content":    "",
			"tool_calls": toolCalls,
		},
		map[string]any{
			"role":         "tool",
			"name":         "search",
			"tool_call_id": "call-1",
			"content":      "tool result",
		},
		map[string]any{"role": "user", "content": "middle user turn"},
		map[string]any{
			"role":    "assistant",
			"content": "middle assistant turn",
		},
		map[string]any{"role": "user", "content": "latest user turn"},
	}

	promptMessages, historyMessages := splitOpenAIHistoryMessages(messages, 1)
	if len(promptMessages) == 0 || len(historyMessages) == 0 {
		t.Fatalf("expected both prompt and history messages, got prompt=%d history=%d", len(promptMessages), len(historyMessages))
	}
	reasoning := extractHistorySplitReasoningContent(historyMessages)
	if reasoning != "" {
		t.Fatalf("expected no reasoning in this fixture, got %q", reasoning)
	}

	promptText, _ := buildHistorySplitPrompt(promptMessages, reasoning, nil, util.DefaultToolChoicePolicy(), false)
	if !strings.Contains(promptText, "latest user turn") {
		t.Fatalf("expected latest user turn in prompt, got %s", promptText)
	}
	if strings.Contains(promptText, "middle user turn") {
		t.Fatalf("expected middle user turn to be split into history, got %s", promptText)
	}

	historyText := buildOpenAIHistoryTranscript(historyMessages)
	if !strings.Contains(historyText, "middle user turn") {
		t.Fatalf("expected middle user turn in HISTORY.txt, got %s", historyText)
	}
	if strings.Contains(historyText, "latest user turn") {
		t.Fatalf("expected latest user turn to remain in prompt, got %s", historyText)
	}
}

func TestApplyHistorySplitSkipsFirstTurn(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &Handler{
		Store: mockOpenAIConfig{
			wideInput:           true,
			historySplitEnabled: true,
			historySplitTurns:   1,
		},
		DS: ds,
	}
	req := map[string]any{
		"model": "deepseek-chat",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
	}
	stdReq, err := normalizeOpenAIChatRequest(h.Store, req, "")
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}

	out, err := h.applyHistorySplit(context.Background(), &auth.RequestAuth{DeepSeekToken: "token"}, stdReq)
	if err != nil {
		t.Fatalf("apply history split failed: %v", err)
	}
	if len(ds.uploadCalls) != 0 {
		t.Fatalf("expected no upload on first turn, got %d", len(ds.uploadCalls))
	}
	if out.FinalPrompt != stdReq.FinalPrompt {
		t.Fatalf("expected prompt unchanged on first turn")
	}
	if len(out.RefFileIDs) != len(stdReq.RefFileIDs) {
		t.Fatalf("expected ref files unchanged on first turn")
	}
}

func TestApplyHistorySplitCarriesHistoryText(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &Handler{
		Store: mockOpenAIConfig{
			wideInput:           true,
			historySplitEnabled: true,
			historySplitTurns:   1,
		},
		DS: ds,
	}
	req := map[string]any{
		"model":    "deepseek-chat",
		"messages": historySplitTestMessages(),
	}
	stdReq, err := normalizeOpenAIChatRequest(h.Store, req, "")
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}

	out, err := h.applyHistorySplit(context.Background(), &auth.RequestAuth{DeepSeekToken: "token"}, stdReq)
	if err != nil {
		t.Fatalf("apply history split failed: %v", err)
	}
	if len(ds.uploadCalls) != 1 {
		t.Fatalf("expected 1 upload call, got %d", len(ds.uploadCalls))
	}
	if out.HistoryText != string(ds.uploadCalls[0].Data) {
		t.Fatalf("expected history text to be preserved on normalized request")
	}
}

func TestChatCompletionsHistorySplitUploadsHistoryAndKeepsLatestPrompt(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &Handler{
		Store: mockOpenAIConfig{
			wideInput:           true,
			historySplitEnabled: true,
			historySplitTurns:   1,
		},
		Auth: streamStatusAuthStub{},
		DS:   ds,
	}
	reqBody, _ := json.Marshal(map[string]any{
		"model":    "deepseek-chat",
		"messages": historySplitTestMessages(),
		"stream":   false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if len(ds.uploadCalls) != 1 {
		t.Fatalf("expected 1 upload call, got %d", len(ds.uploadCalls))
	}
	upload := ds.uploadCalls[0]
	if upload.Filename != "HISTORY.txt" {
		t.Fatalf("unexpected upload filename: %q", upload.Filename)
	}
	if upload.ContentType != "text/plain; charset=utf-8" {
		t.Fatalf("unexpected content type: %q", upload.ContentType)
	}
	if upload.Purpose != "assistants" {
		t.Fatalf("unexpected purpose: %q", upload.Purpose)
	}
	historyText := string(upload.Data)
	if !strings.Contains(historyText, "first user turn") || !strings.Contains(historyText, "tool result") {
		t.Fatalf("expected older turns in HISTORY.txt, got %s", historyText)
	}
	if strings.Contains(historyText, "latest user turn") {
		t.Fatalf("expected latest turn to remain in prompt, got %s", historyText)
	}
	if ds.completionReq == nil {
		t.Fatal("expected completion payload to be captured")
	}
	promptText, _ := ds.completionReq["prompt"].(string)
	if !strings.Contains(promptText, "latest user turn") {
		t.Fatalf("expected latest turn in completion prompt, got %s", promptText)
	}
	if strings.Contains(promptText, "first user turn") {
		t.Fatalf("expected historical turns removed from completion prompt, got %s", promptText)
	}
	if !strings.Contains(promptText, "[reasoning_content]") || !strings.Contains(promptText, "hidden reasoning") {
		t.Fatalf("expected latest assistant reasoning to be attached to completion prompt, got %s", promptText)
	}
	if !strings.Contains(promptText, "HISTORY.txt") {
		t.Fatalf("expected history instruction in completion prompt, got %s", promptText)
	}
	if !strings.Contains(promptText, "Follow the instructions in this prompt first") {
		t.Fatalf("expected stronger prompt override in completion prompt, got %s", promptText)
	}
	if strings.Index(promptText, "Follow the instructions in this prompt first") > strings.Index(promptText, "Continue the conversation") {
		t.Fatalf("expected history split instruction before continuity instructions, got %s", promptText)
	}
	refIDs, _ := ds.completionReq["ref_file_ids"].([]any)
	if len(refIDs) == 0 || refIDs[0] != "file-inline-1" {
		t.Fatalf("expected uploaded history file to be first ref_file_id, got %#v", ds.completionReq["ref_file_ids"])
	}
}

func TestResponsesHistorySplitUploadsHistoryAndKeepsLatestPrompt(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &Handler{
		Store: mockOpenAIConfig{
			wideInput:           true,
			historySplitEnabled: true,
			historySplitTurns:   1,
		},
		Auth: streamStatusAuthStub{},
		DS:   ds,
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)
	reqBody, _ := json.Marshal(map[string]any{
		"model":    "deepseek-chat",
		"messages": historySplitTestMessages(),
		"stream":   false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if len(ds.uploadCalls) != 1 {
		t.Fatalf("expected 1 upload call, got %d", len(ds.uploadCalls))
	}
	if ds.completionReq == nil {
		t.Fatal("expected completion payload to be captured")
	}
	promptText, _ := ds.completionReq["prompt"].(string)
	if !strings.Contains(promptText, "latest user turn") {
		t.Fatalf("expected latest turn in completion prompt, got %s", promptText)
	}
	if strings.Contains(promptText, "first user turn") {
		t.Fatalf("expected historical turns removed from completion prompt, got %s", promptText)
	}
	if !strings.Contains(promptText, "[reasoning_content]") || !strings.Contains(promptText, "hidden reasoning") {
		t.Fatalf("expected latest assistant reasoning to be attached to completion prompt, got %s", promptText)
	}
	if !strings.Contains(promptText, "Follow the instructions in this prompt first") {
		t.Fatalf("expected stronger prompt override in completion prompt, got %s", promptText)
	}
	if strings.Index(promptText, "Follow the instructions in this prompt first") > strings.Index(promptText, "Continue the conversation") {
		t.Fatalf("expected history split instruction before continuity instructions, got %s", promptText)
	}
}

func TestChatCompletionsHistorySplitUploadFailureReturnsInternalServerError(t *testing.T) {
	ds := &inlineUploadDSStub{uploadErr: context.DeadlineExceeded}
	h := &Handler{
		Store: mockOpenAIConfig{
			wideInput:           true,
			historySplitEnabled: true,
			historySplitTurns:   1,
		},
		Auth: streamStatusAuthStub{},
		DS:   ds,
	}
	reqBody, _ := json.Marshal(map[string]any{
		"model":    "deepseek-chat",
		"messages": historySplitTestMessages(),
		"stream":   false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d body=%s", rec.Code, rec.Body.String())
	}
	if ds.completionReq != nil {
		t.Fatalf("did not expect completion payload on upload failure")
	}
}
