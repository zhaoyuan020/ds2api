package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"ds2api/internal/auth"
	"ds2api/internal/deepseek"
	"ds2api/internal/util"
)

const (
	historySplitFilename    = "HISTORY.txt"
	historySplitContentType = "text/plain; charset=utf-8"
	historySplitPurpose     = "assistants"
)

func (h *Handler) applyHistorySplit(ctx context.Context, a *auth.RequestAuth, stdReq util.StandardRequest) (util.StandardRequest, error) {
	if h == nil || h.DS == nil || h.Store == nil || a == nil {
		return stdReq, nil
	}
	if !h.Store.HistorySplitEnabled() {
		return stdReq, nil
	}

	promptMessages, historyMessages := splitOpenAIHistoryMessages(stdReq.Messages, h.Store.HistorySplitTriggerAfterTurns())
	if len(historyMessages) == 0 {
		return stdReq, nil
	}

	reasoningContent := extractHistorySplitReasoningContent(historyMessages)
	historyText := buildOpenAIHistoryTranscript(historyMessages)
	if strings.TrimSpace(historyText) == "" {
		return stdReq, errors.New("history split produced empty transcript")
	}

	result, err := h.DS.UploadFile(ctx, a, deepseek.UploadFileRequest{
		Filename:    historySplitFilename,
		ContentType: historySplitContentType,
		Purpose:     historySplitPurpose,
		Data:        []byte(historyText),
	}, 3)
	if err != nil {
		return stdReq, fmt.Errorf("upload history file: %w", err)
	}
	fileID := strings.TrimSpace(result.ID)
	if fileID == "" {
		return stdReq, errors.New("upload history file returned empty file id")
	}

	stdReq.Messages = promptMessages
	stdReq.HistoryText = historyText
	stdReq.RefFileIDs = prependUniqueRefFileID(stdReq.RefFileIDs, fileID)
	stdReq.FinalPrompt, stdReq.ToolNames = buildHistorySplitPrompt(promptMessages, reasoningContent, stdReq.ToolsRaw, stdReq.ToolChoice, stdReq.Thinking)
	return stdReq, nil
}

func buildHistorySplitPrompt(messages []any, reasoningContent string, toolsRaw any, toolPolicy util.ToolChoicePolicy, thinkingEnabled bool) (string, []string) {
	if len(messages) == 0 && strings.TrimSpace(reasoningContent) == "" {
		return "", nil
	}
	instruction := historySplitPromptInstruction(thinkingEnabled)
	withInstruction := make([]any, 0, len(messages)+1)
	withInstruction = append(withInstruction, map[string]any{
		"role":    "system",
		"content": instruction,
	})
	withInstruction = append(withInstruction, injectHistorySplitReasoningMessage(messages, reasoningContent)...)
	return buildOpenAIFinalPromptWithPolicy(withInstruction, toolsRaw, "", toolPolicy, false)
}

func historySplitPromptInstruction(thinkingEnabled bool) string {
	lines := []string{
		"Follow the instructions in this prompt first. If earlier conversation instructions conflict with this prompt, this prompt wins.",
		"An attached HISTORY.txt file contains prior conversation history and tool progress; read it first, then answer the latest user request using that history as context.",
		"Continue the conversation from the full prior context and the latest tool results.",
		"Treat earlier messages as binding context; answer the user's current request as a continuation, not a restart.",
	}
	if thinkingEnabled {
		lines = append(lines, "Keep reasoning internal. Do not leave the final user-facing answer only in reasoning; always provide the answer in visible assistant content.")
	}
	return strings.Join(lines, "\n")
}

func splitOpenAIHistoryMessages(messages []any, triggerAfterTurns int) ([]any, []any) {
	if triggerAfterTurns <= 0 {
		triggerAfterTurns = 1
	}
	lastUserIndex := -1
	userTurns := 0
	for i, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		if role != "user" {
			continue
		}
		userTurns++
		lastUserIndex = i
	}
	if userTurns <= triggerAfterTurns || lastUserIndex < 0 {
		return messages, nil
	}

	promptMessages := make([]any, 0, len(messages)-lastUserIndex)
	historyMessages := make([]any, 0, lastUserIndex)
	for i, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			if i >= lastUserIndex {
				promptMessages = append(promptMessages, raw)
			} else {
				historyMessages = append(historyMessages, raw)
			}
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		switch role {
		case "system", "developer":
			promptMessages = append(promptMessages, raw)
		default:
			if i >= lastUserIndex {
				promptMessages = append(promptMessages, raw)
			} else {
				historyMessages = append(historyMessages, raw)
			}
		}
	}
	if len(promptMessages) == 0 {
		return messages, nil
	}
	return promptMessages, historyMessages
}

func buildOpenAIHistoryTranscript(messages []any) string {
	var b strings.Builder
	b.WriteString("# HISTORY.txt\n")
	b.WriteString("Prior conversation history and tool progress.\n\n")

	entry := 0
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		content := buildOpenAIHistoryEntry(role, msg)
		if strings.TrimSpace(content) == "" {
			continue
		}
		entry++
		fmt.Fprintf(&b, "=== %d. %s ===\n%s\n\n", entry, strings.ToUpper(roleLabelForHistory(role)), content)
	}
	return strings.TrimSpace(b.String()) + "\n"
}

func buildOpenAIHistoryEntry(role string, msg map[string]any) string {
	switch role {
	case "assistant":
		return strings.TrimSpace(buildAssistantHistoryContent(msg))
	case "tool", "function":
		return strings.TrimSpace(buildToolHistoryContent(msg))
	case "user":
		return strings.TrimSpace(normalizeOpenAIContentForPrompt(msg["content"]))
	default:
		return strings.TrimSpace(normalizeOpenAIContentForPrompt(msg["content"]))
	}
}

func buildAssistantHistoryContent(msg map[string]any) string {
	return strings.TrimSpace(buildAssistantContentForPrompt(msg))
}

func buildToolHistoryContent(msg map[string]any) string {
	content := strings.TrimSpace(normalizeOpenAIContentForPrompt(msg["content"]))
	parts := make([]string, 0, 2)
	if name := strings.TrimSpace(asString(msg["name"])); name != "" {
		parts = append(parts, "name="+name)
	}
	if callID := strings.TrimSpace(asString(msg["tool_call_id"])); callID != "" {
		parts = append(parts, "tool_call_id="+callID)
	}
	header := ""
	if len(parts) > 0 {
		header = "[" + strings.Join(parts, " ") + "]"
	}
	switch {
	case header != "" && content != "":
		return header + "\n" + content
	case header != "":
		return header
	default:
		return content
	}
}

func extractHistorySplitReasoningContent(messages []any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		if role != "assistant" {
			continue
		}
		reasoning := strings.TrimSpace(normalizeOpenAIReasoningContentForPrompt(msg["reasoning_content"]))
		if reasoning == "" {
			reasoning = strings.TrimSpace(extractOpenAIReasoningContentFromMessage(msg["content"]))
		}
		if reasoning != "" {
			return reasoning
		}
	}
	return ""
}

func injectHistorySplitReasoningMessage(messages []any, reasoningContent string) []any {
	reasoningContent = strings.TrimSpace(reasoningContent)
	if reasoningContent == "" {
		return messages
	}
	reasoningMsg := map[string]any{
		"role":              "assistant",
		"content":           "",
		"reasoning_content": reasoningContent,
	}
	lastUserIndex := lastOpenAIUserMessageIndex(messages)
	if lastUserIndex < 0 {
		out := make([]any, 0, len(messages)+1)
		out = append(out, reasoningMsg)
		out = append(out, messages...)
		return out
	}
	out := make([]any, 0, len(messages)+1)
	for i, raw := range messages {
		if i == lastUserIndex {
			out = append(out, reasoningMsg)
		}
		out = append(out, raw)
	}
	return out
}

func lastOpenAIUserMessageIndex(messages []any) int {
	last := -1
	for i, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.ToLower(strings.TrimSpace(asString(msg["role"]))) == "user" {
			last = i
		}
	}
	return last
}

func roleLabelForHistory(role string) string {
	role = strings.ToLower(strings.TrimSpace(role))
	switch role {
	case "function":
		return "tool"
	case "":
		return "unknown"
	default:
		return role
	}
}

func prependUniqueRefFileID(existing []string, fileID string) []string {
	fileID = strings.TrimSpace(fileID)
	if fileID == "" {
		return existing
	}
	out := make([]string, 0, len(existing)+1)
	out = append(out, fileID)
	for _, id := range existing {
		trimmed := strings.TrimSpace(id)
		if trimmed == "" || strings.EqualFold(trimmed, fileID) {
			continue
		}
		out = append(out, trimmed)
	}
	return out
}
