package composio

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/conneroisu/groq-go"
	"github.com/conneroisu/groq-go/pkg/builders"
)

type (
	// Runner is an interface for composio run.
	Runner interface {
		Run(ctx context.Context,
			user ConnectedAccount,
			response groq.ChatCompletionResponse) (
			[]groq.ChatCompletionMessage, error)
	}
	request struct {
		ConnectedAccountID string         `json:"connectedAccountId"`
		EntityID           string         `json:"entityId"`
		AppName            string         `json:"appName"`
		Input              map[string]any `json:"input"`
		Text               string         `json:"text,omitempty"`
		AuthConfig         map[string]any `json:"authConfig,omitempty"`
	}
)

// Run runs the composio client on a chat completion response.
func (c *Composio) Run(
	ctx context.Context,
	user ConnectedAccount,
	response groq.ChatCompletionResponse,
) ([]groq.ChatCompletionMessage, error) {
	var respH []groq.ChatCompletionMessage
	if response.Choices[0].FinishReason != groq.ReasonFunctionCall &&
		response.Choices[0].FinishReason != "tool_calls" {
		return nil, fmt.Errorf("not a function call")
	}

	resultChan := make(chan groq.ChatCompletionMessage)
	errChan := make(chan error)

	for _, toolCall := range response.Choices[0].Message.ToolCalls {
		go func(toolCall tools.ToolCall) {
			var args map[string]any
			if json.Valid([]byte(toolCall.Function.Arguments)) {
				err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
				if err != nil {
					errChan <- err
					return
				}
				c.logger.Debug("arguments", "args", args)
			}
			req, err := builders.NewRequest(
				ctx,
				c.header,
				http.MethodPost,
				fmt.Sprintf("%s/v2/actions/%s/execute", c.baseURL, toolCall.Function.Name),
				builders.WithBody(&request{
					ConnectedAccountID: user.ID,
					EntityID:           "default",
					AppName:            toolCall.Function.Name,
					Input:              args,
					AuthConfig:         map[string]any{},
				}),
			)
			if err != nil {
				errChan <- err
				return
			}
			var body string
			err = c.doRequest(req, &body)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- groq.ChatCompletionMessage{
				Content: body,
				Name:    toolCall.ID,
				Role:    groq.RoleFunction,
			}
		}(toolCall)
	}

	for range response.Choices[0].Message.ToolCalls {
		select {
		case result := <-resultChan:
			respH = append(respH, result)
		case err := <-errChan:
			return nil, err
		}
	}

	return respH, nil
}
