package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/conneroisu/groq-go"
	"github.com/conneroisu/groq-go/extensions/toolhouse"
	"github.com/conneroisu/groq-go/internal/test"
)

func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	toolhouseKey, err := test.GetAPIKey("TOOLHOUSE_API_KEY")
	if err != nil {
		return err
	}
	ext, err := toolhouse.NewExtension(toolhouseKey,
		toolhouse.WithMetadata(map[string]any{
			"id":       "conner",
			"timezone": 5,
		}))
	if err != nil {
		return err
	}
	groqKey, err := test.GetAPIKey("GROQ_KEY")
	if err != nil {
		return err
	}
	client, err := groq.NewClient(groqKey)
	if err != nil {
		return err
	}
	history := []groq.ChatCompletionMessage{
		{
			Role:    groq.RoleUser,
			Content: "Write a python function to print the first 10 prime numbers containing the number 3 then respond with the answer. DO NOT GUESS WHAT THE OUTPUT SHOULD BE. MAKE SURE TO CALL THE TOOL GIVEN.",
		},
	}
	tools, err := ext.GetTools(ctx)
	if err != nil {
		return err
	}
	re, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model:      groq.ModelLlama3Groq70B8192ToolUsePreview,
		Messages:   history,
		Tools:      tools,
		ToolChoice: "required",
	})
	if err != nil {
		return fmt.Errorf("failed to create 1 chat completion: %w", err)
	}
	history = append(history, re.Choices[0].Message)

	resultChan := make(chan []groq.ChatCompletionMessage)
	errChan := make(chan error)

	go func() {
		r, err := ext.Run(ctx, re)
		if err != nil {
			errChan <- err
			return
		}
		resultChan <- r
	}()

	select {
	case r := <-resultChan:
		history = append(history, r...)
	case err := <-errChan:
		return fmt.Errorf("failed to run tool: %w", err)
	}

	finalr, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model:     groq.ModelLlama3Groq70B8192ToolUsePreview,
		Messages:  history,
		MaxTokens: 2000,
	})
	if err != nil {
		return fmt.Errorf("failed to create 2 chat completion: %w", err)
	}
	history = append(history, finalr.Choices[0].Message)
	jsnHistory, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal history: %w", err)
	}
	fmt.Println(string(jsnHistory))
	return nil
}
