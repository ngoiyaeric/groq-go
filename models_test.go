// Code generated by groq-modeler DO NOT EDIT.
//
// Created at: 2024-12-17 10:47:56
//
// groq-modeler Version 1.1.2

package groq_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/conneroisu/groq-go"
	"github.com/conneroisu/groq-go/internal/test"
	"github.com/stretchr/testify/assert"

	_ "embed"
)

//go:embed testdata/whisper.mp3
var whisperBytes []byte

// TestChatModelsGemma29BIt tests the Gemma29BIt model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsGemma29BIt(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Gemma29BIt test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelGemma29BIt,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Gemma29BIt calling ChatCompletion")
	}
}

// TestChatModelsGemma7BIt tests the Gemma7BIt model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsGemma7BIt(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Gemma7BIt test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelGemma7BIt,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Gemma7BIt calling ChatCompletion")
	}
}

// TestChatModelsLlama3170BVersatile tests the Llama3170BVersatile model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3170BVersatile(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3170BVersatile test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3170BVersatile,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3170BVersatile calling ChatCompletion")
	}
}

// TestChatModelsLlama318BInstant tests the Llama318BInstant model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama318BInstant(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama318BInstant test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama318BInstant,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama318BInstant calling ChatCompletion")
	}
}

// TestChatModelsLlama3211BVisionPreview tests the Llama3211BVisionPreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3211BVisionPreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3211BVisionPreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3211BVisionPreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3211BVisionPreview calling ChatCompletion")
	}
}

// TestChatModelsLlama321BPreview tests the Llama321BPreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama321BPreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama321BPreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama321BPreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama321BPreview calling ChatCompletion")
	}
}

// TestChatModelsLlama323BPreview tests the Llama323BPreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama323BPreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama323BPreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama323BPreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama323BPreview calling ChatCompletion")
	}
}

// TestChatModelsLlama3290BVisionPreview tests the Llama3290BVisionPreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3290BVisionPreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3290BVisionPreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3290BVisionPreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3290BVisionPreview calling ChatCompletion")
	}
}

// TestChatModelsLlama3370BSpecdec tests the Llama3370BSpecdec model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3370BSpecdec(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3370BSpecdec test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3370BSpecdec,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3370BSpecdec calling ChatCompletion")
	}
}

// TestChatModelsLlama3370BVersatile tests the Llama3370BVersatile model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3370BVersatile(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3370BVersatile test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3370BVersatile,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3370BVersatile calling ChatCompletion")
	}
}

// TestChatModelsLlama370B8192 tests the Llama370B8192 model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama370B8192(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama370B8192 test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama370B8192,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama370B8192 calling ChatCompletion")
	}
}

// TestChatModelsLlama38B8192 tests the Llama38B8192 model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama38B8192(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama38B8192 test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama38B8192,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama38B8192 calling ChatCompletion")
	}
}

// TestChatModelsLlama3Groq70B8192ToolUsePreview tests the Llama3Groq70B8192ToolUsePreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3Groq70B8192ToolUsePreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3Groq70B8192ToolUsePreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3Groq70B8192ToolUsePreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3Groq70B8192ToolUsePreview calling ChatCompletion")
	}
}

// TestChatModelsLlama3Groq8B8192ToolUsePreview tests the Llama3Groq8B8192ToolUsePreview model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsLlama3Groq8B8192ToolUsePreview(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Llama3Groq8B8192ToolUsePreview test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelLlama3Groq8B8192ToolUsePreview,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Llama3Groq8B8192ToolUsePreview calling ChatCompletion")
	}
}

// TestChatModelsMixtral8X7B32768 tests the Mixtral8X7B32768 model.
//
// It ensures that the model is supported by the groq-go library and the groq
// API. // and the operations are working as expected for the specific model type.
func TestChatModelsMixtral8X7B32768(t *testing.T) {
	if len(os.Getenv("UNIT")) < 1 {
		t.Skip("Skipping Mixtral8X7B32768 test")
	}
	a := assert.New(t)
	ctx := context.Background()
	apiKey, err := test.GetAPIKey("GROQ_KEY")
	a.NoError(err, "GetAPIKey error")
	client, err := groq.NewClient(apiKey)
	a.NoError(err, "NewClient error")
	response, err := client.ChatCompletion(ctx, groq.ChatCompletionRequest{
		Model: groq.ModelMixtral8X7B32768,
		Messages: []groq.ChatCompletionMessage{
			{
				Role:    groq.RoleUser,
				Content: "What is a proface display?",
			},
		},
		MaxTokens:  10,
		RetryDelay: time.Second * 2,
	})
	a.NoError(err, "ChatCompletionJSON error")
	if len(response.Choices[0].Message.Content) == 0 {
		t.Errorf("response.Choices[0].Message.Content is empty for model Mixtral8X7B32768 calling ChatCompletion")
	}
}
