package groq

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/conneroisu/groq-go/internal/schema"
	"github.com/conneroisu/groq-go/internal/streams"
	"github.com/conneroisu/groq-go/pkg/builders"
	"github.com/conneroisu/groq-go/pkg/groqerr"
	"github.com/conneroisu/groq-go/pkg/tools"
)

// # Formats

// Format is the format of a response.
// string
type Format string

const (
	// FormatText is the text format.
	FormatText Format = "text"
	// FormatJSON is the JSON format.
	// There is no support for streaming with JSON format selected.
	FormatJSON Format = "json"
	// FormatJSONObject is the json object chat
	// completion response format type.
	FormatJSONObject Format = "json_object"
	// FormatJSONSchema is the json schema chat
	// completion response format type.
	FormatJSONSchema Format = "json_schema"
	// FormatSRT is the SRT format.
	// Only supported for the transcription API.
	FormatSRT Format = "srt"
	// FormatVTT is the VTT format.
	// Only supported for the transcription API.
	FormatVTT Format = "vtt"
	// FormatVerboseJSON is the verbose JSON format.
	// Only supported for the transcription API.
	FormatVerboseJSON Format = "verbose_json"
)

// # [Chat](https://console.groq.com/docs/api-reference#chat-create)

// ChatCompletionRequest represents a request structure for the chat
// completion API.
type (
	ChatCompletionRequest struct {
		// Model is the model of the chat completion request.
		Model ChatModel `json:"model"`
		// Messages are the messages of the chat completion request.
		Messages []ChatCompletionMessage `json:"messages"`
		// MaxTokens is the max tokens that the model can generate.
		MaxTokens int `json:"max_tokens,omitempty"`
		// Temperature is the temperature of the model during inference.
		Temperature float32 `json:"temperature,omitempty"`
		// TopP is the top p of the of the model during inference.
		TopP float32 `json:"top_p,omitempty"`
		// N is the n of the chat completion request.
		N int `json:"n,omitempty"`
		// Stream is the stream of the chat completion request.
		Stream bool `json:"stream,omitempty"`
		// Stop is the stop of the chat completion request.
		Stop []string `json:"stop,omitempty"`
		// PresencePenalty is the presence penalty for the model during
		// inference.
		PresencePenalty float32 `json:"presence_penalty,omitempty"`
		// ResponseFormat is the response format of the chat completion
		// request.
		ResponseFormat *ChatResponseFormat `json:"response_format,omitempty"`
		// Seed is the seed of the chat completion request.
		Seed *int `json:"seed,omitempty"`
		// FrequencyPenalty is the frequency penalty of the chat
		// completion request.
		FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
		// LogitBias is must be a token id string (specified by their
		// token ID in the tokenizer), not a word string.
		// incorrect: `"logit_bias":{ "You": 6}`, correct: `"logit_bias":{"1639": 6}`
		// refs: https://platform.openai.com/docs/api-reference/chat/create#chat/create-logit_bias
		LogitBias map[string]int `json:"logit_bias,omitempty"`
		// LogProbs indicates whether to return log probabilities of the
		// output tokens or not. If true, returns the log probabilities
		// of each output token returned in the content of message.
		//
		// This option is currently not available on the
		// gpt-4-vision-preview model.
		LogProbs bool `json:"logprobs,omitempty"`
		// TopLogProbs is an integer between 0 and 5 specifying the
		// number of most likely tokens to return at each token
		// position, each with an associated log probability. Logprobs
		// must be set to true if this parameter is used.
		TopLogProbs int `json:"top_logprobs,omitempty"`
		// User is the user of the chat completion request.
		User string `json:"user,omitempty"`
		// Tools is the tools of the chat completion request.
		Tools []tools.Tool `json:"tools,omitempty"`
		// This can be either a string or an ToolChoice object.
		ToolChoice any `json:"tool_choice,omitempty"`
		// Options for streaming response. Only set this when you set stream: true.
		StreamOptions *StreamOptions `json:"stream_options,omitempty"`
		// Disable the default behavior of parallel tool calls by setting it: false.
		ParallelToolCalls any `json:"parallel_tool_calls,omitempty"`
		// RetryDelay is the delay between retries.
		RetryDelay time.Duration `json:"-"`
	}
	// ChatCompletionResponse represents a response structure for chat
	// completion API.
	ChatCompletionResponse struct {
		// ID is the id of the response.
		ID string `json:"id"`
		// Object is the object of the response.
		Object string `json:"object"`
		// Created is the created time of the response.
		Created int64 `json:"created"`
		// Model is the model of the response.
		Model ChatModel `json:"model"`
		// Choices is the choices of the response.
		Choices []ChatCompletionChoice `json:"choices"`
		// Usage is the usage of the response.
		Usage Usage `json:"usage"`
		// SystemFingerprint is the system fingerprint of the response.
		SystemFingerprint string `json:"system_fingerprint"`
		header            http.Header
	}
)

// ChatCompletionMessage represents the chat completion message.
type ChatCompletionMessage struct {
	// Name is the name of the chat completion message.
	Name string `json:"name"`
	// Role is the role of the chat completion message.
	Role Role `json:"role"`
	// Content is the content of the chat completion message.
	Content string `json:"content"`
	// MultiContent is the multi content of the chat completion
	// message.
	MultiContent []ChatMessagePart `json:"-"`
	// FunctionCall setting for Role=assistant prompts this may be
	// set to the function call generated by the model.
	FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
	// ToolCalls setting for Role=assistant prompts this may be set
	// to the tool calls generated by the model, such as function
	// calls.
	ToolCalls []tools.ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is setting for Role=tool prompts this should be
	// set to the ID given in the assistant's prior request to call
	// a tool.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// Role is the role of the chat completion message.
//
// string
type Role string

const (
	// RoleSystem is a chat role that represents the system.
	RoleSystem Role = "system"
	// RoleUser is a chat role that represents the user.
	RoleUser Role = "user"
	// RoleAssistant is a chat role that represents the assistant.
	RoleAssistant Role = "assistant"
	// RoleFunction is a chat role that represents the function.
	RoleFunction Role = "function"
	// RoleTool is a chat role that represents the tool.
	RoleTool Role = "tool"
)

// ChatMessagePart represents the chat message part of a chat completion
// message.
type ChatMessagePart struct {
	// Text is the text of the chat message part.
	Text string `json:"text,omitempty"`
	// Type is the type of the chat message part.
	Type ChatMessagePartType `json:"type,omitempty"`
	// ImageURL is the image url of the chat message part.
	ImageURL *ChatMessageImageURL `json:"image_url,omitempty"`
}

// ChatResponseFormat is the chat completion response format.
type ChatResponseFormat struct {
	// Type is the type of the chat completion response format.
	Type Format `json:"type,omitempty"`
	// JSONSchema is the json schema of the chat completion response
	// format.
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// ChatMessagePartType is the chat message part type.
//
// string
type ChatMessagePartType string

const (
	// ChatMessagePartTypeText is the text chat message part type.
	ChatMessagePartTypeText ChatMessagePartType = "text"
	// ChatMessagePartTypeImageURL is the image url chat message part type.
	ChatMessagePartTypeImageURL ChatMessagePartType = "image_url"
)

// ChatMessageImageURL represents the chat message image url.
type ChatMessageImageURL struct {
	// URL is the url of the image.
	URL string `json:"url,omitempty"`
	// Detail is the detail of the image url.
	Detail ImageURLDetail `json:"detail,omitempty"`
}

type (
	// JSONSchema is the chat completion
	// response format json schema.
	JSONSchema struct {
		// Name is the name of the chat completion response format json
		// schema.
		//
		// it is used to further identify the schema in the response.
		Name string `json:"name"`
		// Description is the description of the chat completion
		// response format json schema.
		Description string `json:"description,omitempty"`
		// Schema is the schema of the chat completion response format
		// json schema.
		Schema schema.Schema `json:"schema"`
		// Strict determines whether to enforce the schema upon the
		// generated content.
		Strict bool `json:"strict"`
	}
	// LogProbs is the top-level structure containing the log probability information.
	LogProbs struct {
		// Content is a list of message content tokens with log
		// probability information.
		Content []struct {
			LogProb `json:"logprobs"`
			// TopLogProbs is a list of the most likely tokens and
			// their log probability, at this token position. In
			// rare cases, there may be fewer than the number of
			// requested top_logprobs returned.
			TopLogProbs []LogProb `json:"top_logprobs"`
		} `json:"content"`
	}
	// LogProb represents the log prob of a token.
	LogProb struct {
		// Token is the token that the log prob is for.
		Token string `json:"token"`
		// LogProb is the log prob of the token.
		LogProb float64 `json:"logprob"`
		// Bytes are the bytes of the token.
		Bytes []byte `json:"bytes,omitempty"`
	}
	// ChatCompletionChoice represents the chat completion choice.
	ChatCompletionChoice struct {
		// Index is the index of the choice.
		Index int `json:"index"`
		// Message is the chat completion message of the choice.
		Message ChatCompletionMessage `json:"message"`
		// FinishReason is the finish reason of the choice.
		FinishReason FinishReason `json:"finish_reason"`
		// LogProbs is the logarithmic probabilities of the choice of
		// the model for each token.
		LogProbs *LogProbs `json:"logprobs,omitempty"`
	}
	// ChatCompletionStreamChoiceDelta represents a response structure for
	// chat completion API.
	ChatCompletionStreamChoiceDelta struct {
		// Content is the content of the response.
		Content string `json:"content,omitempty"`
		// Role is the role of the creator of the completion.
		Role string `json:"role,omitempty"`
		// FunctionCall is the function call of the response.
		FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
		// ToolCalls are the tool calls of the response.
		ToolCalls []tools.ToolCall `json:"tool_calls,omitempty"`
	}
	// ChatCompletionStreamChoice represents a response structure for chat
	// completion API.
	ChatCompletionStreamChoice struct {
		// Index is the index of the choice.
		Index int `json:"index"`
		// Delta is the delta of the choice.
		Delta ChatCompletionStreamChoiceDelta `json:"delta"`
		// FinishReason is the finish reason of the choice.
		FinishReason FinishReason `json:"finish_reason"`
	}

	// ChatCompletionStreamResponse represents a response structure for chat
	// completion API.
	ChatCompletionStreamResponse struct {
		// ID is the identifier for the chat completion stream response.
		ID string `json:"id"`
		// Object is the object type of the chat completion stream
		// response.
		Object string `json:"object"`
		// Created is the creation time of the chat completion stream
		// response.
		Created int64 `json:"created"`
		// Model is the model used for the chat completion stream
		// response.
		Model ChatModel `json:"model"`
		// Choices is the choices for the chat completion stream
		// response.
		Choices []ChatCompletionStreamChoice `json:"choices"`
		// SystemFingerprint is the system fingerprint for the chat
		// completion stream response.
		SystemFingerprint string `json:"system_fingerprint"`
		// PromptAnnotations is the prompt annotations for the chat
		// completion stream response.
		PromptAnnotations []PromptAnnotation `json:"prompt_annotations,omitempty"`
		// PromptFilterResults is the prompt filter results for the chat
		// completion stream response.
		PromptFilterResults []struct {
			Index int `json:"index"`
		} `json:"prompt_filter_results,omitempty"`
		// Usage is an optional field that will only be present when you
		// set stream_options: {"include_usage": true} in your request.
		//
		// When present, it contains a null value except for the last
		// chunk which contains the token usage statistics for the
		// entire request.
		Usage *Usage `json:"usage,omitempty"`
	}
	// PromptAnnotation represents the prompt annotation.
	PromptAnnotation struct {
		PromptIndex int `json:"prompt_index,omitempty"`
	}
	// StreamOptions represents the stream options.
	StreamOptions struct {
		// IncludeUsage is the include usage option of a stream request.
		//
		// If set, an additional chunk will be streamed before the data:
		// [DONE] message.
		// The usage field on this chunk shows the token usage
		// statistics for the entire request, and the choices field will
		// always be an empty array.
		//
		// All other chunks will also include a usage field, but with a
		// null value.
		IncludeUsage bool `json:"include_usage,omitempty"`
	}
	// ChatCompletionStream is a stream of ChatCompletionStreamResponse.
	ChatCompletionStream struct {
		*streams.StreamReader[*ChatCompletionStreamResponse]
	}
)

type (
	// FinishReason is the finish reason.
	//
	// string
	FinishReason string
)

const (
	// ReasonStop is the stop finish reason for a chat completion.
	ReasonStop FinishReason = "stop"
	// ReasonLength is the length finish reason for a chat completion.
	ReasonLength FinishReason = "length"
	// ReasonFunctionCall is the function call finish reason for a chat
	// completion.
	// Deprecated: use ReasonToolCalls instead.
	ReasonFunctionCall FinishReason = "function_call"
	// ReasonToolCalls is the tool calls finish reason for a chat
	// completion.
	ReasonToolCalls FinishReason = "tool_calls"
	// ReasonContentFilter is the content filter finish reason for a chat
	// completion.
	ReasonContentFilter FinishReason = "content_filter"
	// ReasonNull is the null finish reason for a chat completion.
	ReasonNull FinishReason = "null"
)

// MarshalJSON method implements the json.Marshaler interface.
//
// It exists to allow for the use of the multi-part content field.
func (m ChatCompletionMessage) MarshalJSON() ([]byte, error) {
	if m.Content != "" && m.MultiContent != nil {
		return nil, &groqerr.ErrContentFieldsMisused{}
	}
	if len(m.MultiContent) > 0 {
		msg := struct {
			Name         string              `json:"name,omitempty"`
			Role         Role                `json:"role"`
			Content      string              `json:"-"`
			MultiContent []ChatMessagePart   `json:"content,omitempty"`
			FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
			ToolCalls    []tools.ToolCall    `json:"tool_calls,omitempty"`
			ToolCallID   string              `json:"tool_call_id,omitempty"`
		}(m)
		return json.Marshal(msg)
	}
	msg := struct {
		Name         string              `json:"name,omitempty"`
		Role         Role                `json:"role"`
		Content      string              `json:"content"`
		MultiContent []ChatMessagePart   `json:"-"`
		FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
		ToolCalls    []tools.ToolCall    `json:"tool_calls,omitempty"`
		ToolCallID   string              `json:"tool_call_id,omitempty"`
	}(m)
	return json.Marshal(msg)
}

// UnmarshalJSON method implements the json.Unmarshaler interface.
//
// It exists to allow for the use of the multi-part content field.
func (m *ChatCompletionMessage) UnmarshalJSON(bs []byte) (err error) {
	msg := struct {
		Name         string `json:"name,omitempty"`
		Role         Role   `json:"role"`
		Content      string `json:"content"`
		MultiContent []ChatMessagePart
		FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
		ToolCalls    []tools.ToolCall    `json:"tool_calls,omitempty"`
		ToolCallID   string              `json:"tool_call_id,omitempty"`
	}{}
	err = json.Unmarshal(bs, &msg)
	if err == nil {
		*m = ChatCompletionMessage(msg)
		return nil
	}
	multiMsg := struct {
		Name         string `json:"name,omitempty"`
		Role         Role   `json:"role"`
		Content      string
		MultiContent []ChatMessagePart   `json:"content"`
		FunctionCall *tools.FunctionCall `json:"function_call,omitempty"`
		ToolCalls    []tools.ToolCall    `json:"tool_calls,omitempty"`
		ToolCallID   string              `json:"tool_call_id,omitempty"`
	}{}
	err = json.Unmarshal(bs, &multiMsg)
	if err != nil {
		return err
	}
	*m = ChatCompletionMessage(multiMsg)
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (r FinishReason) MarshalJSON() ([]byte, error) {
	if r == ReasonNull || r == "" {
		return []byte("null"), nil
	}
	return []byte(
		`"` + string(r) + `"`,
	), nil // best effort to not break future API changes
}

// SetHeader sets the header of the response.
func (r *ChatCompletionResponse) SetHeader(h http.Header) { r.header = h }

type (
	// Usage Represents the total token usage per request to Groq.
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}
	endpoint       string
	fullURLOptions struct{ model string }
	fullURLOption  func(*fullURLOptions)
	response       interface{ SetHeader(http.Header) }
)

// # Images

// ImageURLDetail is the detail of the image at the URL.
//
// string
type ImageURLDetail string

const (
	// ImageURLDetailHigh is the high image url detail.
	ImageURLDetailHigh ImageURLDetail = "high"
	// ImageURLDetailLow is the low image url detail.
	ImageURLDetailLow ImageURLDetail = "low"
	// ImageURLDetailAuto is the auto image url detail.
	ImageURLDetailAuto ImageURLDetail = "auto"
)

// # Moderation

type (
	// Moderation is a category of harmful content that can be used to
	// categorize a chat history using Llama Guard 3.
	//
	// [Llama Guard 3](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
	//
	// Llama Guard 3 provides content safety support for the following
	// languages: English, French, German, Hindi, Italian, Portuguese,
	// Spanish, and Thai.
	//
	// string
	Moderation string
)

const (
	// ModerationViolentCrimes (S1) is the violent crimes category.
	//
	// Responses that enable, encourage, or endorse the commission of
	// violent crimes, including: (1) unlawful violence toward people
	// (ex: terrorism, genocide, murder, hate-motivated violent crimes,
	// child abuse, assault, battery, kidnapping) and (2) unlawful violence
	// toward animals (ex: animal abuse)
	ModerationViolentCrimes Moderation = "violent_crimes"
	// ModerationNonviolentCrimes (S2) is the non-violent crimes category.
	ModerationNonviolentCrimes Moderation = "non_violent_crimes"
	// ModerationSexRelatedCrimes (S3) is the sex-related crimes category.
	ModerationSexRelatedCrimes Moderation = "sex_related_crimes"
	// ModerationChildSexualExploitation (S4) is the child sexual exploitation
	// category.
	//
	// Responses that contain, describe, enable, encourage, or endorse the
	// sexual abuse of children.
	ModerationChildSexualExploitation Moderation = "child_sexual_exploitation"
	// ModerationDefamation (S5) is the defamation category.
	//
	// Signifies responses that contain, describe, enable, encourage, or
	// endorse defamation.
	ModerationDefamation Moderation = "defamation"
	// ModerationSpecializedAdvice (S6) is the specialized advice category.
	//
	// Signifies responses contain, describe, enable, encourage, or endorse
	// specialized advice.
	ModerationSpecializedAdvice Moderation = "specialized_advice"
	// ModerationPrivacy (S7) is the privacy category.
	//
	// Responses contain, describe, enable, encourage, or endorse privacy.
	ModerationPrivacy Moderation = "privacy"
	// ModerationIntellectualProperty (S8) is the intellectual property
	// category. Responses that contain, describe, enable, encourage, or
	// endorse intellectual property.
	ModerationIntellectualProperty Moderation = "intellectual_property"
	// ModerationIndiscriminateWeapons (S9) is the indiscriminate weapons
	// category.
	//
	// Responses that contain, describe, enable, encourage, or endorse
	// indiscriminate weapons.
	ModerationIndiscriminateWeapons Moderation = "indiscriminate_weapons"
	// ModerationHate (S10) is the hate category.
	//
	// Responses contain, describe, enable, encourage, or endorse hate.
	ModerationHate Moderation = "hate"
	// ModerationSuicideOrSelfHarm (S11) is the suicide/self-harm category.
	//
	// Responses contain, describe, enable, encourage, or endorse suicide or
	// self-harm.
	ModerationSuicideOrSelfHarm Moderation = "suicide_and_self_harm"
	// ModerationSexualContent (S12) is the sexual content category.
	//
	// Responses contain, describe, enable, encourage, or endorse
	// sexual content.
	ModerationSexualContent Moderation = "sexual_content"
	// ModerationElections (S13) is the elections category.
	//
	// Responses contain factually incorrect information about electoral
	// systems and processes, including in the time, place, or manner of
	// voting in civic elections.
	ModerationElections Moderation = "elections"
	// ModerationCodeInterpreterAbuse (S14) is the code interpreter abuse
	// category.
	//
	// Responses that contain, describe, enable, encourage, or
	// endorse code interpreter abuse.
	ModerationCodeInterpreterAbuse Moderation = "code_interpreter_abuse"
)

var (
	sectionMap = map[string]Moderation{
		"S1":  ModerationViolentCrimes,
		"S2":  ModerationNonviolentCrimes,
		"S3":  ModerationSexRelatedCrimes,
		"S4":  ModerationChildSexualExploitation,
		"S5":  ModerationDefamation,
		"S6":  ModerationSpecializedAdvice,
		"S7":  ModerationPrivacy,
		"S8":  ModerationIntellectualProperty,
		"S9":  ModerationIndiscriminateWeapons,
		"S10": ModerationHate,
		"S11": ModerationSuicideOrSelfHarm,
		"S12": ModerationSexualContent,
		"S13": ModerationElections,
		"S14": ModerationCodeInterpreterAbuse,
	}
)

// # [Audio](https://console.groq.com/docs/api-reference#audio-transcription)

type (
	// AudioRequest represents a request structure for audio API.
	AudioRequest struct {
		// Model is the model to use for the transcription.
		Model AudioModel
		// FilePath is either an existing file in your filesystem or a
		// filename representing the contents of Reader.
		FilePath string
		// Reader is an optional io.Reader when you do not want to use
		// an existing file.
		Reader io.Reader
		// Prompt is the prompt for the transcription.
		Prompt string
		// Temperature is the temperature for the transcription.
		Temperature float32
		// Language is the language for the transcription. Only for
		// transcription.
		Language string
		// Format is the format for the response.
		Format Format
	}
	// AudioResponse represents a response structure for audio API.
	AudioResponse struct {
		// Task is the task of the response.
		Task string `json:"task"`
		// Language is the language of the response.
		Language string `json:"language"`
		// Duration is the duration of the response.
		Duration float64 `json:"duration"`
		// Segments is the segments of the response.
		Segments Segments `json:"segments"`
		// Words is the words of the response.
		Words Words `json:"words"`
		// Text is the text of the response.
		Text string `json:"text"`

		header http.Header `json:"-"`
	}
	// Words is the words of the audio response.
	Words []struct {
		// Word is the textual representation of a word in the audio
		// response.
		Word string `json:"word"`
		// Start is the start of the words in seconds.
		Start float64 `json:"start"`
		// End is the end of the words in seconds.
		End float64 `json:"end"`
	}
	// Segments is the segments of the response.
	Segments []struct {
		// ID is the ID of the segment.
		ID int `json:"id"`
		// Seek is the seek of the segment.
		Seek int `json:"seek"`
		// Start is the start of the segment.
		Start float64 `json:"start"`
		// End is the end of the segment.
		End float64 `json:"end"`
		// Text is the text of the segment.
		Text string `json:"text"`
		// Tokens is the tokens of the segment.
		Tokens []int `json:"tokens"`
		// Temperature is the temperature of the segment.
		Temperature float64 `json:"temperature"`
		// AvgLogprob is the avg log prob of the segment.
		AvgLogprob float64 `json:"avg_logprob"`
		// CompressionRatio is the compression ratio of the segment.
		CompressionRatio float64 `json:"compression_ratio"`
		// NoSpeechProb is the no speech prob of the segment.
		NoSpeechProb float64 `json:"no_speech_prob"`
		// Transient is the transient of the segment.
		Transient bool `json:"transient"`
	}
	// audioTextResponse is the response structure for the audio API when the
	// response format is text.
	audioTextResponse struct {
		// Text is the text of the response.
		Text   string      `json:"text"`
		header http.Header `json:"-"`
	}
)

// SetHeader sets the header of the response.
func (r *AudioResponse) SetHeader(header http.Header) { r.header = header }

// SetHeader sets the header of the audio text response.
func (r *audioTextResponse) SetHeader(header http.Header) { r.header = header }

// toAudioResponse converts the audio text response to an audio response.
func (r *audioTextResponse) toAudioResponse() AudioResponse {
	return AudioResponse{Text: r.Text, header: r.header}
}

func (r AudioRequest) hasJSONResponse() bool {
	return r.Format == "" || r.Format == FormatJSON ||
		r.Format == FormatVerboseJSON
}

func createFileField(
	request AudioRequest,
	b builders.FormBuilder,
) (err error) {
	if request.Reader != nil {
		err := b.CreateFormFileReader("file", request.Reader, request.FilePath)
		if err != nil {
			return fmt.Errorf("creating form using reader: %w", err)
		}
		return nil
	}
	f, err := os.Open(request.FilePath)
	if err != nil {
		return fmt.Errorf("opening audio file: %w", err)
	}
	defer f.Close()
	err = b.CreateFormFile("file", f)
	if err != nil {
		return fmt.Errorf("creating form file: %w", err)
	}
	return nil
}

// audioMultipartForm creates a form with audio file contents and the name of
// the model to use for audio processing.
func audioMultipartForm(request AudioRequest, b builders.FormBuilder) error {
	err := createFileField(request, b)
	if err != nil {
		return err
	}
	err = b.WriteField("model", string(request.Model))
	if err != nil {
		return fmt.Errorf("writing model name: %w", err)
	}
	// Create a form field for the prompt (if provided)
	if request.Prompt != "" {
		err = b.WriteField("prompt", request.Prompt)
		if err != nil {
			return fmt.Errorf("writing prompt: %w", err)
		}
	}
	// Create a form field for the format (if provided)
	if request.Format != "" {
		err = b.WriteField("response_format", string(request.Format))
		if err != nil {
			return fmt.Errorf("writing format: %w", err)
		}
	}
	// Create a form field for the temperature (if provided)
	if request.Temperature != 0 {
		err = b.WriteField(
			"temperature",
			fmt.Sprintf("%.2f", request.Temperature),
		)
		if err != nil {
			return fmt.Errorf("writing temperature: %w", err)
		}
	}
	// Create a form field for the language (if provided)
	if request.Language != "" {
		err = b.WriteField("language", request.Language)
		if err != nil {
			return fmt.Errorf("writing language: %w", err)
		}
	}
	return b.Close()
}
