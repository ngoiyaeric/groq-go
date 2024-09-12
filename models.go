// Code generated by groq-modeler DO NOT EDIT.
//
// Created at: 2024-09-12 16:53:14
//
// groq-modeler Version 1.0.0

package groq

// Endpoint is the endpoint for the groq api.
// string
type Endpoint string

// Model is the type for models present on the groq api.
// string
type Model string

// AudioResponseFormat is the response format for the audio API.
//
// Response formatted using AudioResponseFormatJSON by default.
//
// string
type AudioResponseFormat string

// TranscriptionTimestampGranularity is the timestamp granularity for the transcription.
//
// string
type TranscriptionTimestampGranularity string

const (
	completionsSuffix     Endpoint = "/completions"
	chatCompletionsSuffix Endpoint = "/chat/completions"
	transcriptionsSuffix  Endpoint = "/audio/transcriptions"
	translationsSuffix    Endpoint = "/audio/translations"
	embeddingsSuffix      Endpoint = "/embeddings"
	moderationsSuffix     Endpoint = "/moderations"

	AudioResponseFormatJSON        AudioResponseFormat = "json"         // AudioResponseFormatJSON is the JSON format of some audio.
	AudioResponseFormatText        AudioResponseFormat = "text"         // AudioResponseFormatText is the text format of some audio.
	AudioResponseFormatSRT         AudioResponseFormat = "srt"          // AudioResponseFormatSRT is the SRT format of some audio.
	AudioResponseFormatVerboseJSON AudioResponseFormat = "verbose_json" // AudioResponseFormatVerboseJSON is the verbose JSON format of some audio.
	AudioResponseFormatVTT         AudioResponseFormat = "vtt"          // AudioResponseFormatVTT is the VTT format of some audio.

	TranscriptionTimestampGranularityWord    TranscriptionTimestampGranularity = "word"                                  // TranscriptionTimestampGranularityWord is the word timestamp granularity.
	TranscriptionTimestampGranularitySegment TranscriptionTimestampGranularity = "segment"                               // TranscriptionTimestampGranularitySegment is the segment timestamp granularity.
	DistilWhisperLargeV3En                   Model                             = "distil-whisper-large-v3-en"            // DistilWhisperLargeV3En is an AI audio model provided by Hugging Face. It has 448 context window.
	Gemma29BIt                               Model                             = "gemma2-9b-it"                          // Gemma29BIt is an AI text model provided by Google. It has 8192 context window.
	Gemma7BIt                                Model                             = "gemma-7b-it"                           // Gemma7BIt is an AI text model provided by Google. It has 8192 context window.
	Llama3170BVersatile                      Model                             = "llama-3.1-70b-versatile"               // Llama3170BVersatile is an AI text model provided by Meta. It has 131072 context window.
	Llama318BInstant                         Model                             = "llama-3.1-8b-instant"                  // Llama318BInstant is an AI text model provided by Meta. It has 131072 context window.
	Llama370B8192                            Model                             = "llama3-70b-8192"                       // Llama370B8192 is an AI text model provided by Meta. It has 8192 context window.
	Llama38B8192                             Model                             = "llama3-8b-8192"                        // Llama38B8192 is an AI text model provided by Meta. It has 8192 context window.
	Llama3Groq70B8192ToolUsePreview          Model                             = "llama3-groq-70b-8192-tool-use-preview" // Llama3Groq70B8192ToolUsePreview is an AI text model provided by Groq. It has 8192 context window.
	Llama3Groq8B8192ToolUsePreview           Model                             = "llama3-groq-8b-8192-tool-use-preview"  // Llama3Groq8B8192ToolUsePreview is an AI text model provided by Groq. It has 8192 context window.
	LlamaGuard38B                            Model                             = "llama-guard-3-8b"                      // LlamaGuard38B is an AI moderation model provided by Meta. It has 8192 context window.
	LlavaV157B4096Preview                    Model                             = "llava-v1.5-7b-4096-preview"            // LlavaV157B4096Preview is an AI text model provided by Other. It has 4096 context window.
	Mixtral8X7B32768                         Model                             = "mixtral-8x7b-32768"                    // Mixtral8X7B32768 is an AI text model provided by Mistral AI. It has 32768 context window.
	WhisperLargeV3                           Model                             = "whisper-large-v3"                      // WhisperLargeV3 is an AI audio model provided by OpenAI. It has 448 context window.
)

var disabledModelsForEndpoints = map[Endpoint]map[Model]bool{
	completionsSuffix: {
		DistilWhisperLargeV3En: true,
		WhisperLargeV3:         true,
	},
	chatCompletionsSuffix: {
		DistilWhisperLargeV3En: true,
		WhisperLargeV3:         true,
	},
	transcriptionsSuffix: {
		Gemma29BIt:                      true,
		Gemma7BIt:                       true,
		Llama3170BVersatile:             true,
		Llama318BInstant:                true,
		Llama370B8192:                   true,
		Llama38B8192:                    true,
		Llama3Groq70B8192ToolUsePreview: true,
		Llama3Groq8B8192ToolUsePreview:  true,
		LlavaV157B4096Preview:           true,
		Mixtral8X7B32768:                true,
	},
	translationsSuffix: {
		Gemma29BIt:                      true,
		Gemma7BIt:                       true,
		Llama3170BVersatile:             true,
		Llama318BInstant:                true,
		Llama370B8192:                   true,
		Llama38B8192:                    true,
		Llama3Groq70B8192ToolUsePreview: true,
		Llama3Groq8B8192ToolUsePreview:  true,
		LlavaV157B4096Preview:           true,
		Mixtral8X7B32768:                true,
	},
	moderationsSuffix: {
		LlamaGuard38B: true,
	},
}

func endpointSupportsModel(endpoint Endpoint, model Model) bool {
	return !disabledModelsForEndpoints[endpoint][model]
}
