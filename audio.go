package groq

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
)

// AudioRequest represents a request structure for audio API.
type AudioRequest struct {
	Model                  Model                               // Model is the model to use for the transcription.
	FilePath               string                              // FilePath is either an existing file in your filesystem or a filename representing the contents of Reader.
	Reader                 io.Reader                           // Reader is an optional io.Reader when you do not want to use an existing file.
	Prompt                 string                              // Prompt is the prompt for the transcription.
	Temperature            float32                             // Temperature is the temperature for the transcription.
	Language               string                              // Language is the language for the transcription. Only for transcription.
	Format                 AudioResponseFormat                 // Format is the format for the response.
	TimestampGranularities []TranscriptionTimestampGranularity // Only for transcription. TimestampGranularities is the timestamp granularities for the transcription.
}

// AudioResponse represents a response structure for audio API.
type AudioResponse struct {
	Task     string   `json:"task"`     // Task is the task of the response.
	Language string   `json:"language"` // Language is the language of the response.
	Duration float64  `json:"duration"` // Duration is the duration of the response.
	Segments Segments `json:"segments"` // Segments is the segments of the response.
	Words    Words    `json:"words"`    // Words is the words of the response.
	Text     string   `json:"text"`     // Text is the text of the response.

	http.Header // Header is the header of the response.
}

// Words is the words of the response.
type Words []struct {
	Word  string  `json:"word"`  // Word is the word of the words.
	Start float64 `json:"start"` // Start is the start of the words.
	End   float64 `json:"end"`   // End is the end of the words.
}

// Segments is the segments of the response.
type Segments []struct {
	ID               int     `json:"id"`                // ID is the ID of the segment.
	Seek             int     `json:"seek"`              // Seek is the seek of the segment.
	Start            float64 `json:"start"`             // Start is the start of the segment.
	End              float64 `json:"end"`               // End is the end of the segment.
	Text             string  `json:"text"`              // Text is the text of the segment.
	Tokens           []int   `json:"tokens"`            // Tokens is the tokens of the segment.
	Temperature      float64 `json:"temperature"`       // Temperature is the temperature of the segment.
	AvgLogprob       float64 `json:"avg_logprob"`       // AvgLogprob is the avg log prob of the segment.
	CompressionRatio float64 `json:"compression_ratio"` // CompressionRatio is the compression ratio of the segment.
	NoSpeechProb     float64 `json:"no_speech_prob"`    // NoSpeechProb is the no speech prob of the segment.
	Transient        bool    `json:"transient"`         // Transient is the transient of the segment.
}

// SetHeader sets the header of the response.
func (r *AudioResponse) SetHeader(header http.Header) {
	r.Header = header
}

// audioTextResponse is the response structure for the audio API when the response format is text.
type audioTextResponse struct {
	Text   string      `json:"text"` // Text is the text of the response.
	header http.Header // Header is the header of the response.
}

// SetHeader sets the header of the audio text response.
func (r *audioTextResponse) SetHeader(header http.Header) {
	r.header = header
}

// ToAudioResponse converts the audio text response to an audio response.
func (r *audioTextResponse) ToAudioResponse() AudioResponse {
	return AudioResponse{
		Text:   r.Text,
		Header: r.header,
	}
}

// CreateTranscription — API call to create a transcription. Returns transcribed text.
func (c *Client) CreateTranscription(
	ctx context.Context,
	request AudioRequest,
) (response AudioResponse, err error) {
	return c.callAudioAPI(ctx, request, transcriptionsSuffix)
}

// CreateTranslation — API call to translate audio into English.
func (c *Client) CreateTranslation(
	ctx context.Context,
	request AudioRequest,
) (response AudioResponse, err error) {
	return c.callAudioAPI(ctx, request, translationsSuffix)
}

// callAudioAPI — API call to an audio endpoint.
func (c *Client) callAudioAPI(
	ctx context.Context,
	request AudioRequest,
	endpointSuffix Endpoint,
) (response AudioResponse, err error) {
	var formBody bytes.Buffer
	c.requestFormBuilder = c.createFormBuilder(&formBody)
	err = audioMultipartForm(request, c.requestFormBuilder)
	if err != nil {
		return AudioResponse{}, err
	}
	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(endpointSuffix, withModel(request.Model)),
		withBody(&formBody),
		withContentType(c.requestFormBuilder.FormDataContentType()),
	)
	if err != nil {
		return AudioResponse{}, err
	}

	if request.HasJSONResponse() {
		err = c.sendRequest(req, &response)
	} else {
		var textResponse audioTextResponse
		err = c.sendRequest(req, &textResponse)
		response = textResponse.ToAudioResponse()
	}
	if err != nil {
		return AudioResponse{}, err
	}
	return
}

// HasJSONResponse returns true if the response format is JSON.
func (r AudioRequest) HasJSONResponse() bool {
	return r.Format == "" || r.Format == AudioResponseFormatJSON ||
		r.Format == AudioResponseFormatVerboseJSON
}

// audioMultipartForm creates a form with audio file contents and the name of the model to use for
// audio processing.
func audioMultipartForm(request AudioRequest, b FormBuilder) error {
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

	if len(request.TimestampGranularities) > 0 {
		for _, tg := range request.TimestampGranularities {
			err = b.WriteField("timestamp_granularities[]", string(tg))
			if err != nil {
				return fmt.Errorf("writing timestamp_granularities[]: %w", err)
			}
		}
	}
	// Close the multipart writer
	return b.Close()
}

// createFileField creates the "file" form field from either an existing file or by using the reader.
func createFileField(
	request AudioRequest,
	b FormBuilder,
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
