package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"github.com/appthrust/appserver-go/appserver/releaseidentity"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/appthrust/appserver-go/appserver/httpx"
)

const (
	maxRequestBytes = 8 << 20
	maxInFlight     = 4
	requestTimeout  = 120 * time.Second
)

func upstreamClient() *http.Client {
	return &http.Client{
		Timeout: requestTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func newHandler(upstream *url.URL, ready func() bool) http.Handler {
	// Only the routes registered below reach vLLM. Never expose model loading,
	// profiling, tokenizer, or other management APIs, or forward edge credentials.
	proxy := &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(upstream)
			r.Out.URL.RawQuery, r.Out.URL.ForceQuery = "", false
			r.Out.Header = make(http.Header)
			if r.Out.Method == http.MethodPost {
				r.Out.Header.Set("Content-Type", "application/json")
			}
		},
		FlushInterval: -1, // Forward SSE chunks immediately, without buffering a completion.
		ModifyResponse: func(r *http.Response) error {
			r.Header.Set("Cache-Control", "no-store")
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			if errors.Is(err, context.DeadlineExceeded) {
				writeError(w, http.StatusGatewayTimeout, "vision_timeout", "Vision request exceeded 120 seconds")
			} else {
				writeError(w, http.StatusBadGateway, "vision_upstream_unavailable", "The vision upstream could not complete this request")
			}
		},
	}
	slots := make(chan struct{}, maxInFlight)
	modelJSON, _ := json.Marshal(servedModel)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if !ready() {
			writeError(w, http.StatusServiceUnavailable, "vision_unavailable", "The vision model is not ready")
			return
		}
		select {
		case slots <- struct{}{}:
			defer func() { <-slots }()
		default:
			writeError(w, http.StatusServiceUnavailable, "vision_busy", "At most four vision requests may run at once")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
		defer cancel()
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
		decoder := json.NewDecoder(r.Body)
		var body map[string]json.RawMessage
		err := decoder.Decode(&body)
		if err == nil {
			var extra json.RawMessage
			if next := decoder.Decode(&extra); next != io.EOF {
				err = next
				if err == nil {
					err = errors.New("multiple JSON values")
				}
			}
		}
		if err != nil || body == nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				writeError(w, http.StatusRequestEntityTooLarge, "request_too_large", "Vision request body must not exceed 8 MiB")
			} else {
				writeError(w, http.StatusBadRequest, "invalid_request", "Vision request must be one JSON object")
			}
			return
		}
		if raw, ok := body["model"]; ok {
			var requested string
			if err := json.Unmarshal(raw, &requested); err != nil || (requested != "" && requested != servedModel) {
				writeError(w, http.StatusBadRequest, "invalid_request", "model must be qwen3-vl-4b-instruct-fp8")
				return
			}
		}
		body["model"] = modelJSON
		payload, err := json.Marshal(body)
		if err != nil {
			httpx.WriteError(w, err, "vision")
			return
		}
		r = r.WithContext(ctx)
		r.Body = io.NopCloser(bytes.NewReader(payload))
		r.ContentLength = int64(len(payload))
		proxy.ServeHTTP(w, r)
	})
	for _, path := range []string{"/v1/models", "/metrics"} {
		mux.HandleFunc("GET "+path, func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
			defer cancel()
			proxy.ServeHTTP(w, r.WithContext(ctx))
		})
	}
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		if !ready() {
			writeError(w, http.StatusServiceUnavailable, "vision_unavailable", "The vision model is not ready")
			return
		}
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.Handle("GET /releasez", releaseidentity.NewHandler("VISION", func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, http.StatusServiceUnavailable, "release_unavailable", "Vision release status unavailable")
	}))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"model": servedModel})
	})
	return mux
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	httpx.WriteError(w, httpx.Fail(status, code, message), "vision")
}
