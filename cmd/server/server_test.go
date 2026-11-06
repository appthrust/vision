package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func handlerForTest(t *testing.T, upstream string, ready func() bool) http.Handler {
	t.Helper()
	u, err := url.Parse(upstream)
	require.NoError(t, err)
	return newHandler(u, ready)
}

func requestForTest(h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
	return w
}

func TestProxyForcesBakedModelAndOnlyForwardsPublicRoutes(t *testing.T) {
	requests := make(chan *http.Request, 8)
	bodies := make(chan map[string]any, 8)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		requests <- r
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hello"}}]}`)
	}))
	defer upstream.Close()
	h := handlerForTest(t, upstream.URL, func() bool { return true })
	for _, model := range []string{"", `,"model":""`, `,"model":"qwen3-vl-4b-instruct-fp8"`} {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions?ignored=1", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]`+model+`}`))
		r.Header.Set("Authorization", "Bearer private-user-token")
		r.Header.Set("Cookie", "session=private")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		forwarded, body := <-requests, <-bodies
		require.Equal(t, "/v1/chat/completions", forwarded.URL.Path)
		require.Empty(t, forwarded.URL.RawQuery)
		require.Empty(t, forwarded.Header.Get("Authorization"))
		require.Empty(t, forwarded.Header.Get("Cookie"))
		require.Equal(t, servedModel, body["model"])
		require.Len(t, body["messages"], 1)
	}
	for _, path := range []string{"/load_lora_adapter", "/unload_lora_adapter", "/start_profile", "/tokenize", "/v1/completions", "/v1/chat/completions/extra", "/other"} {
		for _, method := range []string{http.MethodGet, http.MethodPost, http.MethodDelete} {
			require.Equal(t, http.StatusNotFound, requestForTest(h, method, path, `{}`).Code, path)
		}
	}
	require.Equal(t, http.StatusMethodNotAllowed, requestForTest(h, http.MethodGet, "/v1/chat/completions", "").Code)
	require.Empty(t, requests)
}

func TestProxyPreservesVisionContent(t *testing.T) {
	content := `[{"type":"text","text":"Read this business card"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aW1hZ2U="}}]`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		require.Equal(t, "qwen3-vl-4b-instruct-fp8", body.Model)
		require.Len(t, body.Messages, 1)
		require.Equal(t, "user", body.Messages[0].Role)
		require.JSONEq(t, content, string(body.Messages[0].Content))
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"Ada Lovelace"}}]}`)
	}))
	defer upstream.Close()
	h := handlerForTest(t, upstream.URL, func() bool { return true })
	w := requestForTest(h, http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":`+content+`}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.JSONEq(t, `{"choices":[{"message":{"content":"Ada Lovelace"}}]}`, w.Body.String())
}

func TestProxyRejectsInvalidAndOversizedRequests(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()
	h := handlerForTest(t, upstream.URL, func() bool { return true })
	for _, body := range []string{``, `null`, `[]`, `{`, `{} {}`, `{"model":"other"}`, `{"model":42}`} {
		w := requestForTest(h, http.MethodPost, "/v1/chat/completions", body)
		require.Equal(t, http.StatusBadRequest, w.Code, body)
		require.Contains(t, w.Body.String(), "invalid_request")
	}
	prefix, suffix := `{"messages":"`, `"}`
	body := prefix + strings.Repeat("x", maxRequestBytes-len(prefix)-len(suffix)) + suffix
	require.Equal(t, http.StatusOK, requestForTest(h, http.MethodPost, "/v1/chat/completions", body).Code)
	w := requestForTest(h, http.MethodPost, "/v1/chat/completions", body+" ")
	require.Equal(t, http.StatusRequestEntityTooLarge, w.Code)
	require.Contains(t, w.Body.String(), "request_too_large")
	require.Equal(t, int32(1), calls.Load())
}

func TestProxyRejectsFifthInFlightRequest(t *testing.T) {
	entered, release := make(chan struct{}, 4), make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		entered <- struct{}{}
		<-release
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()
	unblock := sync.OnceFunc(func() { close(release) })
	defer unblock()
	h := handlerForTest(t, upstream.URL, func() bool { return true })
	results := make(chan int, 4)
	for range 4 {
		go func() { results <- requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`).Code }()
	}
	for range 4 {
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("four requests did not reach upstream")
		}
	}
	w := requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`)
	require.Equal(t, http.StatusServiceUnavailable, w.Code)
	require.Contains(t, w.Body.String(), "vision_busy")
	unblock()
	for range 4 {
		select {
		case status := <-results:
			require.Equal(t, http.StatusOK, status)
		case <-time.After(5 * time.Second):
			t.Fatal("admitted request did not complete")
		}
	}
	require.Equal(t, http.StatusOK, requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`).Code)
}

func TestProxyStreamsBeforeCompletion(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()
	proxy := httptest.NewServer(handlerForTest(t, upstream.URL, func() bool { return true }))
	defer proxy.Close()
	defer close(release)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	require.NoError(t, err)
	require.Equal(t, "data: {\"choices\":[]}\n", line)
}

func TestModelsAndMetricsAreForwardedUnchanged(t *testing.T) {
	metrics := "# TYPE vllm:request_success_total counter\nvllm:request_success_total{model_name=\"qwen3-vl-4b-instruct-fp8\"} 2\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			w.Header().Set("Content-Type", "text/plain; version=0.0.4")
			_, _ = io.WriteString(w, metrics)
			return
		}
		_, _ = io.WriteString(w, `{"data":[{"id":"qwen3-vl-4b-instruct-fp8"}]}`)
	}))
	defer upstream.Close()
	h := handlerForTest(t, upstream.URL, func() bool { return false })
	w := requestForTest(h, http.MethodGet, "/metrics", "")
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, metrics, w.Body.String())
	require.Equal(t, "text/plain; version=0.0.4", w.Header().Get("Content-Type"))
	require.JSONEq(t, `{"data":[{"id":"qwen3-vl-4b-instruct-fp8"}]}`, requestForTest(h, http.MethodGet, "/v1/models", "").Body.String())
	require.Equal(t, http.StatusServiceUnavailable, requestForTest(h, http.MethodGet, "/health", "").Code)
	require.Equal(t, http.StatusOK, requestForTest(h, http.MethodGet, "/healthz", "").Code)
	require.Equal(t, http.StatusServiceUnavailable, requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`).Code)
}

func TestProxyDoesNotFollowRedirectsAndPreservesErrors(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if calls.Load() == 1 {
			http.Redirect(w, r, "/load_lora_adapter", http.StatusTemporaryRedirect)
			return
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"messages is required"}}`)
	}))
	defer upstream.Close()
	h := handlerForTest(t, upstream.URL, func() bool { return true })
	require.Equal(t, http.StatusTemporaryRedirect, requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`).Code)
	require.Equal(t, int32(1), calls.Load())
	w := requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`)
	require.Equal(t, http.StatusBadRequest, w.Code)
	require.JSONEq(t, `{"error":{"message":"messages is required"}}`, w.Body.String())
}

func TestProxyDeadlineAndUnavailable(t *testing.T) {
	h := handlerForTest(t, "http://127.0.0.1:1", func() bool { return true })
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	require.Equal(t, http.StatusGatewayTimeout, w.Code)
	require.Contains(t, w.Body.String(), "vision_timeout")
	w = requestForTest(h, http.MethodPost, "/v1/chat/completions", `{}`)
	require.Equal(t, http.StatusBadGateway, w.Code)
	require.Contains(t, w.Body.String(), "vision_upstream_unavailable")
}

func TestReleaseIdentityEnvironmentAndFile(t *testing.T) {
	values := map[string]string{
		"VISION_RELEASE_VERSION": "release", "VISION_SOURCE_REVISION": "",
		"VISION_ARTIFACT_DIGEST": "digest",
		"VISION_RELEASE_FILE":    "",
	}
	for key := range values {
		t.Setenv(key, "")
	}
	handler := func() http.Handler { return handlerForTest(t, "http://127.0.0.1:1", func() bool { return false }) }
	require.Equal(t, http.StatusServiceUnavailable, requestForTest(handler(), http.MethodGet, "/releasez", "").Code)
	file := filepath.Join(t.TempDir(), "release.json")
	stamp := `{"releaseVersion":"file","sourceRevision":"file-revision","artifactDigest":"file-digest"}`
	require.NoError(t, os.WriteFile(file, []byte(stamp), 0600))
	t.Setenv("VISION_RELEASE_FILE", file)
	require.JSONEq(t, stamp, requestForTest(handler(), http.MethodGet, "/releasez", "").Body.String())
	for key, value := range values {
		if key != "VISION_RELEASE_FILE" {
			t.Setenv(key, value)
		}
	}
	require.JSONEq(t, `{"releaseVersion":"release","artifactDigest":"digest"}`, requestForTest(handler(), http.MethodGet, "/releasez", "").Body.String())
}

func TestEnvironmentDefaultsAndUpstreamValidation(t *testing.T) {
	for _, name := range []string{"VISION_UPSTREAM", "VISION_VLLM_BIN", "PLATFORM_PROJECT_ID"} {
		t.Setenv(name, "")
		require.NoError(t, os.Unsetenv(name))
	}
	cfg, err := configFromEnv()
	require.NoError(t, err)
	require.Equal(t, "vllm", cfg.vllmBin)
	require.Equal(t, upstreamURL, cfg.upstream.String())
	t.Setenv("VISION_UPSTREAM", "http://localhost:9000")
	_, err = configFromEnv()
	require.Error(t, err, "spawned server must not probe a different origin")
	t.Setenv("VISION_VLLM_BIN", "")
	_, err = configFromEnv()
	require.NoError(t, err)
	for _, origin := range []string{"ftp://localhost", "http://user:pass@localhost", "http://localhost/path", "http://localhost?query", "http://localhost#fragment", ":bad"} {
		t.Setenv("VISION_UPSTREAM", origin)
		_, err = configFromEnv()
		require.Error(t, err, origin)
	}
}
