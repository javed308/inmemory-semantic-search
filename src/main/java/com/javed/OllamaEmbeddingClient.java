package com.javed;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class OllamaEmbeddingClient implements EmbeddingClient{

    private final URI endpoint;
    private final String model;
    private final Duration timeout;
    private final HttpClient http;
    private String returnedModel;
    private int requestCount;
    private int inputCount;

    public OllamaEmbeddingClient(String model) {
        this(URI.create("http://localhost:11434/api/embed"), model, Duration.ofMinutes(2));
    }

    // Package-private: lets offline tests use a loopback HTTP server.
    OllamaEmbeddingClient(URI endpoint, String model, Duration timeout) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("Model must not be blank");
        }
        this.model = model;
        this.timeout = Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative() || timeout.isZero()) {
            throw new IllegalArgumentException("Timeout must be positive");
        }
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .followRedirects(HttpClient.Redirect.NEVER)
                .build();
    }


    @Override
    public List<float[]> embed(List<String> texts) throws IOException, InterruptedException {

        if (texts == null || texts.stream().anyMatch(t -> t == null || t.isBlank())) {
            throw new IllegalArgumentException("Input texts must not be null or blank");
        }
        List<String> batch = List.copyOf(texts);
        if (batch.isEmpty()) return List.of();
        String body = Json.stringify(Map.of(
                "model", model, "input", batch, "truncate", false));
        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
        requestCount++;
        inputCount += batch.size();
        HttpResponse<String> response = http.send(
                request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (response.statusCode() != 200) {
            // Do not echo the response body: it could contain private input text.
            throw new IOException("Ollama HTTP " + response.statusCode()
                    + ". Check the model, server logs, and input length.");
        }
        Object parsed;
        try {
            parsed = Json.parse(response.body());
        } catch (IllegalArgumentException e) {
            throw new IOException("Ollama returned invalid JSON", e);
        }
        if (!(parsed instanceof Map<?, ?> root)) {
            throw new IOException("Expected an Ollama JSON object");
        }
        if (!(root.get("model") instanceof String reported) || reported.isBlank()) {
            throw new IOException("Ollama response has no model name");
        }
        if (returnedModel != null && !returnedModel.equals(reported)) {
            throw new IOException("Returned model changed; rebuild with one model");
        }
        if (!(root.get("embeddings") instanceof List<?> rows) || rows.size() != batch.size()) {
            throw new IOException("Embedding count does not match input count");
        }
        List<float[]> vectors = new ArrayList<>();
        int dimensions = -1;
        for (Object row : rows) {
            if (!(row instanceof List<?> values) || values.isEmpty()) {
                throw new IOException("Embedding must be a nonempty number array");
            }
            float[] vector = new float[values.size()];
            for (int i = 0; i < values.size(); i++) {
                if (!(values.get(i) instanceof Number number)) {
                    throw new IOException("Embedding contains a nonnumeric value");
                }
                double value = number.doubleValue();
                if (!Double.isFinite(value) || Math.abs(value) > Float.MAX_VALUE) {
                    throw new IOException("Coordinate is outside finite float range");
                }
                vector[i] = (float) value;
            }
            try {
                VectorMath.requireValid(vector);
            } catch (IllegalArgumentException e) {
                throw new IOException("Invalid embedding: " + e.getMessage(), e);
            }
            if (dimensions < 0) dimensions = vector.length;
            if (vector.length != dimensions) {
                throw new IOException("Embedding dimensions differ within a batch");
            }
            vectors.add(vector);
        }
        returnedModel = reported;
        return vectors;
    }

    public String model() {
        return model;
    }
    public String returnedModel() {
        return returnedModel;
    }
    public int requestCount() {
        return requestCount;
    }
    public int inputCount() {
        return inputCount;
    }

    @Override
    public void close() { http.close(); }

}
