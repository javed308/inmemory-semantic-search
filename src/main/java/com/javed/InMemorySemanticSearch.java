package com.javed;

import java.io.IOException;
import java.util.*;

public class InMemorySemanticSearch {

    public record Document(String id, String text) {
        public Document {
            if (id == null || id.isBlank()) {
                throw new IllegalArgumentException("Document ID must not be blank");
            }
            if (text == null || text.isBlank()) {
                throw new IllegalArgumentException("Document text must not be blank");
            }
        }
    }

    public record SearchResult(Document document, double score) {}

    private final EmbeddingClient embeddingClient;
    private final List<Document> documents = new ArrayList<>();
    private final List<float[]> embeddings = new ArrayList<>();
    private int dimensions = -1;

    public InMemorySemanticSearch(EmbeddingClient embeddingClient) {
        this.embeddingClient = Objects.requireNonNull(embeddingClient, "embeddingClient");
    }

    /** Append a batch. Duplicate IDs are rejected, not silently replaced. */
    public void addDocuments(List<Document> input)
            throws IOException, InterruptedException {
        if (input == null || input.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Documents must not be null");
        }
        List<Document> batch = List.copyOf(input);
        if (batch.isEmpty()) return;

        Set<String> ids = new HashSet<>();
        for (Document document : documents) ids.add(document.id());
        for (Document document : batch) {
            if (!ids.add(document.id())) {
                throw new IllegalArgumentException("Duplicate document ID: " + document.id());
            }
        }

        // Only NEW documents go to the model. Existing vectors stay in memory.
        List<String> texts = batch.stream().map(Document::text).toList();
        List<float[]> generated = embeddingClient.embed(texts);
        List<float[]> checked = copyAndValidate(generated, batch.size(), dimensions);

        // Commit only after every vector passes validation.
        // Invariant: documents.get(i) always belongs to embeddings.get(i).
        documents.addAll(batch);
        embeddings.addAll(checked);
        dimensions = embeddings.getFirst().length;
    }

    public void addDocument(Document document) throws IOException, InterruptedException {
        if (document == null) throw new IllegalArgumentException("Document must not be null");
        addDocuments(List.of(document));
    }

    /** Return up to k results. Negative cosine scores are NOT removed by default. */
    public List<SearchResult> search(String query, int k)
            throws IOException, InterruptedException {
        return search(query, k, -1.0);
    }

    /** Optional inclusive threshold: a score must be >= minScore. */
    public List<SearchResult> search(String query, int k, double minScore)
            throws IOException, InterruptedException {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("Query must not be blank");
        }
        if (k <= 0) throw new IllegalArgumentException("K must be positive");
        if (!Double.isFinite(minScore) || minScore < -1.0 || minScore > 1.0) {
            throw new IllegalArgumentException("Minimum score must be between -1 and 1");
        }
        if (documents.isEmpty()) return List.of(); // No API call needed.

        // One query embedding. Do NOT embed the documents again here.
        float[] queryVector = copyAndValidate(
                embeddingClient.embed(List.of(query)), 1, dimensions).getFirst();

        List<SearchResult> scored = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            double score = VectorMath.cosineSimilarity(queryVector, embeddings.get(i));
            if (score >= minScore) {
                scored.add(new SearchResult(documents.get(i), score));
            }
        }
        scored.sort(Comparator.comparingDouble(SearchResult::score).reversed()
                .thenComparing(result -> result.document().id()));
        return List.copyOf(scored.subList(0, Math.min(k, scored.size())));
    }

    /** Remove both items at the same position to keep the two lists aligned. */
    public boolean removeDocument(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Document ID must not be blank");
        }
        for (int i = 0; i < documents.size(); i++) {
            if (documents.get(i).id().equals(id)) {
                documents.remove(i);
                embeddings.remove(i);
                if (documents.isEmpty()) dimensions = -1;
                return true;
            }
        }
        return false;
    }

    /** Read-only snapshot of document metadata; vectors remain private. */
    public List<Document> documents() { return List.copyOf(documents); }

    public int size() { return documents.size(); }
    public int dimensions() { return dimensions < 0 ? 0 : dimensions; }

    public void clear() {
        documents.clear();
        embeddings.clear();
        dimensions = -1;
    }

    private static List<float[]> copyAndValidate(
            List<float[]> vectors, int expectedCount, int expectedDimensions) throws IOException {
        if (vectors == null || vectors.size() != expectedCount) {
            throw new IOException("Embedding count does not match input count");
        }
        List<float[]> copies = new ArrayList<>(expectedCount);
        int width = expectedDimensions;
        for (float[] vector : vectors) {
            try {
                VectorMath.requireValid(vector);
            } catch (IllegalArgumentException e) {
                throw new IOException("Invalid embedding: " + e.getMessage(), e);
            }
            if (width < 0) width = vector.length;
            if (vector.length != width) {
                throw new IOException("Embedding dimensions changed; rebuild the index");
            }
            copies.add(vector.clone()); // Do not store a caller-owned mutable array.
        }
        return copies;
    }


}
