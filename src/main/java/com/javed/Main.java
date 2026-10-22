package com.javed;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

public class Main {

    public static void main(String[] args)
            throws IOException, InterruptedException {

        try (OllamaEmbeddingClient client =
                     new OllamaEmbeddingClient("nomic-embed-text:latest")) {

            InMemorySemanticSearch search =
                    new InMemorySemanticSearch(client);

            // Step 1: create the documents.
            List<InMemorySemanticSearch.Document> documents = List.of(
                    new InMemorySemanticSearch.Document(
                            "D1", "Java Spring Boot"),

                    new InMemorySemanticSearch.Document(
                            "D2", "Spring Framework application"),

                    new InMemorySemanticSearch.Document(
                            "D3", "Banana fruit"),

                    new InMemorySemanticSearch.Document(
                            "D4", "Kubernetes deployment")
            );

            // Step 2: generate embeddings and store them with the documents.
            // Your addDocuments() method does both operations.
            search.addDocuments(documents);

            System.out.println("Documents indexed: " + search.size());
            System.out.println("Embedding dimensions: " + search.dimensions());

            // Step 3: search.
            // search() creates the query embedding and compares cosine scores.
            String query = "Build a Java web application with Spring Boot";
            int topK = 3;

            List<InMemorySemanticSearch.SearchResult> results =
                    search.search(query, topK);

            // Step 4: print the results.
            System.out.println("\nQuery: " + query);
            System.out.println("Top " + topK + " results:");

            for (InMemorySemanticSearch.SearchResult result : results) {
                System.out.printf(
                        Locale.ROOT,
                        "%s | %.4f | %s%n",
                        result.document().id(),
                        result.score(),
                        result.document().text()
                );
            }
        }
    }
}
