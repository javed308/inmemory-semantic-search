package com.javed;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;

@FunctionalInterface
public interface EmbeddingClient extends Closeable {

    List<float[]> embed(List<String> texts) throws IOException, InterruptedException;

    @Override
    default void close() {}
}
