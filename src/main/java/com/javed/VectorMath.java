package com.javed;

public class VectorMath {
    private VectorMath() {}

    public static void requireValid(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector must not be null or empty");
        }
        boolean nonzero = false;
        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("Coordinates must be finite");
            }
            if (value != 0.0f) nonzero = true;
        }
        if (!nonzero) {
            throw new IllegalArgumentException("Zero-vector cosine is undefined");
        }
    }

    public static double cosineSimilarity(float[] a, float[] b) {
        requireValid(a);
        requireValid(b);
        if (a.length != b.length) {
            throw new IllegalArgumentException("Dimensions must match");
        }
        double dot = 0.0, sumA = 0.0, sumB = 0.0;
        for (int i = 0; i < a.length; i++) {
            double x = a[i], y = b[i];
            dot += x * y;
            sumA += x * x;
            sumB += y * y;
        }
        double score = dot / (Math.sqrt(sumA) * Math.sqrt(sumB));
        // Restrict tiny floating-point overshoots to the mathematical range.
        double clamped = Math.max(-1.0, Math.min(1.0, score));
        return clamped == 0.0 ? 0.0 : clamped;
    }
}
