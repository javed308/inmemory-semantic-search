package com.javed;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Json {
    private Json() {}

    public static Object parse(String text) {
        Parser p = new Parser(text);
        Object result = p.value(0);
        p.space();
        if (p.pos != text.length()) throw p.error("Unexpected trailing content");
        return result;
    }

    public static String stringify(Object value) {
        StringBuilder out = new StringBuilder();
        write(value, out);
        return out.toString();
    }

    private static void write(Object value, StringBuilder out) {
        if (value == null) out.append("null");
        else if (value instanceof String s) quote(s, out);
        else if (value instanceof Boolean b) out.append(b);
        else if (value instanceof Number n) {
            if (!Double.isFinite(n.doubleValue()))
                throw new IllegalArgumentException("JSON numbers must be finite");
            out.append(n);
        } else if (value instanceof Map<?, ?> map) {
            out.append('{');
            boolean first = true;
            for (var e : map.entrySet()) {
                if (!(e.getKey() instanceof String key))
                    throw new IllegalArgumentException("JSON keys must be strings");
                if (!first) out.append(',');
                first = false;
                quote(key, out); out.append(':'); write(e.getValue(), out);
            }
            out.append('}');
        } else if (value instanceof Iterable<?> items) {
            out.append('[');
            boolean first = true;
            for (Object item : items) {
                if (!first) out.append(',');
                first = false;
                write(item, out);
            }
            out.append(']');
        } else if (value instanceof double[] numbers) {
            out.append('[');
            for (int i = 0; i < numbers.length; i++) {
                if (i > 0) out.append(',');
                write(numbers[i], out);
            }
            out.append(']');
        } else throw new IllegalArgumentException("Unsupported JSON value: " + value.getClass());
    }

    private static void quote(String text, StringBuilder out) {
        out.append('"');
        for (char c : text.toCharArray()) {
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 32) out.append(String.format("\\u%04x", (int) c));
                    else out.append(c);
                }
            }
        }
        out.append('"');
    }

    private static final class Parser {
        private final String text;
        private int pos;
        Parser(String text) { this.text = java.util.Objects.requireNonNull(text); }
        IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at JSON position " + pos);
        }
        void space() {
            while (pos < text.length() && " \n\r\t".indexOf(text.charAt(pos)) >= 0) pos++;
        }
        boolean take(char c) {
            if (pos < text.length() && text.charAt(pos) == c) { pos++; return true; }
            return false;
        }
        void require(char c) { if (!take(c)) throw error("Expected '" + c + "'"); }
        Object value(int depth) {
            if (depth > 100) throw error("JSON nesting is too deep");
            space();
            if (pos >= text.length()) throw error("Unexpected end");
            char c = text.charAt(pos);
            if (c == '"') return string();
            if (take('{')) {
                Map<String, Object> map = new LinkedHashMap<>();
                space(); if (take('}')) return map;
                do {
                    space(); String key = string(); space(); require(':');
                    if (map.containsKey(key)) throw error("Duplicate key");
                    map.put(key, value(depth + 1)); space();
                    if (take('}')) return map;
                    require(',');
                } while (true);
            }
            if (take('[')) {
                List<Object> list = new ArrayList<>();
                space(); if (take(']')) return list;
                do {
                    list.add(value(depth + 1)); space();
                    if (take(']')) return list;
                    require(',');
                } while (true);
            }
            for (String literal : List.of("true", "false", "null")) {
                if (text.startsWith(literal, pos)) {
                    pos += literal.length();
                    return switch (literal) { case "true" -> true; case "false" -> false; default -> null; };
                }
            }
            return number();
        }
        String string() {
            require('"'); StringBuilder out = new StringBuilder();
            while (pos < text.length()) {
                char c = text.charAt(pos++);
                if (c == '"') return out.toString();
                if (c < 32) throw error("Unescaped control character");
                if (c != '\\') { out.append(c); continue; }
                if (pos >= text.length()) throw error("Incomplete escape");
                char e = text.charAt(pos++);
                switch (e) {
                    case '"', '\\', '/' -> out.append(e);
                    case 'n' -> out.append('\n'); case 'r' -> out.append('\r');
                    case 't' -> out.append('\t'); case 'b' -> out.append('\b'); case 'f' -> out.append('\f');
                    case 'u' -> {
                        if (pos + 4 > text.length()) throw error("Incomplete Unicode escape");
                        int code = 0;
                        for (int i = 0; i < 4; i++) {
                            int digit = Character.digit(text.charAt(pos++), 16);
                            if (digit < 0) throw error("Invalid Unicode escape");
                            code = code * 16 + digit;
                        }
                        out.append((char) code);
                    }
                    default -> throw error("Invalid escape");
                }
            }
            throw error("Unclosed string");
        }
        Double number() {
            int start = pos;
            take('-');
            if (take('0')) {
                if (pos < text.length() && digit(text.charAt(pos))) throw error("Leading zero");
            } else {
                int before = pos;
                while (pos < text.length() && digit(text.charAt(pos))) pos++;
                if (before == pos) throw error("Expected number");
            }
            if (take('.')) digits();
            if (take('e') || take('E')) { if (!take('+')) take('-'); digits(); }
            double n;
            try { n = Double.parseDouble(text.substring(start, pos)); }
            catch (NumberFormatException e) { throw error("Invalid number"); }
            if (!Double.isFinite(n)) throw error("Number outside finite double range");
            return n;
        }
        void digits() {
            int before = pos;
            while (pos < text.length() && digit(text.charAt(pos))) pos++;
            if (before == pos) throw error("Expected digits");
        }
        boolean digit(char c) { return c >= '0' && c <= '9'; }
    }
}
