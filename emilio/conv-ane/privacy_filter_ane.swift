// privacy_filter_ane.swift
// ANE inference runner for OpenAI Privacy Filter (token classification for PII detection)
//
// Unlike the autoregressive Qwen runner, this is a single-pass bidirectional encoder.
// No KV cache needed — feed the entire sequence and get per-token labels back.
//
// Build:
//   swiftc -O -o privacy_filter_ane privacy_filter_ane.swift \
//     -framework CoreML -framework Foundation
//
// Usage:
//   ./privacy_filter_ane PrivacyFilterANE_512.mlpackage PrivacyFilterANE_512_meta.json \
//     "My name is Alice Smith and my email is alice@example.com"

import CoreML
import Foundation

// MARK: - Label Definitions

let id2label: [Int: String] = [
    0: "O",
    1: "B-account_number", 2: "I-account_number", 3: "E-account_number", 4: "S-account_number",
    5: "B-private_address", 6: "I-private_address", 7: "E-private_address", 8: "S-private_address",
    9: "B-private_date", 10: "I-private_date", 11: "E-private_date", 12: "S-private_date",
    13: "B-private_email", 14: "I-private_email", 15: "E-private_email", 16: "S-private_email",
    17: "B-private_person", 18: "I-private_person", 19: "E-private_person", 20: "S-private_person",
    21: "B-private_phone", 22: "I-private_phone", 23: "E-private_phone", 24: "S-private_phone",
    25: "B-private_url", 26: "I-private_url", 27: "E-private_url", 28: "S-private_url",
    29: "B-secret", 30: "I-secret", 31: "E-secret", 32: "S-secret",
]

let numLabels = 33

// MARK: - Metadata

struct ModelMeta: Codable {
    let model: String
    let task: String
    let num_labels: Int
    let d_model: Int
    let n_heads: Int
    let n_kv_heads: Int
    let head_dim: Int
    let n_layers: Int
    let n_experts: Int
    let experts_per_token: Int
    let intermediate_size: Int
    let sliding_window: Int
    let max_seq_len: Int
    let vocab_size: Int
    let dtype: String
    let quantized: Bool
}

// MARK: - Simple BPE Tokenizer (loads HF tokenizer.json)

class SimpleTokenizer {
    let vocab: [String: Int]
    let merges: [(String, String)]
    let padId: Int

    init(tokenizerJsonPath: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: tokenizerJsonPath))
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Build vocab from tokenizer.json "model.vocab"
        let model = json["model"] as! [String: Any]
        let vocabDict = model["vocab"] as! [String: Int]
        self.vocab = vocabDict

        // Merges
        let mergesList = model["merges"] as? [String] ?? []
        self.merges = mergesList.map { line in
            let parts = line.split(separator: " ", maxSplits: 1)
            return (String(parts[0]), String(parts[1]))
        }

        self.padId = vocabDict["<|endoftext|>"] ?? 199999
    }

    func encode(_ text: String) -> [Int] {
        // Simplified: encode each byte as a token (fallback for BPE)
        // For production, implement full BPE merge algorithm
        var tokens: [Int] = []
        for char in text.utf8 {
            let key = String(format: "%c", char)
            if let id = vocab[key] {
                tokens.append(id)
            } else {
                // Try byte fallback token
                let byteKey = String(format: "bytes:\\x%02x", char)
                if let id = vocab[byteKey] {
                    tokens.append(id)
                } else {
                    tokens.append(0)  // unknown
                }
            }
        }
        return tokens
    }

    func padToLength(_ tokens: [Int], length: Int) -> [Int] {
        if tokens.count >= length {
            return Array(tokens.prefix(length))
        }
        return tokens + Array(repeating: padId, count: length - tokens.count)
    }
}

// MARK: - ANE Inference

func runInference(modelPath: String, metaPath: String, text: String) throws {
    // Load metadata
    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta = try JSONDecoder().decode(ModelMeta.self, from: metaData)
    let seqLen = meta.max_seq_len

    print("Model: \(meta.model)")
    print("Seq len: \(seqLen), dtype: \(meta.dtype)")
    print("Input: \"\(text)\"")
    print()

    // Load CoreML model
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let modelURL = URL(fileURLWithPath: modelPath)
    let model = try MLModel(contentsOf: MLModel.compileModel(at: modelURL), configuration: config)

    // Tokenize (simplified — for production use the full BPE tokenizer)
    // For now, use byte-level encoding as a placeholder
    let tokenIds: [Int32] = text.utf8.map { Int32($0) }
    let paddedTokens = Array(tokenIds.prefix(seqLen)) +
        Array(repeating: Int32(199999), count: max(0, seqLen - tokenIds.count))

    // Build RoPE tables
    let dHalf = meta.head_dim / 2
    var ropeCos = [Float16](repeating: 0, count: seqLen * dHalf)
    var ropeSin = [Float16](repeating: 0, count: seqLen * dHalf)

    let ropeTheta: Float = 150000.0
    for pos in 0..<seqLen {
        for i in 0..<dHalf {
            let freq = 1.0 / pow(ropeTheta, Float(2 * i) / Float(meta.head_dim))
            let angle = Float(pos) * freq
            ropeCos[pos * dHalf + i] = Float16(cos(angle))
            ropeSin[pos * dHalf + i] = Float16(sin(angle))
        }
    }

    // Build banded attention mask (with padding mask)
    let window = meta.sliding_window
    var attnMask = [Float16](repeating: Float16(-10000.0), count: seqLen * seqLen)
    for i in 0..<seqLen {
        let left = max(0, i - window)
        let right = min(seqLen, i + window + 1)
        for j in left..<right {
            // Only attend to real (non-padding) tokens
            if j < actualTokenCount {
                attnMask[i * seqLen + j] = Float16(0.0)
            }
        }
    }

    // Create MLMultiArrays
    let tokenArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
    for i in 0..<seqLen {
        tokenArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: paddedTokens[i])
    }

    let cosArray = try MLMultiArray(shape: [NSNumber(value: seqLen), NSNumber(value: dHalf)],
                                     dataType: .float16)
    let sinArray = try MLMultiArray(shape: [NSNumber(value: seqLen), NSNumber(value: dHalf)],
                                     dataType: .float16)
    for i in 0..<(seqLen * dHalf) {
        cosArray[i] = NSNumber(value: Float(ropeCos[i]))
        sinArray[i] = NSNumber(value: Float(ropeSin[i]))
    }

    let maskArray = try MLMultiArray(
        shape: [NSNumber(value: seqLen), NSNumber(value: seqLen)], dataType: .float16)
    for i in 0..<(seqLen * seqLen) {
        maskArray[i] = NSNumber(value: Float(attnMask[i]))
    }

    // Run inference
    let startTime = CFAbsoluteTimeGetCurrent()
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "token_ids": MLFeatureValue(multiArray: tokenArray),
        "rope_cos": MLFeatureValue(multiArray: cosArray),
        "rope_sin": MLFeatureValue(multiArray: sinArray),
        "attn_mask": MLFeatureValue(multiArray: maskArray),
    ])

    let output = try model.prediction(from: input)
    let elapsed = CFAbsoluteTimeGetCurrent() - startTime

    // Parse output logits: (1, NUM_LABELS, T)
    guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
        print("ERROR: No logits output")
        return
    }

    let actualTokenCount = min(tokenIds.count, seqLen)
    print("Results (\(String(format: "%.1f", elapsed * 1000)) ms):")
    print(String(repeating: "-", count: 60))

    // Decode predictions (argmax per token)
    var spans: [(label: String, start: Int, end: Int)] = []
    var currentLabel: String? = nil
    var spanStart = 0

    for t in 0..<actualTokenCount {
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        for c in 0..<numLabels {
            let val = logitsArray[[0, NSNumber(value: c), NSNumber(value: t)] as [NSNumber]].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = c
            }
        }
        let label = id2label[maxIdx] ?? "O"

        if label == "O" {
            if let cl = currentLabel {
                spans.append((label: cl, start: spanStart, end: t))
                currentLabel = nil
            }
        } else if label.hasPrefix("B-") || label.hasPrefix("S-") {
            if let cl = currentLabel {
                spans.append((label: cl, start: spanStart, end: t))
            }
            let category = String(label.dropFirst(2))
            currentLabel = category
            spanStart = t
            if label.hasPrefix("S-") {
                spans.append((label: category, start: t, end: t + 1))
                currentLabel = nil
            }
        } else if label.hasPrefix("E-") {
            if let cl = currentLabel {
                spans.append((label: cl, start: spanStart, end: t + 1))
                currentLabel = nil
            }
        }
        // I- continues the current span
    }
    if let cl = currentLabel {
        spans.append((label: cl, start: spanStart, end: actualTokenCount))
    }

    // Print detected spans
    if spans.isEmpty {
        print("  No PII detected.")
    } else {
        let textBytes = Array(text.utf8)
        for span in spans {
            let startByte = min(span.start, textBytes.count)
            let endByte = min(span.end, textBytes.count)
            let spanText = String(bytes: Array(textBytes[startByte..<endByte]), encoding: .utf8) ?? "?"
            print("  [\(span.label)] \"\(spanText)\" (tokens \(span.start)-\(span.end))")
        }
    }

    print(String(repeating: "-", count: 60))
    print("Inference time: \(String(format: "%.1f", elapsed * 1000)) ms")
    print("Throughput: \(String(format: "%.0f", Double(actualTokenCount) / elapsed)) tok/s")
}

// MARK: - Main

guard CommandLine.arguments.count >= 4 else {
    print("Usage: privacy_filter_ane <model.mlpackage> <meta.json> <text>")
    print()
    print("Run OpenAI Privacy Filter on Apple Neural Engine for PII detection.")
    print()
    print("Example:")
    print("  ./privacy_filter_ane PrivacyFilterANE_512.mlpackage \\")
    print("    PrivacyFilterANE_512_meta.json \\")
    print("    \"My name is Alice Smith and my email is alice@example.com\"")
    exit(1)
}

let modelPath = CommandLine.arguments[1]
let metaPath = CommandLine.arguments[2]
let text = CommandLine.arguments[3..<CommandLine.arguments.count].joined(separator: " ")

do {
    try runInference(modelPath: modelPath, metaPath: metaPath, text: text)
} catch {
    print("ERROR: \(error)")
    exit(1)
}
