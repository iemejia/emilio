//! emilio — EML inference engine
//!
//! Usage: emilio <model.gguf> [--explore | --generate <text> | --chat <message>]

use eml_rust_core::gguf::GGUFFile;
use eml_rust_core::emilio::*;
use eml_rust_core::tokenizer::Tokenizer;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: emilio <model.gguf> [--explore | --generate <text> | --chat <message>]");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  emilio model.gguf --explore");
        eprintln!("  emilio model.gguf --generate \"Hello world\"");
        eprintln!("  emilio model.gguf --chat \"What is 2+2?\"");
        eprintln!("  emilio model.gguf --tokens \"1,2,3\"     (raw token IDs)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("--explore");

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  emilio — EML inference engine                          ║");
    println!("║  Every result flows through eml(x,y) = exp(x) - ln(y)  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Parse GGUF ─────────────────────────────────────────────────
    println!("Loading GGUF: {model_path}");
    let t0 = Instant::now();
    let gguf = match GGUFFile::parse(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing GGUF: {e}");
            std::process::exit(1);
        }
    };
    println!("  Parsed header in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ── Build tokenizer ────────────────────────────────────────────
    let tokenizer = Tokenizer::from_gguf(&gguf);
    if let Some(ref tok) = tokenizer {
        println!("  Tokenizer: {} tokens, {} merges",
            tok.vocab_size(), tok.merges.len());
    }
    println!();

    match mode {
        "--explore" => explore(&gguf),
        "--generate" => {
            let text = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let prompt_ids = tok.encode(text);
            println!("  Tokenized: \"{}\" → {} tokens: {:?}",
                text, prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            generate(&gguf, &prompt_ids, tok);
        }
        "--chat" => {
            let msg = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let prompt_ids = tok.encode_chat(msg);
            println!("  Chat prompt: \"{msg}\"");
            println!("  Tokenized to {} tokens: {:?}...",
                prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            generate(&gguf, &prompt_ids, tok);
        }
        "--tokens" => {
            let tok_str = args.get(3).map(|s| s.as_str()).unwrap_or("1,2,3");
            let prompt: Vec<usize> = tok_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let tok = tokenizer.as_ref();
            generate_raw(&gguf, &prompt, tok);
        }
        _ => {
            eprintln!("Unknown mode: {mode}");
            std::process::exit(1);
        }
    }
}

fn explore(gguf: &GGUFFile) {
    println!("═══════════════════════════════════════════════════");
    println!("  MODEL METADATA");
    println!("═══════════════════════════════════════════════════");
    gguf.print_summary();

    let config = QwenConfig::from_gguf(gguf);
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  QWEN2 ARCHITECTURE → EML MAPPING");
    println!("═══════════════════════════════════════════════════");
    config.print();

    // Show tensor table
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  TENSOR MAP ({} tensors)", gguf.tensors.len());
    println!("═══════════════════════════════════════════════════");
    println!("  {:50} {:>12} {:>8} {:>10}",
        "Name", "Shape", "Type", "Size");
    println!("  {}", "─".repeat(84));

    for t in &gguf.tensors {
        let shape = t.dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let size_mb = t.byte_size() as f64 / 1e6;
        println!("  {:50} {:>12} {:>8?} {:>8.2}MB",
            t.name, shape, t.dtype, size_mb);
    }

    // EML complexity analysis
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  EML COMPLEXITY ANALYSIS");
    println!("═══════════════════════════════════════════════════");

    let d = config.d_model;
    let d_ff = config.d_ff;
    let n_heads = config.n_heads;
    let n_kv = config.n_kv_heads;
    let d_head = config.d_head;
    let v = config.vocab_size;
    let n_layers = config.n_layers;

    println!();
    println!("  Per-layer EML ops (seq_len=T, CSE-optimized):");
    println!();

    // QKV projections: 3 matmuls
    let qkv_naive = 3 * d * (n_heads * d_head + 2 * n_kv * d_head); // per token, K=d
    let qkv_cse_per_tok = n_heads * d_head * d + 2 * n_kv * d_head * d; // exp calls
    println!("    QKV projection:  T × {} exp (CSE) vs T × {} (naive)",
        qkv_cse_per_tok / d, qkv_naive);

    // Attention: T² × d_head dot products per head
    let attn_per_token = n_heads * d_head * 3; // per score: d_head muls
    println!("    Attention:       T² × {} transcendentals per head × {n_heads} heads",
        d_head * 3);

    // FFN: gate + up + down
    let ffn_matmul = 2 * d * d_ff + d_ff * d;
    println!("    SwiGLU FFN:      T × {} exp (3 matmuls, CSE)", ffn_matmul / d);

    // RMSNorm: per token
    let rms_per_tok = d + 3; // d ln's + div + sqrt + scale
    println!("    RMSNorm:         T × ~{rms_per_tok} transcendentals (CSE on ln(x))");

    // LM head
    let lm_head = d * v;
    println!("    LM head:         T × {} exp (CSE)", v);

    println!();
    println!("  Total layers: {n_layers} × above, + final norm + LM head");
    println!();

    // Dequant a small tensor to show it works
    if let Some(info) = gguf.tensor_info("token_embd.weight") {
        println!("═══════════════════════════════════════════════════");
        println!("  DEQUANTIZATION SPOT CHECK");
        println!("═══════════════════════════════════════════════════");
        let t0 = Instant::now();
        match gguf.load_tensor_f64(info) {
            Ok(data) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let n = data.len();
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mean: f64 = data.iter().sum::<f64>() / n as f64;
                println!("  token_embd.weight: {} elements, dequantized in {ms:.1}ms", n);
                println!("    range: [{min:.6}, {max:.6}]");
                println!("    mean:  {mean:.6}");
                println!("    first 8: {:?}", &data[..8.min(n)]);
            }
            Err(e) => println!("  Error loading tensor: {e}"),
        }
    }
}

fn generate(gguf: &GGUFFile, prompt: &[usize], tok: &Tokenizer) {
    println!("Loading model weights...");
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Weights loaded in {load_ms:.0}ms");

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML inference, KV-cached)...");
    println!("  Prompt: {} tokens", prompt.len());
    println!("  Config: {} layers, {} heads, d_model={}",
        weights.config.n_layers, weights.config.n_heads, weights.config.d_model);

    let max_new = 16;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, &weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    let prompt_text = tok.decode(prompt);
    let generated_text = tok.decode(generated);

    println!();
    println!("  ┌─────────────────────────────────────────────────");
    println!("  │ Prompt:    \"{}\"", prompt_text);
    println!("  │ Generated: \"{}\"", generated_text);
    println!("  │ Token IDs: {:?}", generated);
    println!("  └─────────────────────────────────────────────────");
    println!("  Time:      {gen_s:.2}s ({:.4} tokens/s)",
        generated.len() as f64 / gen_s);
    println!();
    println!("  Every multiply was exp(ln(a) + ln(b)).");
    println!("  Every division was exp(ln(a) - ln(b)).");
}

fn generate_raw(gguf: &GGUFFile, prompt: &[usize], tok: Option<&Tokenizer>) {
    println!("Loading model weights...");
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Weights loaded in {load_ms:.0}ms");

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML, KV-cached)...");
    println!("  Prompt token IDs: {:?}", prompt);

    let max_new = 8;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, &weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    println!();
    println!("  Generated IDs: {:?}", generated);
    if let Some(tok) = tok {
        println!("  Decoded: \"{}\"", tok.decode(generated));
    }
    println!("  Time: {gen_s:.2}s ({:.4} tokens/s)",
        generated.len() as f64 / gen_s);
}
