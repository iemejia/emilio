//! BPE tokenizer for Qwen2.5 — extracted from GGUF metadata.
//!
//! Implements GPT-2 style byte-pair encoding:
//!  - Vocab + merges from tokenizer.ggml.tokens / tokenizer.ggml.merges
//!  - UTF-8 byte fallback for unknown bytes
//!  - Special tokens: <|im_start|>, <|im_end|>, <|endoftext|>

use crate::gguf::{GGUFFile, MetaValue};
use std::collections::HashMap;

/// GPT-2 byte-to-unicode mapping.
/// Bytes 33..=126, 161..=172, 174..=255 map to themselves as Unicode chars.
/// Remaining bytes (0..=32, 127..=160, 173) map to 256..=511 range.
fn byte_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut n = 256u32;
    for b in 0u8..=255 {
        let c = match b {
            33..=126 | 161..=172 | 174..=255 => b as u32,
            _ => {
                let ch = n;
                n += 1;
                ch
            }
        };
        map.insert(b, char::from_u32(c).unwrap());
    }
    map
}

fn unicode_to_byte() -> HashMap<char, u8> {
    byte_to_unicode().into_iter().map(|(b, c)| (c, b)).collect()
}

pub struct Tokenizer {
    /// token_id → string
    pub vocab: Vec<String>,
    /// string → token_id
    pub token_to_id: HashMap<String, usize>,
    /// BPE merge priority: (left, right) → rank (lower = merge first)
    pub merges: HashMap<(String, String), usize>,
    /// GPT-2 byte → unicode char mapping
    pub byte_to_char: HashMap<u8, char>,
    /// GPT-2 unicode char → byte mapping
    pub char_to_byte: HashMap<char, u8>,
    /// Special token IDs
    pub bos_id: usize,
    pub eos_id: usize,
    pub im_start_id: Option<usize>,
    pub im_end_id: Option<usize>,
}

impl Tokenizer {
    /// Build tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GGUFFile) -> Option<Self> {
        // Extract vocab
        let tokens_meta = gguf.meta("tokenizer.ggml.tokens")?;
        let vocab: Vec<String> = match tokens_meta {
            MetaValue::Array(arr) => {
                arr.iter().filter_map(|v| {
                    if let MetaValue::Str(s) = v { Some(s.clone()) } else { None }
                }).collect()
            }
            _ => return None,
        };

        // Build reverse map
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i);
        }

        // Extract merges
        let merges_meta = gguf.meta("tokenizer.ggml.merges")?;
        let mut merges = HashMap::new();
        if let MetaValue::Array(arr) = merges_meta {
            for (rank, val) in arr.iter().enumerate() {
                if let MetaValue::Str(s) = val {
                    // Each merge is "left right"
                    if let Some((left, right)) = s.split_once(' ') {
                        merges.insert((left.to_string(), right.to_string()), rank);
                    }
                }
            }
        }

        let bos_id = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(151643) as usize;
        let eos_id = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(151643) as usize;

        let im_start_id = token_to_id.get("<|im_start|>").copied();
        let im_end_id = token_to_id.get("<|im_end|>").copied();

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            byte_to_char: byte_to_unicode(),
            char_to_byte: unicode_to_byte(),
            bos_id,
            eos_id,
            im_start_id,
            im_end_id,
        })
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![];
        }

        // Handle special tokens first — split text around them
        let special_tokens: Vec<(&str, usize)> = [
            ("<|im_start|>", self.im_start_id),
            ("<|im_end|>", self.im_end_id),
            ("<|endoftext|>", Some(self.eos_id)),
        ]
        .iter()
        .filter_map(|(s, id)| id.map(|id| (*s, id)))
        .collect();

        // Split on special tokens
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find earliest special token
            let mut earliest: Option<(&str, usize, usize)> = None;
            for &(tok_str, tok_id) in &special_tokens {
                if let Some(pos) = remaining.find(tok_str) {
                    if earliest.is_none() || pos < earliest.unwrap().2 {
                        earliest = Some((tok_str, tok_id, pos));
                    }
                }
            }

            if let Some((tok_str, tok_id, pos)) = earliest {
                // Encode text before special token
                if pos > 0 {
                    result.extend(self.encode_chunk(&remaining[..pos]));
                }
                result.push(tok_id);
                remaining = &remaining[pos + tok_str.len()..];
            } else {
                result.extend(self.encode_chunk(remaining));
                break;
            }
        }

        result
    }

    /// Encode a text chunk (no special tokens) using BPE.
    fn encode_chunk(&self, text: &str) -> Vec<usize> {
        // Convert bytes to GPT-2 unicode characters
        let unicode_chars: String = text.as_bytes().iter()
            .map(|&b| self.byte_to_char[&b])
            .collect();

        // Start with single-character tokens
        let mut symbols: Vec<String> = unicode_chars.chars()
            .map(|c| c.to_string())
            .collect();

        // Apply BPE merges
        loop {
            let mut best_rank = usize::MAX;
            let mut best_pos = 0;

            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let merged = format!("{}{}", symbols[best_pos], symbols[best_pos + 1]);
            symbols[best_pos] = merged;
            symbols.remove(best_pos + 1);
        }

        // Convert to IDs
        symbols.iter()
            .map(|s| self.token_to_id.get(s).copied().unwrap_or(0))
            .collect()
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < self.vocab.len() {
                let token = &self.vocab[id];
                // Skip special tokens in output
                if token.starts_with("<|") && token.ends_with("|>") {
                    continue;
                }
                // Convert GPT-2 unicode chars back to bytes
                for c in token.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Format a chat prompt with Qwen's ChatML template.
    pub fn encode_chat(&self, user_message: &str) -> Vec<usize> {
        // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        // <|im_start|>user\n{message}<|im_end|>\n
        // <|im_start|>assistant\n
        let prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        self.encode(&prompt)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
