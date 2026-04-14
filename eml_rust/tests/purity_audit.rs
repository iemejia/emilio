//! EML Purity Audit — automated enforcement that no raw transcendental
//! or arithmetic operator bypasses the `eml_ops` module.
//!
//! This test scans every `.rs` source file and flags raw `.exp()`, `.ln()`,
//! `.cos()`, `.sin()`, `.powf()` calls outside of allowed zones.
//!
//! Allowed zones (files that MAY contain raw calls):
//!   - eml_ops.rs           — the canonical EML primitive implementations
//!   - autoeml_kernel.rs    — the c_exp()/c_ln() counted wrappers + inner loop
//!   - autoeml_reference.rs — reference (non-EML) implementation for comparison
//!
//! Within audited files, lines tagged with `// EML_AUDIT:OK` are whitelisted.
//! This includes:
//!   - Matmul inner loop real-exp bypass (mathematically justified)
//!   - audited_exp()/audited_ln() counting wrappers
//!   - audited_matmul_cse() counting wrappers
//!
//! Run: `cargo test --test purity_audit`

use std::fs;
use std::path::Path;

/// Files that are EXEMPT from audit (they define the primitives themselves).
const EXEMPT_FILES: &[&str] = &[
    "eml_ops.rs",           // canonical EML implementations (raw .exp()/.ln() IS the primitive)
    "autoeml_kernel.rs",    // c_exp()/c_ln() counted wrappers + optimized inner loop
    "autoeml_reference.rs", // non-EML reference implementation (for comparison only)
];

/// Patterns that indicate raw transcendental / bypass usage.
const FORBIDDEN_PATTERNS: &[&str] = &[
    ".exp()",
    ".ln()",
    ".cos()",
    ".sin()",
    ".powf(",
    ".powi(",
    ".tan()",
    ".asin(",
    ".acos(",
    ".atan(",
    ".sinh(",
    ".cosh(",
    ".tanh(",
];

/// Marker comment that whitelists a line (must appear on the SAME line).
const AUDIT_OK_MARKER: &str = "EML_AUDIT:OK";

/// Additional patterns that are false positives (not actual raw calls).
const FALSE_POSITIVE_PATTERNS: &[&str] = &[
    "expect(",      // .expect() is a Rust Option/Result method, not math
    "export",       // export contains "exp"
    "explain",      // text containing "exp"
    "expression",   // text containing "exp"
    "explicit",     // text containing "exp"
    "\"",           // string literals (comments about .exp() etc.)
    "///",          // doc comments
    "//!",          // module-level doc comments
];

#[test]
fn no_raw_transcendentals_outside_allowed_zones() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(src_dir.exists(), "src/ directory not found at {:?}", src_dir);

    let mut violations: Vec<String> = Vec::new();

    for entry in fs::read_dir(&src_dir).expect("Failed to read src/") {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();

        // Only check .rs files
        if path.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy();

        // Skip exempt files
        if EXEMPT_FILES.iter().any(|&f| filename == f) {
            continue;
        }

        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));

        for (line_no, line) in content.lines().enumerate() {
            let line_num = line_no + 1; // 1-indexed
            let trimmed = line.trim();

            // Skip pure comment lines
            if trimmed.starts_with("//") {
                continue;
            }

            // Skip lines with the audit-OK marker
            if line.contains(AUDIT_OK_MARKER) {
                continue;
            }

            // Check each forbidden pattern
            for &pattern in FORBIDDEN_PATTERNS {
                if line.contains(pattern) {
                    // Filter false positives
                    let is_false_positive = FALSE_POSITIVE_PATTERNS.iter().any(|&fp| {
                        // For .expect(), check that the ".exp" is actually ".expect"
                        if fp == "expect(" && pattern == ".exp()" {
                            return line.contains(".expect(");
                        }
                        // For string/comment patterns, only if the pattern
                        // appears INSIDE a string literal or comment suffix
                        if fp == "\"" {
                            // Check if pattern appears only within a string
                            // Simple heuristic: if the line has quotes before the pattern
                            if let Some(pat_pos) = line.find(pattern) {
                                let before = &line[..pat_pos];
                                let quote_count = before.matches('"').count();
                                return quote_count % 2 == 1; // odd = inside string
                            }
                            return false;
                        }
                        if fp == "///" || fp == "//!" {
                            return trimmed.starts_with(fp);
                        }
                        false
                    });

                    if is_false_positive {
                        continue;
                    }

                    // Special case: .expect() is not .exp()
                    if pattern == ".exp()" {
                        if let Some(pos) = line.find(".exp(") {
                            // Check if it's actually .expect(
                            if line[pos..].starts_with(".expect(") {
                                continue;
                            }
                        }
                    }

                    violations.push(format!(
                        "  {}:{}: `{}` found\n    {}",
                        filename, line_num, pattern, trimmed
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        panic!(
            "\n╔══════════════════════════════════════════════════════════════╗\n\
             ║  EML PURITY VIOLATION — raw transcendental found!          ║\n\
             ╚══════════════════════════════════════════════════════════════╝\n\n\
             {} violation(s) found:\n\n{}\n\n\
             Fix: route through eml_ops (eml_exp, eml_ln, eml_add, etc.)\n\
             Or add `// EML_AUDIT:OK — <justification>` to whitelist.\n",
            violations.len(),
            violations.join("\n")
        );
    }
}

#[test]
fn audit_ok_markers_have_justification() {
    // Every EML_AUDIT:OK line must include a justification after the marker
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut bare_markers: Vec<String> = Vec::new();

    for entry in fs::read_dir(&src_dir).expect("Failed to read src/") {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();

        if path.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy();
        let content = fs::read_to_string(&path).unwrap();

        for (line_no, line) in content.lines().enumerate() {
            if line.contains(AUDIT_OK_MARKER) {
                // Must have text after the marker (justification)
                if let Some(pos) = line.find(AUDIT_OK_MARKER) {
                    let after = line[pos + AUDIT_OK_MARKER.len()..].trim();
                    // Must start with em-dash or hyphen followed by text
                    if after.is_empty()
                        || (!after.starts_with("— ")
                            && !after.starts_with("- ")
                            && !after.starts_with("\u{2014} "))
                    {
                        bare_markers.push(format!(
                            "  {}:{}: marker without justification",
                            filename,
                            line_no + 1
                        ));
                    }
                }
            }
        }
    }

    if !bare_markers.is_empty() {
        panic!(
            "\nEML_AUDIT:OK markers without justification:\n{}\n\n\
             Fix: add `// EML_AUDIT:OK — <reason>` with an explanation.\n",
            bare_markers.join("\n")
        );
    }
}
