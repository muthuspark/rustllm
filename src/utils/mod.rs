//! Utility functions for the Rust-based LLM chat tool

use anyhow::{Context, Result};
use colored::Colorize;
use home::home_dir;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info};

/// Get the default models directory (~/.rustllm/models)
pub fn get_default_models_dir() -> Result<PathBuf> {
    let mut models_dir = home_dir().context("Could not determine home directory")?;
    models_dir.push(".rustllm");
    models_dir.push("models");
    
    // Create the directory if it doesn't exist
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).context("Failed to create models directory")?;
        info!("Created models directory at {:?}", models_dir);
    }
    
    Ok(models_dir)
}

/// Format file size in human-readable format
pub fn format_file_size(size_bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if size_bytes >= GB {
        format!("{:.2} GB", size_bytes as f64 / GB as f64)
    } else if size_bytes >= MB {
        format!("{:.2} MB", size_bytes as f64 / MB as f64)
    } else if size_bytes >= KB {
        format!("{:.2} KB", size_bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", size_bytes)
    }
}

/// Format user and assistant messages with colors
pub fn format_message(role: &str, content: &str) -> String {
    match role.to_lowercase().as_str() {
        "user" => format!("{}: {}", "User".green().bold(), content),
        "assistant" => format!("{}: {}", "Assistant".blue().bold(), content),
        "system" => format!("{}: {}", "System".yellow().bold(), content),
        _ => format!("{}: {}", role.purple().bold(), content),
    }
}

/// Ensure a directory exists, creating it if necessary
pub fn ensure_dir_exists(dir: &Path) -> Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir).with_context(|| format!("Failed to create directory: {:?}", dir))?;
        info!("Created directory: {:?}", dir);
    }
    Ok(())
}

/// Check if a file exists and has a minimum size
pub fn validate_file(path: &Path, min_size: Option<u64>) -> bool {
    if !path.exists() || !path.is_file() {
        return false;
    }
    
    if let Some(min_size) = min_size {
        match fs::metadata(path) {
            Ok(metadata) => metadata.len() >= min_size,
            Err(_) => false,
        }
    } else {
        true
    }
}

/// Parse key=value pairs from a string
pub fn parse_key_value_pairs(input: &str) -> Result<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    
    for pair in input.split(',') {
        let parts: Vec<&str> = pair.splitn(2, '=').collect();
        if parts.len() == 2 {
            pairs.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
        } else {
            anyhow::bail!("Invalid key=value format: {}", pair);
        }
    }
    
    Ok(pairs)
}

/// Format a duration in seconds to a human-readable format
pub fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Get a temporary directory for downloads
pub fn get_temp_dir() -> Result<PathBuf> {
    let mut temp_dir = std::env::temp_dir();
    temp_dir.push("rustllm");
    
    ensure_dir_exists(&temp_dir)?;
    
    Ok(temp_dir)
}

/// Sanitize a filename by removing invalid characters
pub fn sanitize_filename(name: &str) -> String {
    let invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|'];
    
    name.chars()
        .map(|c| if invalid_chars.contains(&c) { '_' } else { c })
        .collect()
}

/// Clean temporary files older than specified days
pub fn clean_temp_files(days: u64) -> Result<()> {
    let temp_dir = get_temp_dir()?;
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(days * 24 * 60 * 60);
    
    for entry in fs::read_dir(temp_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Ok(metadata) = entry.metadata() {
            if let Ok(modified) = metadata.modified() {
                if modified < cutoff {
                    if metadata.is_file() {
                        if let Err(e) = fs::remove_file(&path) {
                            error!("Failed to remove old temp file {:?}: {}", path, e);
                        } else {
                            info!("Removed old temp file: {:?}", path);
                        }
                    } else if metadata.is_dir() {
                        if let Err(e) = fs::remove_dir_all(&path) {
                            error!("Failed to remove old temp directory {:?}: {}", path, e);
                        } else {
                            info!("Removed old temp directory: {:?}", path);
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}