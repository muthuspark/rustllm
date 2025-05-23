use anyhow::{Context, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tracing::{debug, error, info};

// Model information structure
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub filename: String,
    pub download_url: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub description: Option<String>,
}

/// Get information about a model by name or URL
pub async fn get_model_info(model_identifier: &str) -> Result<ModelInfo> {
    // This is a simplified implementation - in a real-world scenario, 
    // we would query an API to get model information
    
    // For now, we'll handle a few known models or assume it's a direct URL
    let model_info = if model_identifier.starts_with("http") {
        // Direct URL
        let url = model_identifier;
        let filename = url
            .split('/')
            .last()
            .context("Invalid URL format")?
            .to_string();
        
        ModelInfo {
            name: filename.clone(),
            filename,
            download_url: url.to_string(),
            sha256: String::new(), // No hash verification for direct URLs
            size_bytes: 0, // Unknown size
            description: None,
        }
    } else {
        // Known model names (in a real implementation, this would come from an API)
        match model_identifier {
            "llama2-7b" => ModelInfo {
                name: "llama2-7b".to_string(),
                filename: "llama2-7b.Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf".to_string(),
                sha256: "6d8bbd42948f56e7b2d68e92b976deaae03d2f7e8a8da8432f8487b8237dafcc".to_string(),
                size_bytes: 4_000_000_000, // Approximate size
                description: Some("Llama 2 7B quantized to 4-bit".to_string()),
            },
            "mistral-7b" => ModelInfo {
                name: "mistral-7b".to_string(),
                filename: "mistral-7b.Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf".to_string(),
                sha256: "121e7a20a0a5e4db86f57d5ffabb534d6e1efa8c11ed0692a74987787580a6c5".to_string(),
                size_bytes: 4_200_000_000, // Approximate size
                description: Some("Mistral 7B quantized to 4-bit".to_string()),
            },
            "phi-2" => ModelInfo {
                name: "phi-2".to_string(),
                filename: "phi-2.Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf".to_string(),
                sha256: "324356668fa5ba9f4135de348447bb2bbe2467eaa1b8fcfb53719de62fbd2499".to_string(),
                size_bytes: 1_800_000_000, // Approximate size
                description: Some("Phi-2 quantized to 4-bit".to_string()),
            },
            "neural-chat-7b" => ModelInfo {
                name: "neural-chat-7b".to_string(),
                filename: "neural-chat-7b.Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf".to_string(),
                sha256: "e7eb44a9c9a3ccbc92fc0bdcf5a9575d4c6e2f98f5e160e4283c0c3d627a9e50".to_string(),
                size_bytes: 4_300_000_000, // Approximate size
                description: Some("Neural Chat 7B v3.1 quantized to 4-bit".to_string()),
            },
            _ => {
                // Unknown model - try to normalize the name and guess
                let normalized = model_identifier.to_lowercase();
                if normalized.contains("llama") {
                    Box::pin(get_model_info("llama2-7b")).await?
                } else if normalized.contains("mistral") {
                    Box::pin(get_model_info("mistral-7b")).await?
                } else if normalized.contains("phi") {
                    Box::pin(get_model_info("phi-2")).await?
                } else if normalized.contains("neural") || normalized.contains("chat") {
                    Box::pin(get_model_info("neural-chat-7b")).await?
                } else {
                    anyhow::bail!("Unknown model: {}. Please provide a URL or a supported model name.", model_identifier);
                }
            }
        }
    };
    
    Ok(model_info)
}

/// Download a model file from the given URL to the target path
pub async fn download_model_file(url: &str, target_path: &Path, expected_hash: &str) -> Result<()> {
    let client = Client::new();
    
    // Get content length for progress bar
    let response = client
        .head(url)
        .send()
        .await
        .context("Failed to send HEAD request")?;
    
    let total_size = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|cl| cl.to_str().ok())
        .and_then(|cl_str| cl_str.parse::<u64>().ok())
        .unwrap_or(0);
    
    // Create a temporary file
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path().join("model_download.tmp");
    
    // Set up progress bar
    let progress_bar = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {bytes} downloaded")?
        );
        pb
    };
    
    // Start the download
    info!("Downloading model from {}", url);
    println!("Downloading model from {}", url);
    
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to send GET request")?;
    
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(&temp_path).await?;
    let mut downloaded_bytes = 0u64;
    let mut hasher = Sha256::new();
    
    while let Some(item) = stream.next().await {
        let chunk = item.context("Error while downloading file")?;
        file.write_all(&chunk).await?;
        hasher.update(&chunk);
        
        downloaded_bytes += chunk.len() as u64;
        progress_bar.set_position(downloaded_bytes);
    }
    
    // Close the file
    file.flush().await?;
    drop(file);
    
    progress_bar.finish_with_message("Download completed");
    
    // Verify hash if provided
    if !expected_hash.is_empty() {
        let hash = hex::encode(hasher.finalize());
        if hash != expected_hash {
            println!("⚠️  Hash verification failed!");
            println!("   Expected: {}", expected_hash);
            println!("   Got:      {}", hash);
            println!("   This usually means the model file has been updated.");
            println!("   You can either:");
            println!("   1. Report this issue if you believe the hash in the code is wrong");
            println!("   2. Use a direct URL download which skips hash verification");
            println!("   3. Continue anyway if you trust the source (not recommended)");
            
            anyhow::bail!(
                "Hash verification failed. Expected {}, got {}. See above for solutions.",
                expected_hash,
                hash
            );
        }
        println!("✅ Hash verification successful");
        debug!("Hash verification successful");
    } else {
        println!("⚠️  Skipping hash verification (no expected hash provided)");
    }
    
    // Create parent directories if they don't exist
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Move file to final location
    std::fs::rename(&temp_path, target_path)
        .or_else(|_| -> anyhow::Result<()> {
            // If rename fails (e.g., across different filesystems), try copy + delete
            std::fs::copy(&temp_path, target_path)?;
            std::fs::remove_file(&temp_path)?;
            Ok(())
        })?;
    
    info!("Model downloaded and saved to {:?}", target_path);
    println!("Model downloaded and saved to {:?}", target_path);
    
    Ok(())
}