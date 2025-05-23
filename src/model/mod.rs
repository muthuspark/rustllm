pub mod download;
pub mod inference;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tracing::{error, info};

// Main functions exposed from this module
pub async fn download_model(model_name: &str, models_dir: &Path) -> Result<()> {
    download_model_with_options(model_name, models_dir, false).await
}

pub async fn download_model_with_options(model_name: &str, models_dir: &Path, skip_hash: bool) -> Result<()> {
    let model_info = download::get_model_info(model_name).await?;
    let model_path = models_dir.join(&model_info.filename);
    
    // Check if model already exists
    if model_path.exists() {
        info!("Model {} already exists at {:?}", model_name, model_path);
        
        if !skip_hash && !model_info.sha256.is_empty() {
            // Verify hash
            let file_hash = calculate_file_hash(&model_path)?;
            if file_hash == model_info.sha256 {
                info!("Model hash verified successfully");
                return Ok(());
            } else {
                info!("Model hash verification failed, redownloading");
                fs::remove_file(&model_path)?;
            }
        } else {
            info!("Skipping hash verification for existing model");
            return Ok(());
        }
    }
    
    let expected_hash = if skip_hash { String::new() } else { model_info.sha256 };
    download::download_model_file(&model_info.download_url, &model_path, &expected_hash).await?;
    info!("Model {} downloaded successfully to {:?}", model_name, model_path);
    Ok(())
}

pub async fn list_models(models_dir: &Path) -> Result<()> {
    info!("Listing models in {:?}", models_dir);
    
    // Ensure the directory exists
    if !models_dir.exists() {
        println!("Models directory does not exist. No models available.");
        return Ok(());
    }
    
    // List local models
    let mut models_found = false;
    println!("Available local models:");
    
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            if let Some(model_name) = path.file_name().and_then(|n| n.to_str()) {
                let size_bytes = entry.metadata()?.len();
                let size_mb = size_bytes as f64 / 1_048_576.0;
                
                println!("- {} ({:.2} MB)", model_name, size_mb);
                models_found = true;
            }
        }
    }
    
    if !models_found {
        println!("No models found. Use 'rustllm model pull <model>' to download a model.");
    }
    
    // List available models to download (from a hypothetical registry)
    println!("\nModels available for download:");
    println!("- llama2-7b.Q4_K_M.gguf");
    println!("- mistral-7b.Q4_K_M.gguf");
    println!("- phi-2.Q4_K_M.gguf");
    println!("- neural-chat-7b.Q4_K_M.gguf");
    
    Ok(())
}

pub async fn delete_model(model_name: &str, models_dir: &Path) -> Result<()> {
    let model_path = find_model_path(model_name, models_dir)?;
    
    // Delete the file
    fs::remove_file(&model_path)
        .with_context(|| format!("Failed to delete model file at {:?}", model_path))?;
    
    info!("Model {} deleted successfully", model_name);
    println!("Model {} deleted successfully", model_name);
    
    Ok(())
}

// Helper functions
fn find_model_path(model_name: &str, models_dir: &Path) -> Result<PathBuf> {
    // Check if the exact filename exists
    let exact_path = models_dir.join(model_name);
    if exact_path.exists() {
        return Ok(exact_path);
    }
    
    // Check if model_name with .gguf extension exists
    let with_extension = if model_name.ends_with(".gguf") {
        models_dir.join(model_name)
    } else {
        models_dir.join(format!("{}.gguf", model_name))
    };
    
    if with_extension.exists() {
        return Ok(with_extension);
    }
    
    // Try to find a partial match
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.contains(model_name) {
                    return Ok(path);
                }
            }
        }
    }
    
    anyhow::bail!("Model {} not found in {:?}", model_name, models_dir)
}

pub fn calculate_file_hash(file_path: &Path) -> Result<String> {
    let mut file = File::open(file_path)?;
    let mut hasher = Sha256::new();
    
    let mut buffer = [0; 1024 * 1024]; // 1MB buffer
    loop {
        let bytes_read = std::io::Read::read(&mut file, &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

// Load a model for inference
pub fn load_model(model_name: &str, models_dir: &Path) -> Result<inference::Model> {
    let model_path = find_model_path(model_name, models_dir)?;
    inference::Model::load(&model_path)
}