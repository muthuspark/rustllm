//! Implementation of model management CLI commands (download, list, delete)

use anyhow::{Context as AnyhowContext, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{error, info};

use crate::model;
use crate::utils::{format_duration, format_file_size};

/// Download a model using the CLI interface
pub async fn download_model_command(model_name: &str, models_dir: &Path) -> Result<()> {
    println!("{} {}", "Downloading model:".bold(), model_name.bold().green());
    
    // Check if model already exists
    let model_info = match model::download::get_model_info(model_name).await {
        Ok(info) => {
            println!("Found model: {} ({})", info.name.bold(), format_file_size(info.size_bytes));
            if let Some(desc) = &info.description {
                println!("Description: {}", desc);
            }
            info
        },
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to get model information: {}", e));
        }
    };
    
    let model_path = models_dir.join(&model_info.filename);
    
    // Check if model already exists
    if model_path.exists() {
        println!("Model {} already exists at {:?}", model_name.bold(), model_path);
        
        // Verify hash if available
        if !model_info.sha256.is_empty() {
            println!("Verifying model hash...");
            
            let file_hash = model::calculate_file_hash(&model_path)?;
            if file_hash == model_info.sha256 {
                println!("{}", "Model hash verified successfully ✓".bold().green());
                return Ok(());
            } else {
                println!("{}", "Model hash verification failed, redownloading...".bold().yellow());
                std::fs::remove_file(&model_path)?;
            }
        } else {
            return Ok(());
        }
    }
    
    // Start timer for download
    let start_time = Instant::now();
    
    // Download the model
    match model::download::download_model_file(
        &model_info.download_url, 
        &model_path, 
        &model_info.sha256
    ).await {
        Ok(()) => {
            let duration = start_time.elapsed();
            println!(
                "{} in {}",
                "Download completed successfully ✓".bold().green(),
                format_duration(duration.as_secs()).bold()
            );
            Ok(())
        },
        Err(e) => {
            // Clean up partial download
            if model_path.exists() {
                let _ = std::fs::remove_file(&model_path);
            }
            Err(anyhow::anyhow!("Failed to download model: {}", e))
        }
    }
}

/// List available models using the CLI interface
pub async fn list_models_command(models_dir: &Path) -> Result<()> {
    println!("{}", "Available Models".bold().green());
    println!("Models directory: {:?}", models_dir);
    println!();
    
    // Ensure the directory exists
    if !models_dir.exists() {
        println!("Models directory does not exist. No models available.");
        return Ok(());
    }
    
    // Count and collect models
    let mut models_found = false;
    let mut models_info = Vec::new();
    
    for entry in std::fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            if let Some(model_name) = path.file_name().and_then(|n| n.to_str()) {
                let metadata = entry.metadata()?;
                let size_bytes = metadata.len();
                let modified = metadata.modified()
                    .unwrap_or_else(|_| std::time::SystemTime::UNIX_EPOCH);
                
                models_info.push((model_name.to_string(), size_bytes, modified));
                models_found = true;
            }
        }
    }
    
    // Sort models by name
    models_info.sort_by(|a, b| a.0.cmp(&b.0));
    
    // Display models table
    if models_found {
        println!("{:<40} {:<15} {}", "Model Name".bold(), "Size".bold(), "Last Modified".bold());
        println!("{}", "-".repeat(70));
        
        for (name, size, modified) in models_info {
            let size_str = format_file_size(size);
            
            // Format the modified time
            let modified_str = match modified.duration_since(std::time::SystemTime::UNIX_EPOCH) {
                Ok(duration) => {
                    use chrono::prelude::*;
                    let datetime = DateTime::<Utc>::from_timestamp(duration.as_secs() as i64, 0)
                        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
                    datetime.format("%Y-%m-%d %H:%M:%S").to_string()
                },
                Err(_) => "Unknown".to_string(),
            };
            
            println!("{:<40} {:<15} {}", name, size_str, modified_str);
        }
    } else {
        println!("No models found. Use 'rustllm model pull <model>' to download a model.");
    }
    
    // List available models for download
    println!("\n{}", "Models available for download:".bold().green());
    println!("- llama2-7b       (Llama 2 7B quantized to 4-bit)");
    println!("- mistral-7b      (Mistral 7B quantized to 4-bit)");
    println!("- phi-2           (Phi-2 quantized to 4-bit)");
    println!("- neural-chat-7b  (Neural Chat 7B v3.1 quantized to 4-bit)");
    
    Ok(())
}

/// Delete a model using the CLI interface
pub async fn delete_model_command(model_name: &str, models_dir: &Path) -> Result<()> {
    println!("{} {}", "Deleting model:".bold(), model_name.bold().red());
    
    // Find the model path
    let model_path = match find_model_path(model_name, models_dir) {
        Ok(path) => path,
        Err(_) => {
            return Err(anyhow::anyhow!("Model {} not found in {:?}", model_name, models_dir));
        }
    };
    
    // Confirm deletion
    println!("Are you sure you want to delete {}? (y/N)", model_path.display().to_string().bold());
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    
    if input.trim().to_lowercase() == "y" {
        // Delete the file
        std::fs::remove_file(&model_path)
            .with_context(|| format!("Failed to delete model file at {:?}", model_path))?;
        
        println!("{} {}", "Model".bold(), model_name.bold().red());
        println!("{}", "deleted successfully ✓".bold().green());
        Ok(())
    } else {
        println!("Deletion cancelled.");
        Ok(())
    }
}

/// Helper function to find a model path from a model name
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
    for entry in std::fs::read_dir(models_dir)? {
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