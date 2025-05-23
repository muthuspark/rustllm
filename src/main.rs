mod model;
mod server;
mod cli;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,

    /// Path to models directory
    #[clap(long, env = "RUSTLLM_MODELS_PATH", global = true)]
    models_path: Option<PathBuf>,

    /// Enable verbose logging
    #[clap(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the local LLM server
    Serve {
        /// Host address to bind the server
        #[clap(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind the server
        #[clap(long, default_value = "8000")]
        port: u16,
    },
    
    /// Run the interactive chat CLI
    Chat {
        /// Model to use for chat
        #[clap(long)]
        model: String,
    },
    
    /// Manage models (download, list, delete)
    Model {
        #[clap(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download a model
    Pull {
        /// Model name or URL to download
        model: String,
        
        /// Skip hash verification (use with caution)
        #[clap(long)]
        skip_hash: bool,
    },
    
    /// List all available models
    List,
    
    /// Delete a model
    Delete {
        /// Model name to delete
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let cli = Cli::parse();
    
    // Set up logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    // Get models path, default to ~/.rustllm/models if not specified
    let models_path = match cli.models_path {
        Some(path) => path,
        None => {
            let mut home_dir = home::home_dir().expect("Could not find home directory");
            home_dir.push(".rustllm");
            home_dir.push("models");
            home_dir
        }
    };
    
    // Create models directory if it doesn't exist
    if !models_path.exists() {
        std::fs::create_dir_all(&models_path)?;
        info!("Created models directory at {:?}", models_path);
    }
    
    // Process command
    match cli.command {
        Commands::Serve { host, port } => {
            info!("Starting server on {}:{}", host, port);
            server::start_server(host, port, models_path).await?;
        },
        
        Commands::Chat { model } => {
            info!("Starting chat with model: {}", model);
            cli::start_chat(&model, &models_path).await?;
        },
        
        Commands::Model { action } => match action {
            ModelAction::Pull { model, skip_hash } => {
                info!("Downloading model: {}", model);
                model::download_model_with_options(&model, &models_path, skip_hash).await?;
            },
            
            ModelAction::List => {
                info!("Listing available models");
                model::list_models(&models_path).await?;
            },
            
            ModelAction::Delete { model } => {
                info!("Deleting model: {}", model);
                model::delete_model(&model, &models_path).await?;
            },
        },
    }
    
    Ok(())
}