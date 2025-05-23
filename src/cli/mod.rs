//! CLI interface for the Rust-based LLM chat tool

use crate::model::{
    self, 
    inference::{ChatContext, ChatMessage, Model}
};
use crate::utils::{format_message, format_duration};
use anyhow::{Context as AnyhowContext, Result};
use colored::Colorize;
use rustyline::{DefaultEditor, Result as RustylineResult};
use rustyline::error::ReadlineError;
use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};
use tracing::{error, info};

/// Start the interactive chat CLI with the specified model
pub async fn start_chat(model_name: &str, models_dir: &Path) -> Result<()> {
    println!("{}", "Starting RustLLM Chat".bold().green());
    println!("Loading model: {}", model_name.bold());
    
    // Load the model
    let start_time = Instant::now();
    let mut model = model::load_model(model_name, models_dir)?;
    let load_duration = start_time.elapsed();
    println!("Model loaded in {}", format_duration(load_duration.as_secs()).bold());
    
    // Initialize chat context
    let mut context = ChatContext::default();
    
    // Print welcome message
    println!("\n{}", "Welcome to RustLLM Chat!".bold().green());
    println!("Type your messages to chat with the model.");
    println!("Use {}, {}, or {} to exit the chat.", "/quit".bold(), "/exit".bold(), "Ctrl+D".bold());
    println!("Use {} to change parameters (temperature, etc.)", "/params".bold());
    println!("Use {} to clear the conversation history.", "/clear".bold());
    println!("");
    
    // Start interactive prompt
    let mut rl = DefaultEditor::new()?;
    loop {
        // Display prompt and get user input
        let readline = rl.readline("You: ");
        
        match readline {
            Ok(line) => {
                // Add input to history
                let _ = rl.add_history_entry(&line);
                
                // Check for commands
                if line.trim().starts_with("/") {
                    match handle_command(&line, &mut model, &mut context) {
                        Ok(should_exit) => {
                            if should_exit {
                                println!("{}", "Goodbye!".bold().green());
                                break;
                            }
                            continue;
                        }
                        Err(e) => {
                            println!("{}: {}", "Error".bold().red(), e);
                            continue;
                        }
                    }
                }
                
                // Skip empty messages
                if line.trim().is_empty() {
                    continue;
                }
                
                // Add the user message to context
                context.add_message(ChatMessage::user(&line));
                
                // Generate a response
                println!("\n{}: ", "Assistant".bold().blue());
                match model.generate(&context) {
                    Ok(response) => {
                        // Add the assistant's response to the context
                        context.add_message(ChatMessage::assistant(&response));
                        println!(); // Add a newline after the response
                    }
                    Err(e) => {
                        println!("{}: Failed to generate response: {}", "Error".bold().red(), e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("{}", "Goodbye!".bold().green());
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }
    
    Ok(())
}

/// Handle chat commands that begin with "/"
fn handle_command(
    command: &str, 
    model: &mut Model, 
    context: &mut ChatContext
) -> Result<bool> {
    let cmd = command.trim();
    
    match cmd {
        "/quit" | "/exit" => {
            return Ok(true); // Signal to exit
        }
        
        "/clear" => {
            // Clear conversation history
            *context = ChatContext::new(&context.system_prompt);
            println!("{}", "Conversation history cleared.".bold().green());
        }
        
        "/help" => {
            println!("{}", "Available commands:".bold());
            println!("  {} - Exit the chat", "/quit or /exit".bold());
            println!("  {} - Clear conversation history", "/clear".bold());
            println!("  {} - Show this help message", "/help".bold());
            println!("  {} - Show current parameters", "/params".bold());
            println!("  {} - Change temperature (0.0-1.0)", "/temp <value>".bold());
            println!("  {} - Change maximum response tokens", "/max_tokens <value>".bold());
            println!("  {} - Change system prompt", "/system <prompt>".bold());
        }
        
        "/params" => {
            // Display current parameters
            println!("{}", "Current parameters:".bold());
            println!("  System prompt: {}", context.system_prompt);
            println!("  Temperature: {}", model.get_temperature());
            println!("  Max tokens: {}", model.get_max_tokens());
            println!("  Messages in context: {}/{}", context.messages.len(), context.max_messages);
        }
        
        _ if cmd.starts_with("/temp ") => {
            // Change temperature
            if let Some(temp_str) = cmd.strip_prefix("/temp ") {
                match temp_str.trim().parse::<f32>() {
                    Ok(temp) if (0.0..=1.0).contains(&temp) => {
                        model.set_temperature(temp);
                        println!("Temperature set to {}", temp);
                    }
                    Ok(_) => {
                        println!("{}: Temperature must be between 0.0 and 1.0", "Error".bold().red());
                    }
                    Err(_) => {
                        println!("{}: Invalid temperature value", "Error".bold().red());
                    }
                }
            }
        }
        
        _ if cmd.starts_with("/max_tokens ") => {
            // Change max tokens
            if let Some(tokens_str) = cmd.strip_prefix("/max_tokens ") {
                match tokens_str.trim().parse::<usize>() {
                    Ok(tokens) if tokens > 0 => {
                        model.set_max_tokens(tokens);
                        println!("Max tokens set to {}", tokens);
                    }
                    Ok(_) => {
                        println!("{}: Max tokens must be greater than 0", "Error".bold().red());
                    }
                    Err(_) => {
                        println!("{}: Invalid max tokens value", "Error".bold().red());
                    }
                }
            }
        }
        
        _ if cmd.starts_with("/system ") => {
            // Change system prompt
            if let Some(prompt) = cmd.strip_prefix("/system ") {
                context.system_prompt = prompt.to_string();
                println!("System prompt updated");
            }
        }
        
        _ => {
            println!("{}: Unknown command: {}", "Error".bold().red(), cmd);
            println!("Type {} for a list of commands", "/help".bold());
        }
    }
    
    Ok(false) // Don't exit
}

/// Format the chat history for display
pub fn display_chat_history(context: &ChatContext) -> String {
    let mut result = String::new();
    
    result.push_str(&format!("{}\n", "System:".bold().yellow()));
    result.push_str(&format!("{}\n\n", context.system_prompt));
    
    for message in &context.messages {
        let role = match message.role {
            crate::model::inference::ChatRole::User => "User",
            crate::model::inference::ChatRole::Assistant => "Assistant",
        };
        
        result.push_str(&format_message(role, &message.content));
        result.push_str("\n\n");
    }
    
    result
}