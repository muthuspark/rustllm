use anyhow::Result;
use llama_cpp_2::{
    context::LlamaContext,
    model::LlamaModel,
    llama_backend::LlamaBackend,
};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Context structure for maintaining conversation history
#[derive(Debug, Clone)]
pub struct ChatContext {
    /// System prompt to define LLM behavior
    pub system_prompt: String,
    /// List of user/assistant message pairs
    pub messages: Vec<ChatMessage>,
    /// Maximum number of messages to keep in context (older messages get trimmed)
    pub max_messages: usize,
    /// Maximum token context window size for the model
    pub context_size: usize,
}

/// Chat message representation
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role of the message sender (user or assistant)
    pub role: ChatRole,
    /// Content of the message
    pub content: String,
}

/// Message role (user or assistant)
#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
}

/// Prompt template formats for different model types
#[derive(Debug, Clone)]
pub enum PromptTemplate {
    /// ChatML format (OpenAI style)
    ChatML,
    /// Alpaca instruction format
    Alpaca,
    /// Llama2 chat format
    Llama2,
}

/// Model wrapper for LLM inference using llama-cpp-2
pub struct Model {
    /// Model path for reference
    model_path: std::path::PathBuf,
    /// Loaded llama model (None if not loaded)
    llama_model: Option<LlamaModel>,
    /// Llama context for inference
    llama_context: Option<LlamaContext<'static>>,
    /// Backend instance
    backend: Arc<LlamaBackend>,
    /// Model parameters
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    /// Model state
    loaded: bool,
    /// Model configuration
    config: ModelConfig,
}

/// Configuration for model loading and inference
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Context window size
    pub context_size: usize,
    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: i32,
    /// Number of threads for CPU inference
    pub n_threads: Option<usize>,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for ChatContext {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.".to_string(),
            messages: Vec::new(),
            max_messages: 20,
            context_size: 4096,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            n_gpu_layers: 0, // CPU only by default
            n_threads: None, // Let the system decide
            batch_size: 1,  // Single request at a time
        }
    }
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

impl ChatContext {
    /// Create a new chat context with a custom system prompt
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            max_messages: 20,
            context_size: 4096,
        }
    }

    /// Add a message to the context
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        
        // Trim older messages if we exceed max_messages
        if self.messages.len() > self.max_messages {
            let excess = self.messages.len() - self.max_messages;
            self.messages.drain(0..excess);
        }
    }

    /// Format the entire context as a string for the model
    pub fn format_prompt(&self) -> String {
        self.format_prompt_with_template(&PromptTemplate::ChatML)
    }
    
    /// Format prompt with specific template
    pub fn format_prompt_with_template(&self, template: &PromptTemplate) -> String {
        match template {
            PromptTemplate::ChatML => self.format_chatml(),
            PromptTemplate::Alpaca => self.format_alpaca(),
            PromptTemplate::Llama2 => self.format_llama2(),
        }
    }
    
    fn format_chatml(&self) -> String {
        let mut prompt = format!("<|im_start|>system\n{}<|im_end|>\n", self.system_prompt);
        
        for message in &self.messages {
            match message.role {
                ChatRole::User => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", message.content));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", message.content));
                }
            }
        }
        
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
    
    fn format_alpaca(&self) -> String {
        let mut prompt = format!("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n", self.system_prompt);
        
        if let Some(last_message) = self.messages.last() {
            if last_message.role == ChatRole::User {
                prompt.push_str(&format!("### Input:\n{}\n\n", last_message.content));
            }
        }
        
        prompt.push_str("### Response:\n");
        prompt
    }
    
    fn format_llama2(&self) -> String {
        let mut prompt = format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", self.system_prompt);
        
        for message in &self.messages {
            match message.role {
                ChatRole::User => {
                    prompt.push_str(&format!("{} [/INST]", message.content));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!(" {} [INST] ", message.content));
                }
            }
        }
        
        if !prompt.ends_with("[/INST]") {
            prompt.push_str(" [/INST]");
        }
        
        prompt
    }
}

impl Model {
    /// Load a model from the given path
    pub fn load(model_path: &Path) -> Result<Self> {
        Self::load_with_config(model_path, ModelConfig::default())
    }
    
    /// Load a model with custom configuration
    pub fn load_with_config(model_path: &Path, config: ModelConfig) -> Result<Self> {
        info!("Loading model from {:?} with config: {:?}", model_path, config);
        
        // Initialize backend
        let backend = LlamaBackend::init()?;
        let backend = Arc::new(backend);
        
        // Validate that the file exists and is a GGUF file
        if !model_path.exists() {
            anyhow::bail!("Model file does not exist: {:?}", model_path);
        }
        
        if !model_path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false) {
            warn!("Model file {:?} does not have .gguf extension", model_path);
        }
        
        // Check file size to ensure it's reasonable
        let metadata = std::fs::metadata(model_path)?;
        let file_size_mb = metadata.len() as f64 / 1_048_576.0;
        info!("Model file size: {:.2} MB", file_size_mb);
        
        if file_size_mb < 10.0 {
            warn!("Model file seems very small ({:.2} MB), this might not be a valid model", file_size_mb);
        }
        
        // Load the model using llama-cpp-2 - simplified approach
        let llama_model = LlamaModel::load_from_file(&backend, model_path, &Default::default())
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {}", e))?;
        
        info!("Model loaded successfully");
        
        // Create context for inference - simplified approach
        let llama_context = llama_model.new_context(&backend, Default::default())
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;
        
        info!("Context created successfully");
        
        Ok(Self {
            model_path: model_path.to_path_buf(),
            llama_model: Some(llama_model),
            llama_context: Some(llama_context),
            backend,
            temperature: 0.7,
            max_tokens: 1024,
            top_p: 0.95,
            loaded: true,
            config,
        })
    }
    
    /// Generate a response for the given context (simplified version)
    pub fn generate(&mut self, context: &ChatContext) -> Result<String> {
        if !self.loaded {
            anyhow::bail!("Model is not loaded");
        }
        
        let prompt = context.format_prompt();
        debug!("Using prompt: {}", prompt);
        debug!("Model parameters: temp={}, max_tokens={}, top_p={}", 
               self.temperature, self.max_tokens, self.top_p);
        
        // For now, return a simple response indicating the model is loaded
        let response = format!("Model response to: {}", prompt);
        info!("Generated response: {}", response);
        
        Ok(response)
    }
    
    /// Update temperature (0.0 - 1.0)
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }
    
    /// Update top_p (0.0 - 1.0)
    pub fn set_top_p(&mut self, top_p: f32) {
        self.top_p = top_p;
    }
    
    /// Update max_new_tokens
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }
    
    /// Get current temperature
    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }
    
    /// Get current max_tokens
    pub fn get_max_tokens(&self) -> usize {
        self.max_tokens
    }
    
    /// Get current top_p
    pub fn get_top_p(&self) -> f32 {
        self.top_p
    }
    
    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
    
    /// Get model configuration
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    
    /// Generate response without streaming (for API interface)
    pub fn generate_sync(&mut self, context: &ChatContext) -> Result<String> {
        self.generate(context)
    }
    
    /// Unload the model to free memory
    pub fn unload(&mut self) {
        info!("Unloading model: {:?}", self.model_path);
        self.llama_context = None;
        self.llama_model = None;
        self.loaded = false;
    }
}