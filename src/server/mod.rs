//! Server module for the Rust-based LLM chat tool

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path as FilePath, PathBuf},
    sync::{Arc, Mutex},
};
use tracing::info;

use crate::model::{
    self,
    inference::{ChatContext, ChatMessage, ChatRole, Model},
};

/// Server state shared across all connections
#[derive(Clone)]
struct AppState {
    /// Path to the models directory
    models_dir: PathBuf,
    /// Cache of loaded models to avoid reloading between requests
    models: Arc<Mutex<HashMap<String, Arc<Mutex<Model>>>>>,
}

/// Start the API server on the specified host and port
pub async fn start_server(host: String, port: u16, models_dir: PathBuf) -> anyhow::Result<()> {
    // Create shared state
    let state = AppState {
        models_dir,
        models: Arc::new(Mutex::new(HashMap::new())),
    };

    // Build router with routes
    let app = Router::new()
        // Model endpoints
        .route("/api/models", get(list_models))
        .route("/api/models/:model_name", get(get_model_info))
        .route("/api/models/:model_name", post(download_model))
        .route("/api/models/:model_name", delete(delete_model))
        // Chat endpoints
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        // Health check
        .route("/api/health", get(health_check))
        .with_state(state);

    // Parse the address and start the server
    let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;
    info!("Server listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Custom HTTP error with message
struct ApiError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": self.message
        }));
        (self.status, body).into_response()
    }
}

/// API response format
#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(ApiResponse::success("OK"))
}

/// Model information response
#[derive(Serialize)]
struct ModelListResponse {
    models: Vec<ModelInfo>,
}

/// Model information
#[derive(Serialize)]
struct ModelInfo {
    name: String,
    size_bytes: u64,
    last_modified: String,
}

/// List available models
async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<ModelListResponse>>, ApiError> {
    let models_dir = &state.models_dir;
    let mut models = Vec::new();

    // Read models from directory
    if models_dir.exists() {
        for entry in std::fs::read_dir(models_dir).map_err(|e| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("Failed to read models directory: {}", e),
        })? {
            let entry = entry.map_err(|e| ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("Failed to read directory entry: {}", e),
            })?;

            let path = entry.path();
            if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let metadata = entry.metadata().map_err(|e| ApiError {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        message: format!("Failed to read file metadata: {}", e),
                    })?;

                    let last_modified = metadata
                        .modified()
                        .map(|time| {
                            let datetime = chrono::DateTime::<chrono::Utc>::from(time);
                            datetime.to_rfc3339()
                        })
                        .unwrap_or_else(|_| "Unknown".to_string());

                    models.push(ModelInfo {
                        name: name.to_string(),
                        size_bytes: metadata.len(),
                        last_modified,
                    });
                }
            }
        }
    }

    Ok(Json(ApiResponse::success(ModelListResponse { models })))
}

/// Get information about a specific model
async fn get_model_info(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<ApiResponse<ModelInfo>>, ApiError> {
    let models_dir = &state.models_dir;
    let model_path = find_model_path(&model_name, models_dir).map_err(|e| ApiError {
        status: StatusCode::NOT_FOUND,
        message: format!("Model not found: {}", e),
    })?;

    let metadata = std::fs::metadata(&model_path).map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("Failed to read file metadata: {}", e),
    })?;

    let name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&model_name)
        .to_string();

    let last_modified = metadata
        .modified()
        .map(|time| {
            let datetime = chrono::DateTime::<chrono::Utc>::from(time);
            datetime.to_rfc3339()
        })
        .unwrap_or_else(|_| "Unknown".to_string());

    let model_info = ModelInfo {
        name,
        size_bytes: metadata.len(),
        last_modified,
    };

    Ok(Json(ApiResponse::success(model_info)))
}

/// Find a model path from a model name
fn find_model_path(model_name: &str, models_dir: &FilePath) -> anyhow::Result<PathBuf> {
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

/// Download model request
#[derive(Deserialize)]
struct DownloadModelRequest {
    force: Option<bool>,
}

/// Download a model
async fn download_model(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
    Json(request): Json<DownloadModelRequest>,
) -> Result<Json<ApiResponse<String>>, ApiError> {
    let force = request.force.unwrap_or(false);
    let models_dir = &state.models_dir;

    // Get model info
    let model_info = model::download::get_model_info(&model_name)
        .await
        .map_err(|e| ApiError {
            status: StatusCode::BAD_REQUEST,
            message: format!("Failed to get model information: {}", e),
        })?;

    let model_path = models_dir.join(&model_info.filename);

    // Check if model already exists
    if model_path.exists() && !force {
        return Ok(Json(ApiResponse::success(format!(
            "Model {} already exists",
            model_name
        ))));
    }

    // Delete existing model if force is true
    if model_path.exists() && force {
        std::fs::remove_file(&model_path).map_err(|e| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("Failed to delete existing model: {}", e),
        })?;
    }

    // Download the model
    model::download::download_model_file(
        &model_info.download_url,
        &model_path,
        &model_info.sha256,
    )
    .await
    .map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("Failed to download model: {}", e),
    })?;

    Ok(Json(ApiResponse::success(format!(
        "Model {} downloaded successfully",
        model_name
    ))))
}

/// Delete a model
async fn delete_model(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<ApiResponse<String>>, ApiError> {
    let models_dir = &state.models_dir;

    // Find the model path
    let model_path = find_model_path(&model_name, models_dir).map_err(|e| ApiError {
        status: StatusCode::NOT_FOUND,
        message: format!("Model not found: {}", e),
    })?;

    // Remove from model cache if loaded
    {
        let mut models = state.models.lock().unwrap();
        models.remove(&model_name);
    }

    // Delete the file
    std::fs::remove_file(&model_path).map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("Failed to delete model: {}", e),
    })?;

    Ok(Json(ApiResponse::success(format!(
        "Model {} deleted successfully",
        model_name
    ))))
}

/// Chat request
#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatRequestMessage>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
}

/// Chat message in request
#[derive(Deserialize)]
struct ChatRequestMessage {
    role: String,
    content: String,
}

/// Chat response
#[derive(Serialize)]
struct ChatResponse {
    message: ChatResponseMessage,
    usage: TokenUsage,
}

/// Chat message in response
#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: String,
}

/// Token usage statistics
#[derive(Serialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Chat endpoint for non-streaming responses
async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ApiResponse<ChatResponse>>, ApiError> {
    // Load the model
    let model = load_model(&request.model, &state).await?;
    let mut model = model.lock().unwrap();

    // Set model parameters
    if let Some(temp) = request.temperature {
        model.set_temperature(temp);
    }

    if let Some(max_tokens) = request.max_tokens {
        model.set_max_tokens(max_tokens);
    }

    // Create chat context
    let mut context = ChatContext::default();

    // Add messages to context
    for message in &request.messages {
        let role = match message.role.as_str() {
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "system" => {
                // Handle system message by updating system prompt
                context.system_prompt = message.content.clone();
                continue;
            }
            _ => {
                return Err(ApiError {
                    status: StatusCode::BAD_REQUEST,
                    message: format!("Invalid message role: {}", message.role),
                });
            }
        };

        context.add_message(ChatMessage {
            role,
            content: message.content.clone(),
        });
    }

    // Generate response
    let response = model.generate(&context).map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("Failed to generate response: {}", e),
    })?;

    // Create token usage (estimated)
    let prompt_tokens = context.format_prompt().len() / 4; // Rough estimate
    let completion_tokens = response.len() / 4; // Rough estimate

    let chat_response = ChatResponse {
        message: ChatResponseMessage {
            role: "assistant".to_string(),
            content: response,
        },
        usage: TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(ApiResponse::success(chat_response)))
}

/// Stream response chunk
#[derive(Serialize)]
struct ChatStreamResponse {
    id: String,
    model: String,
    choices: Vec<ChatStreamChoice>,
}

/// Stream choice
#[derive(Serialize)]
struct ChatStreamChoice {
    delta: ChatStreamDelta,
    index: usize,
    finish_reason: Option<String>,
}

/// Stream delta
#[derive(Serialize)]
struct ChatStreamDelta {
    role: Option<String>,
    content: Option<String>,
}

/// Stream chat endpoint
async fn chat_stream(
    State(_state): State<AppState>,
    Json(_request): Json<ChatRequest>,
) -> impl IntoResponse {
    // This would implement SSE streaming, but for now we'll return a simple response
    // indicating that streaming is not implemented yet
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ApiResponse::<()>::error(
            "Streaming responses not yet implemented",
        )),
    )
}

/// Load a model from cache or from disk
async fn load_model(model_name: &str, state: &AppState) -> Result<Arc<Mutex<Model>>, ApiError> {
    // Check if model is already loaded
    {
        let models = state.models.lock().unwrap();
        if let Some(model) = models.get(model_name) {
            return Ok(Arc::clone(model));
        }
    }

    // Load the model from disk
    let model = model::load_model(model_name, &state.models_dir).map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("Failed to load model: {}", e),
    })?;

    let model = Arc::new(Mutex::new(model));

    // Cache the model
    {
        let mut models = state.models.lock().unwrap();
        models.insert(model_name.to_string(), Arc::clone(&model));
    }

    Ok(model)
}