//! Chat template rendering (the model's own Jinja template) and tool-call parsing.

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde_json::Value;

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Loads `chat_template` from the model's tokenizer_config.json.
    pub fn from_tokenizer_config(json: &[u8]) -> Result<Self> {
        let cfg: Value = serde_json::from_slice(json)?;
        let template = cfg["chat_template"].as_str().context("tokenizer_config.json has no chat_template")?.to_string();
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template)?;
        Ok(Self { env })
    }

    pub fn render(&self, messages: &Value, tools: Option<&Value>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        Ok(tmpl.render(context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => true,
        })?)
    }
}

/// Splits Qwen-style output into plain content and `<tool_call>{json}</tool_call>` blocks,
/// returned as OpenAI `tool_calls` entries.
pub fn parse_tool_calls(text: &str) -> (String, Vec<Value>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            content.push_str(&rest[start..]);
            rest = "";
            break;
        };
        if let Ok(call) = serde_json::from_str::<Value>(after[..end].trim()) {
            let arguments = match &call["arguments"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            calls.push(serde_json::json!({
                "id": format!("call_{}", calls.len()),
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            }));
        } else {
            content.push_str(&rest[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// Splits `<think>...</think>` reasoning off the front of a reply, as vLLM's reasoning
/// parser does. Returns (reasoning if any, the rest).
pub fn split_reasoning(text: &str) -> (Option<String>, String) {
    match text.find("</think>") {
        Some(end) => {
            let reasoning = text[..end].trim_start().trim_start_matches("<think>").trim().to_string();
            (Some(reasoning), text[end + "</think>".len()..].trim_start().to_string())
        }
        None => (None, text.to_string()),
    }
}
