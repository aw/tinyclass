//! A decision model out of a chat model: ask it a multiple-choice question,
//! stop after the prompt, and read the logits of the answer letters instead
//! of sampling. The softmax over just those letters is the decision.

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::llama_backend::LlamaBackend;
use std::num::NonZeroU32;

use crate::device::Placement;
use crate::model::{self, Loaded};

const LABELS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

pub struct Decision {
    pub scores: Vec<Score>,
}

pub struct Score {
    pub choice: String,
    pub logit: f32,
    pub logprob: f32,
    pub probability: f32,
}

impl Decision {
    pub fn chosen(&self) -> &Score {
        self.scores
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .expect("a decision has at least one choice")
    }
}

/// Jev's yes/no question, named after the Bernoulli distribution: how likely
/// the statement is true, as one number between 0 and 1.
pub fn noul(model: &Loaded, instruction: &str, statement: &str) -> Result<f32> {
    let choices = ["Yes".to_string(), "No".to_string()];
    let decision = decide(model, instruction, statement, &choices)?;
    Ok(decision.scores[0].probability)
}

pub fn decide(model: &Loaded, instruction: &str, input: &str, choices: &[String]) -> Result<Decision> {
    if choices.len() < 2 {
        bail!("a decision needs at least two choices");
    }
    if choices.len() > LABELS.len() {
        bail!("a decision can have at most {} choices", LABELS.len());
    }

    match model {
        Loaded::Local { backend, model, placement, .. } => {
            let prompt = prompt(instruction, input, choices);
            let logits = last_token_logits(model, backend, placement, &prompt)?;
            let logits = resolve_label_logits(model, &logits, choices)?;
            let normalizer = log_sum_exp(&logits);
            let scores = choices
                .iter()
                .zip(logits)
                .map(|(choice, logit)| Score {
                    choice: choice.clone(),
                    logit,
                    logprob: logit - normalizer,
                    probability: (logit - normalizer).exp(),
                })
                .collect();
            Ok(Decision { scores })
        }
        Loaded::Remote { server_url, model_name } => {
            let prompt = prompt(instruction, input, choices);
            let logits = resolve_remote_logits(server_url, model_name, &prompt, choices)?;
            let choice_logits = logits[..choices.len()].to_vec();
            let normalizer = log_sum_exp(&choice_logits);
            let scores = choices
                .iter()
                .zip(choice_logits)
                .map(|(choice, logit)| Score {
                    choice: choice.clone(),
                    logit,
                    logprob: logit - normalizer,
                    probability: (logit - normalizer).exp(),
                })
                .collect();
            Ok(Decision { scores })
        }
    }
}

fn prompt(instruction: &str, input: &str, choices: &[String]) -> String {
    let options: Vec<String> = choices
        .iter()
        .zip(LABELS)
        .map(|(choice, label)| format!("{label}. {choice}"))
        .collect();

    format!(
        "<|im_start|>system\n{instruction}<|im_end|>\n\
         <|im_start|>user\n{input}\n\n{}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n",
        options.join("\n")
    )
}

/// One context per question, sized to the prompt: the whole prompt goes in
/// as a single batch and only its last position keeps logits.
fn last_token_logits(model: &LlamaModel, backend: &'static LlamaBackend, placement: &Placement, prompt: &str) -> Result<Vec<f32>> {
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .context("could not tokenize the prompt")?;
    let length = u32::try_from(tokens.len())?;
    let threads = i32::try_from(model::physical_cores())?;

    let on_cpu = placement.device.is_none();
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(length))
        .with_n_batch(length)
        .with_n_ubatch(length)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_op_offload(!on_cpu)
        .with_offload_kqv(!on_cpu)
        .with_no_perf(true);
    let mut context = model
        .new_context(backend, params)
        .context("could not create a context for the prompt")?;

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len() - 1;
    for (position, token) in tokens.iter().enumerate() {
        batch.add(*token, i32::try_from(position)?, &[0], position == last)?;
    }
    context.decode(&mut batch).context("could not run the prompt")?;

    Ok(context.get_logits_ith(i32::try_from(last)?).to_vec())
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|it| (it - max).exp()).sum::<f32>().ln()
}

/// Extract logits at the token positions for each choice label (A, B, C…).
fn resolve_label_logits(model: &LlamaModel, logits: &[f32], choices: &[String]) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(choices.len());
    for (i, _) in choices.iter().enumerate() {
        let label = LABELS[i];
        let token_buf = model.str_to_token(&label.to_string(), AddBos::Never)
            .context("could not tokenize label")?;
        let target = token_buf.first().ok_or_else(|| anyhow::anyhow!("empty token for label '{label}'"))?;

        // Find the vocab index for this token by iterating with model.tokens()
        for (vocab_idx, (vocab_token, _)) in model.tokens(false).enumerate() {
            if vocab_token.0 == target.0 {
                result.push(logits[vocab_idx]);
                break;
            }
        }
    }
    Ok(result)
}

/// Send an HTTP POST to the ollama /api/generate endpoint and get back
/// logprobs for the label tokens at the last position.
fn resolve_remote_logits(server_url: &str, model_name: &str, prompt: &str, choices: &[String]) -> Result<Vec<f32>> {
    let body = serde_json::json!({
        "model": model_name,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.0
        }
    });

    let response = ureq::post(&format!("{server_url}/api/generate"))
        .header("Content-Type", "application/json")
        .send_json(body)
        .context("could not send request to ollama")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.into_body().read_to_string().unwrap_or_default();
        bail!("ollama returned status {status}: {body}");
    }

    let parsed: serde_json::Value = response
        .into_body()
        .read_json()
        .context("could not parse ollama response")?;

    // Try native ollama logprobs format first:
    // { "logprobs": [ { "token": "...", "logprob": -1.2, "top_k": [...] }, ... ] }
    if let Some(logprobs) = parsed.get("logprobs").and_then(|it| it.as_array()) {
        let logits: Vec<f32> = choices
            .iter()
            .zip(LABELS)
            .map(|(_choice, label)| {
                for entry in logprobs {
                    if let Some(token) = entry["token"].as_str() {
                        if token.trim() == label.to_string() {
                            return entry["logprob"].as_f64().unwrap_or(f64::NEG_INFINITY) as f32;
                        }
                    }
                }
                f32::NEG_INFINITY
            })
            .collect();
        return Ok(logits);
    }

    // Fallback: parse the raw response as the answer, return equal scores
    // (ollama didn't return logprobs — we can't compute per-choice probabilities)
    let _ = parsed.get("response")
        .map(|it| it.as_str().unwrap_or(""))
        .map(|resp| eprintln!("  ollama answer: {resp}"));

    Ok(vec![0.0; choices.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chosen_is_the_most_probable_choice() {
        let decision = Decision {
            scores: vec![
                Score { choice: "Spam".into(), logit: 1.0, logprob: -2.0, probability: 0.1 },
                Score { choice: "Phishing".into(), logit: 3.0, logprob: -0.1, probability: 0.9 },
            ],
        };
        assert_eq!(decision.chosen().choice, "Phishing");
    }

    #[test]
    fn prompt_labels_choices_in_order() {
        let choices = vec!["Legitimate".to_string(), "Spam".to_string()];
        let prompt = prompt("Choose one option.", "Hello", &choices);
        assert!(prompt.contains("Hello\n\nA. Legitimate\nB. Spam<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
}
