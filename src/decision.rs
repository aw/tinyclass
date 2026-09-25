//! A decision model out of a chat model: ask it a multiple-choice question,
//! stop after the prompt, and read the logits of the answer letters instead
//! of sampling. The softmax over just those letters is the decision.

use anyhow::{Result, anyhow, bail};
use candle_core::Tensor;

use crate::model::Loaded;

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
pub fn noul(model: &mut Loaded, instruction: &str, statement: &str) -> Result<f32> {
    let choices = ["Yes".to_string(), "No".to_string()];
    let decision = decide(model, instruction, statement, &choices)?;
    Ok(decision.scores[0].probability)
}

pub fn decide(model: &mut Loaded, instruction: &str, input: &str, choices: &[String]) -> Result<Decision> {
    if choices.len() < 2 {
        bail!("a decision needs at least two choices");
    }
    if choices.len() > LABELS.len() {
        bail!("a decision can have at most {} choices", LABELS.len());
    }

    let logits = last_token_logits(model, &prompt(instruction, input, choices))?;
    let choice_logits = choices
        .iter()
        .zip(LABELS)
        .map(|(_, label)| Ok(logits[label_token(model, label)?]))
        .collect::<Result<Vec<f32>>>()?;

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

fn last_token_logits(model: &mut Loaded, prompt: &str) -> Result<Vec<f32>> {
    let encoding = model
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("could not tokenize the prompt: {error}"))?;
    let input = Tensor::new(encoding.get_ids(), &model.device)?.unsqueeze(0)?;

    model.weights.clear_kv_cache();
    let logits = model.weights.forward(&input, 0)?.squeeze(0)?;
    Ok(logits.to_vec1::<f32>()?)
}

fn label_token(model: &Loaded, label: char) -> Result<usize> {
    let encoding = model
        .tokenizer
        .encode(label.to_string(), false)
        .map_err(|error| anyhow!("could not tokenize '{label}': {error}"))?;
    Ok(encoding.get_ids()[0] as usize)
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|it| (it - max).exp()).sum::<f32>().ln()
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
