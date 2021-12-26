use std::collections::{HashMap, HashSet};

extern crate regex;
use regex::Regex;

pub fn tokenize(lower_case_text: &str) -> HashSet<&str> {
    Regex::new(r"[a-z0-9']+")
        .unwrap()
        .find_iter(lower_case_text)
        .map(|mat| mat.as_str())
        .collect()
}

pub struct Message<'a> {
    pub text: &'a str,
    pub is_spam: bool,
}

pub struct NaiveBayesClassifier {
    pub alpha: f64,
    pub tokens: HashSet<String>,
    pub token_ham_counts: HashMap<String, i32>,
    pub token_spam_counts: HashMap<String, i32>,
    pub spam_messages_count: i32,
    pub ham_messages_count: i32,
}

impl NaiveBayesClassifier {
    pub fn train(&mut self, messages: &[Message]) {
        for message in messages.iter() {
            self.increment_message_classifications_count(message);
            for token in tokenize(&message.text.to_lowercase()) {
                self.tokens.insert(token.to_string());
                self.increment_word_counts(token, message.is_spam)
            }
        }
    }

    fn increment_message_classifications_count(&mut self, message: &Message) {
        if message.is_spam {
            self.spam_messages_count += 1;
        } else {
            self.ham_messages_count += 1;
        }
    }

    fn increment_word_counts(&mut self, token: &str, is_spam: bool) {
        if !self.token_spam_counts.contains_key(token) {
            self.token_spam_counts.insert(token.to_string(), 0);
        }

        if !self.token_ham_counts.contains_key(token) {
            self.token_ham_counts.insert(token.to_string(), 0);
        }

        if is_spam {
            self.increment_spam_count(token);
        } else {
            self.increment_ham_count(token);
        }
    }

    fn increment_spam_count(&mut self, token: &str) {
        *self.token_spam_counts.get_mut(token).unwrap() += 1;
    }

    fn increment_ham_count(&mut self, token: &str) {
        *self.token_ham_counts.get_mut(token).unwrap() += 1;
    }

    pub fn predict(&self, text: &str) -> f64 {
        let lower_case_text = text.to_lowercase();
        let message_tokens = tokenize(&lower_case_text);
        let (prob_if_spam, prob_if_ham) = self.probabilities_of_message(message_tokens);

        return prob_if_spam / (prob_if_spam + prob_if_ham);
    }

    fn probabilities_of_message(&self, message_tokens: HashSet<&str>) -> (f64, f64) {
        let mut log_prob_if_spam = 0.;
        let mut log_prob_if_ham = 0.;

        for token in self.tokens.iter() {
            let (prob_if_spam, prob_if_ham) = self.probabilites_of_token(&token);

            if message_tokens.contains(token.as_str()) {
                log_prob_if_spam += prob_if_spam.ln();
                log_prob_if_ham += prob_if_ham.ln();
            } else {
                log_prob_if_spam += (1. - prob_if_spam).ln();
                log_prob_if_ham += (1. - prob_if_ham).ln();
            }
        }

        let prob_if_spam = log_prob_if_spam.exp();
        let prob_if_ham = log_prob_if_ham.exp();

        return (prob_if_spam, prob_if_ham);
    }

    fn probabilites_of_token(&self, token: &str) -> (f64, f64) {
        let prob_of_token_spam = (self.token_spam_counts[token] as f64 + self.alpha)
            / (self.spam_messages_count as f64 + 2. * self.alpha);

        let prob_of_token_ham = (self.token_ham_counts[token] as f64 + self.alpha)
            / (self.ham_messages_count as f64 + 2. * self.alpha);

        return (prob_of_token_spam, prob_of_token_ham);
    }
}

pub fn new_classifier(alpha: f64) -> NaiveBayesClassifier {
    return NaiveBayesClassifier {
        alpha,
        tokens: HashSet::new(),
        token_ham_counts: HashMap::new(),
        token_spam_counts: HashMap::new(),
        spam_messages_count: 0,
        ham_messages_count: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_bayes() {
        //Maybe make overlap between spam messages
        let train_messages = [
            Message {
                text: "Free Bitcoin viagra XXX christmas deals 😻😻😻",
                is_spam: true,
            },
            Message {
                text: "Dear Granddaughter, please explain Bitcoin over Christmas dinner",
                is_spam: false,
            },
            Message {
                text: "Our dear alumni, please consider making a donation...",
                is_spam: true,
            },
        ];

        let alpha = 1.;
        let num_spam_messages = 2.;
        let num_ham_messages = 1.;

        let mut model = new_classifier(alpha);
        model.train(&train_messages);

        let mut expected_tokens: HashSet<String> = HashSet::new();
        for message in train_messages.iter() {
            for token in tokenize(&message.text.to_lowercase()) {
                expected_tokens.insert(token.to_string());
            }
        }

        let input_text = "Bitcoin crypto academy Christmas deals";

        let probs_if_spam = [
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "Free"  (not present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "Bitcoin"  (present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "viagra"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "XXX"  (not present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "christmas"  (present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "deals"  (present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "dear"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "granddaughter"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "please"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "explain"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "over"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "dinner"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "our"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "alumni"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "consider"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "making"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "a"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "donation"  (not present)
        ];

        let probs_if_ham = [
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "Free"  (not present)
            (1. + alpha) / (num_ham_messages + 2. * alpha),      // "Bitcoin"  (present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "viagra"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "XXX"  (not present)
            (1. + alpha) / (num_ham_messages + 2. * alpha),      // "christmas"  (present)
            (0. + alpha) / (num_ham_messages + 2. * alpha),      // "deals"  (present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "dear"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "granddaughter"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "please"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "explain"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "over"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "dinner"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "our"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "alumni"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "consider"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "making"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "a"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "donation"  (not present)
        ];

        let p_if_spam_log: f64 = probs_if_spam.iter().map(|p| p.ln()).sum();
        let p_if_spam = p_if_spam_log.exp();

        let p_if_ham_log: f64 = probs_if_ham.iter().map(|p| p.ln()).sum();
        let p_if_ham = p_if_ham_log.exp();

        // P(message | spam) / (P(messge | spam) + P(message | ham)) rounds to 0.79
        assert!((model.predict(input_text) - p_if_spam / (p_if_spam + p_if_ham)).abs() < 0.000001);
    }
}
