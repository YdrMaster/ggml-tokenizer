use ggus::{GGuf, GGufMetaMapExt};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File};

fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let file = File::open(path).unwrap();
    let file = unsafe { Mmap::map(&file) }.unwrap();
    let gguf = GGuf::new(&file).unwrap();

    let model = gguf.tokenizer_ggml_model().unwrap();
    println!("tokenizer model = {model}");
    assert_eq!(model, "gpt2");

    let mut config = TokenizerConfig::new();

    let bpe_ranks = load_gpt2(&gguf);

    println!("gpt2 n rank = {}", bpe_ranks.len());

    config.bos = 11;
    config.eos = 11;
    config.unk = NULL;

    let tokens = gguf.tokenizer_ggml_tokens().unwrap();
    let scores = gguf
        .tokenizer_ggml_scores()
        .ok()
        .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>());
    let token_type = gguf
        .tokenizer_ggml_token_type()
        .ok()
        .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>());

    println!(
        "{} {} {}",
        tokens.len(),
        scores.is_some(),
        token_type.is_some()
    );

    // println!("{}", gguf.general_architecture().unwrap())
}

fn load_gpt2(gguf: &GGuf) -> HashMap<(String, String), usize> {
    gguf.tokenizer_ggml_merges()
        .unwrap()
        .map(|x| {
            let piece = x.unwrap();
            let (first, second) = piece.split_once(' ').unwrap();
            (first.to_string(), second.to_string())
        })
        .enumerate()
        .map(|(i, pair)| (pair, i))
        .collect()
}

struct TokenizerConfig {
    bos: u32,
    eos: u32,
    eot: u32,
    eom: u32,
    unk: u32,
    sep: u32,
    pad: u32,
    mask: u32,
    add_space_prefix: bool,
    add_bos: bool,
    add_eos: bool,
    ignore_merges: bool,
    clean_spaces: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
}

const NULL: u32 = u32::MAX;

impl TokenizerConfig {
    pub fn new() -> Self {
        Self {
            bos: 1,
            eos: 2,
            eot: NULL,
            eom: NULL,
            unk: 0,
            sep: NULL,
            pad: NULL,
            mask: NULL,
            add_space_prefix: false,
            add_bos: false,
            add_eos: false,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
        }
    }
}

struct TokenData {
    text: String,
    score: f32,
    attribute: TokenAttribute,
}

#[repr(i32)]
enum TokenAttribute {
    Undefined = 0,
    Unknown = 1 << 0,
    Unused = 1 << 1,
    Normal = 1 << 2,
    Control = 1 << 3, // SPECIAL?
    UserDefined = 1 << 4,
    Byte = 1 << 5,
    Normalized = 1 << 6,
    LStrIp = 1 << 7,
    RStrIp = 1 << 8,
    SingleWord = 1 << 9,
}
