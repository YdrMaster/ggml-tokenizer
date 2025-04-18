use ggus::{GGuf, GGufMetaMapExt};
use memmap2::Mmap;
use std::collections::HashSet;
use std::{collections::HashMap, fs::File};
type TokenId = u32;

fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let file = File::open(path).unwrap();
    let file = unsafe { Mmap::map(&file) }.unwrap();
    let gguf = GGuf::new(&file).unwrap();

    let model = gguf.tokenizer_ggml_model().unwrap();
    println!("tokenizer model = {model}");
    assert_eq!(model, "gpt2");

    let mut config = TokenizerConfig::new();
    // 此处等同于llama.cpp的合并
    let bpe_ranks = load_gpt2(&gguf);

    println!("gpt2 n rank = {}", bpe_ranks.len());

    config.bos = 11;
    config.eos = 11;
    config.unk = NULL;
    config.sep = NULL;
    config.pad = NULL;
    config.mask = NULL;

    // bpe 需要预填充数据，设置字段
    config.add_space_prefix = false;
    config.clean_spaces = true;
    // gpt2 默认填充规则  LLAMA_VOCAB_PRE_TYPE_GPT2

    // 检查是是否有填充字段，
    // ggml 库中需要添加
    config.add_space_prefix = get_bool(
        gguf.get_str("tokenizer.ggml.add_space_prefix").is_ok(),
        config.add_space_prefix,
    );
    // remove_extra_whitespaces
    config.remove_extra_whitespaces = get_bool(
        gguf.get_str("tokenizer.ggml.remove_extra_whitespaces")
            .is_ok(),
        config.remove_extra_whitespaces,
    );

    let tokens = gguf.tokenizer_ggml_tokens().unwrap();
    let scores = gguf
        .tokenizer_ggml_scores()
        .ok()
        .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>());
    let token_type = gguf
        .tokenizer_ggml_token_type()
        .ok()
        .map(|arr| arr.map(|r| r.unwrap()).collect::<Vec<_>>());

    let mut id_to_token = Vec::with_capacity(tokens.len());
    let mut token_to_id: HashMap<String, TokenId> = HashMap::with_capacity(tokens.len());

    for (i, text) in tokens.into_iter().enumerate() {
        let text = text.unwrap().to_string();
        let score = scores.as_ref().map_or(0.0, |s| s[i]);
        let attribute = token_type
            .as_ref()
            .map_or(TokenAttribute::Normal, |t| unsafe {
                std::mem::transmute(t[i])
            });

        id_to_token.push(TokenData {
            text: text.clone(),
            score,
            attribute,
        });

        token_to_id.insert(text, i as u32);
    }
    // TODO 待完善 linefeed_id  构造换行符
    match config.vocab_type {
        VocabType::None | VocabType::Bpe => {
            // const std::vector<int> ids = tokenize("\n", false);

            // //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
            // if (ids.empty()) {
            //     LLAMA_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            //     linefeed_id = special_pad_id;
            // } else {
            //     linefeed_id = ids[0];
            // }
        }
        VocabType::Spm => {
            config.linefeed = if token_to_id.contains_key("\n") {
                *token_to_id.get("\n").unwrap()
            } else {
                config.pad
            };
        }
        VocabType::Wpm => todo!(),
        VocabType::Ugm => todo!(),
        VocabType::Rwkv => todo!(),
    }
    // 检查特殊字符special_bos_id等的有效性

    // 判断模型是否有 add_bos   add_eos
    {
        config.add_space_prefix = get_bool(
            gguf.get_str("tokenizer.ggml.add_bos_token").is_ok(),
            config.add_space_prefix,
        );
        config.add_space_prefix = get_bool(
            gguf.get_str("tokenizer.ggml.add_eos_token").is_ok(),
            config.add_space_prefix,
        );
    }
    // todo 为特殊token值为null构建字符,高度类似，能偶抽象成方法或者声明宏
    for (key, value) in &token_to_id {
        if config.eot == NULL {
            if key == "<|eot_id|>"
                || key == "<|im_end|>"
                || key == "<|end|>"
                || key == "<end_of_turn>"
                || key == "<|endoftext|>"
                || key == "< EOT >"
                || key == "_< EOT >"
                || key == "<｜end▁of▁sentence｜>"
            // DeepSeek
            {
                config.eos = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.eom == NULL {
            if key == "<|eom_id|>" {
                config.eom = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_pre == NULL {
            if key == "<|fim_prefix|>" // Qwen
                || key == "<fim-prefix>"
                || key == "<｜fim▁begin｜>" // DeepSeek
                || key == "<PRE>"
                || key == "▁<PRE>"
            {
                config.fim_pre = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_suf == NULL {
            if key == "<|fim_suffix|>" // Qwen
            || key == "<fim-suffix>"
            || key == "<｜fim▁hole｜>" // DeepSeek
            || key == "<SUF>"
            || key == "▁<SUF>"
            // CodeLlama
            {
                config.fim_suf = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_mid == NULL {
            if key == "<|fim_middle|>" // Qwen
            || key == "<fim-middle>"
            || key == "<｜fim▁end｜>" // DeepSeek
            || key == "<MID>"
            || key == "▁<MID>"
            // CodeLlama
            {
                config.fim_mid = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_mid == NULL {
            if key== "<|fim_middle|>" // Qwen
            || key== "<fim-middle>"
            || key== "<｜fim▁end｜>"  // DeepSeek
            || key== "<MID>"
            || key== "▁<MID>"
            // CodeLlama
            {
                config.fim_mid = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_pad == NULL {
            if key == "<|fim_pad|>" // Qwen
                || key == "<fim-pad>"
                || key == "<PAD>"
            {
                config.fim_pad = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_rep == NULL {
            if key == "<|fim_repo|>"  // Qwen
            || key == "<|repo_name|>"
            || key == "<fim-repo>"
            || key == "<REPO>"
            {
                config.fim_rep = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
        if config.fim_sep == NULL {
            if key == "<|file_sep|>"
            // Qwen
            {
                config.fim_sep = *value;
                if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32)
                    == 0
                {
                    id_to_token[*value as usize].attribute = TokenAttribute::Control;
                }
            }
        }
    }

    let mut special_eog_ids = HashSet::new();
    // maintain a list of tokens that cause end-of-generation
    if config.fim_pad != NULL && !special_eog_ids.contains(&config.fim_pad) {
        special_eog_ids.insert(config.fim_pad);
    }
    if config.fim_rep != NULL && !special_eog_ids.contains(&config.fim_rep) {
        special_eog_ids.insert(config.fim_rep);
    }
    if config.fim_sep != NULL && !special_eog_ids.contains(&config.fim_sep) {
        special_eog_ids.insert(config.fim_sep);
    }
    // 第二个循环也使用引用
    for (key, value) in &token_to_id {
        if key == "<|eot_id|>"
            || key == "<|im_end|>"
            || key == "<|end|>"
            || key == "<end_of_turn>"
            || key == "<|endoftext|>"
            || key == "<|eom_id|>"
            || key == "< EOT >"
            || key == "_< EOT >"
        {
            special_eog_ids.insert(*value);
            if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32) == 0
            {
                id_to_token[*value as usize].attribute = TokenAttribute::Control;
            }
        } else {
            if (id_to_token[*value as usize].attribute as i32 & TokenAttribute::Control as i32) == 0
                && !special_eog_ids.contains(value)
            {
                log::warn!("{}", key);
            }
        }
    }
    print!("{:?}", config)

    // 构建终止字符表

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
fn get_bool(model: bool, config: bool) -> bool {
    match (model, config) {
        (true, true) => true,
        (true, false) => true,
        (false, true) => false,
        (false, false) => false,
    }
}
#[derive(Debug)]
struct TokenizerConfig {
    vocab_type: VocabType,
    bos: u32,
    eos: u32,
    eot: u32,
    eom: u32,
    unk: u32,
    sep: u32,
    pad: u32,
    fim_pre: u32,
    fim_suf: u32,
    fim_mid: u32,
    fim_pad: u32,
    fim_rep: u32,
    fim_sep: u32,
    linefeed: u32,
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
            vocab_type: VocabType::None,
            bos: 1,
            eos: 2,
            eot: NULL,
            eom: NULL,
            unk: 0,
            sep: NULL,
            pad: NULL,
            fim_pre: NULL,
            fim_suf: NULL,
            fim_mid: NULL,
            fim_pad: NULL,
            fim_rep: NULL,
            fim_sep: NULL,
            linefeed: NULL,
            mask: NULL,
            add_space_prefix: false,
            add_bos: true,
            add_eos: true,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
        }
    }
}

struct TokenData {
    pub text: String,
    pub score: f32,
    pub attribute: TokenAttribute,
}

#[repr(i32)]
#[derive(Copy, Clone)]
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
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum VocabType {
    None = 0, // For models without vocab
    Spm = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
    Bpe = 2,  // GPT-2 tokenizer based on byte-level BPE
    Wpm = 3,  // BERT tokenizer based on WordPiece
    Ugm = 4,  // T5 tokenizer based on Unigram
    Rwkv = 5, // RWKV tokenizer based on greedy tokenization
}
