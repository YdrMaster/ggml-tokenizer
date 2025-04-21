#![feature(linked_list_cursors)]
use ggus::{GGuf, GGufMetaMapExt};
use memmap2::Mmap;
use std::collections::{HashSet, LinkedList};
use std::fmt::Error;
use std::sync::OnceLock;
use std::{collections::HashMap, fs::File};
type TokenId = u32;

static GLOBAL_CONFIG: OnceLock<TokenizerConfig> = OnceLock::new();

// 获取或初始化全局配置的函数
fn get_config() -> &'static TokenizerConfig {
    GLOBAL_CONFIG.get_or_init(|| {
        println!("Initializing TokenizerConfig for the first time..."); // 仅在首次调用时打印
        // 这里创建 TokenizerConfig 的实例
        TokenizerConfig::new()
    })
}
// load 函数
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

    // TODO 待完善 linefeed_id 暂时不支持SPM  构造换行符
    match config.vocab_type {
        VocabType::None | VocabType::Bpe => {
            // const std::vector<int> ids = tokenize("\n", false);

            // //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
            if token_to_id.get("\n").is_none() {
                config.linefeed = config.pad;
            } else {
                config.linefeed = *token_to_id.get("\n").unwrap();
            }
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
    // 收集特殊token
    config.special_tokens = id_to_token
        .iter()
        .enumerate() // 获取索引 (TokenId) 和 TokenData
        .filter(|(_, token_data)| {
            // 检查 token 的属性是否为 Control, UserDefined 或 Unknown
            let attr_val = token_data.attribute as i32;
            let special_mask = TokenAttribute::Control as i32
                | TokenAttribute::UserDefined as i32
                | TokenAttribute::Unknown as i32;
            (attr_val & special_mask) != 0
        })
        .map(|(index, _)| index as TokenId) // 提取符合条件的 TokenId (索引)
        .collect(); // 收集到 Vec<TokenId> 中

    config.token_to_id = token_to_id;
    config.id_to_token = id_to_token;
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
fn byte_to_token(ch: u8) -> Result<TokenId, Error> {
    let config = get_config();
    assert!(config.vocab_type != VocabType::None);

    // 十六进制字符表
    const HEX: &[u8; 16] = b"0123456789ABCDEF";

    match config.vocab_type {
        VocabType::Spm | VocabType::Ugm => {
            // 构建形如 <0xHH> 的格式
            let buf = [
                b'<',
                b'0',
                b'x',
                HEX[(ch >> 4) as usize],
                HEX[(ch & 15) as usize],
                b'>',
                0,
            ];

            // 首先尝试查找十六进制格式的token
            if let Some(token) = config.token_to_id.get(std::str::from_utf8(&buf).unwrap()) {
                return Ok(*token);
            }

            // 回退到直接使用字节值
            let buf2 = [ch, 0];
            Ok(*config
                .token_to_id
                .get(std::str::from_utf8(&buf2).unwrap())
                .expect("Token not found"))
        }

        // VocabType::Wpm | VocabType::Bpe => {
        //     *config.token_to_id.get(&unicode_byte_to_utf8(ch))
        //         .expect("Token not found")
        // },
        _ => panic!("Fatal error: unsupported vocab type"),
    }
}
#[derive(Debug, Clone)]
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
    token_to_id: HashMap<String, TokenId>,
    special_tokens: Vec<TokenId>,
    id_to_token: Vec<TokenData>,
}
fn tokenize(raw_text: String, add_special: bool, parse_special: bool) {
    let config = get_config();

    let mut fragment_buffer = if !raw_text.is_empty() {
        FragmentBufferVariant::new_raw_text(raw_text.clone(), 0, raw_text.len() as i64)
    } else {
        // TODO 待完善
        FragmentBufferVariant::new_raw_text(raw_text.clone(), 0, raw_text.len() as i64)
    };
    match config.vocab_type {
        VocabType::None => todo!(),
        VocabType::Spm => todo!(),
        VocabType::Bpe => todo!(),
        VocabType::Wpm => todo!(),
        VocabType::Ugm => todo!(),
        VocabType::Rwkv => todo!(),
    }
}

fn tokenizer_st_partition(buffer: &mut LinkedList<FragmentBufferVariant>, parse_special: bool) {
    let config = get_config();
    // 遍历每个特殊标记
    for special_id in &config.special_tokens {
        let data = config.id_to_token[*special_id as usize].clone();
        let text = &data.text;

        // 如果不解析特殊标记且当前标记是控制标记或未知标记，则跳过
        if !parse_special
            && ((data.attribute as u32)
                & (TokenAttribute::Control as u32 | TokenAttribute::Unknown as u32))
                != 0
        {
            continue;
        }

        // 遍历每个文本片段
        let mut cursor = buffer.cursor_front_mut();
        while let Some(fragment) = cursor.current() {
            // 如果片段是原始文本（尚未处理）
            if fragment.variant_type == FragmentBufferVariantType::RawText {
                let FragmentBufferVariant {
                    raw_text,
                    offset,
                    length,
                    ..
                } = &fragment.clone();
                let mut raw_text_base_offset = *offset;
                let mut raw_text_base_length = *length;

                // 在文本中循环查找特殊标记
                loop {
                    // 在当前片段中查找特殊标记的第一次出现
                    let text_slice = &raw_text[raw_text_base_offset as usize
                        ..(raw_text_base_offset + raw_text_base_length) as usize];
                    let match_pos = text_slice.find(text);

                    // 如果没有找到，停止处理该片段
                    let match_pos = match match_pos {
                        None => break,
                        Some(pos) => raw_text_base_offset as usize + pos,
                    };

                    // 如果匹配位置在基础偏移量之后，处理左侧文本
                    if match_pos > raw_text_base_offset as usize {
                        let left_reminder_offset = raw_text_base_offset as i64;
                        let mut left_reminder_length =
                            match_pos as i64 - raw_text_base_offset as i64;

                        // 如果需要去除左侧空白
                        if (data.attribute as u32 & TokenAttribute::LStrIp as u32) != 0 {
                            while left_reminder_length > 0 {
                                let last_char = raw_text
                                    .chars()
                                    .nth((left_reminder_offset + left_reminder_length - 1) as usize)
                                    .unwrap();
                                if !last_char.is_whitespace() {
                                    break;
                                }
                                left_reminder_length -= 1;
                            }
                        }

                        // 插入左侧文本片段
                        if left_reminder_length > 0 {
                            cursor.insert_after(
                                FragmentBufferVariant::new_raw_text(
                                    raw_text.clone(),
                                    left_reminder_offset,
                                    left_reminder_length,
                                )
                                .unwrap(),
                            );
                            cursor.move_next();
                        }
                    }

                    // 插入特殊标记
                    cursor.insert_after(FragmentBufferVariant::new_token(*special_id));
                    cursor.move_next();

                    // 处理右侧文本
                    let right_start = match_pos + text.len();
                    if right_start < (raw_text_base_offset + raw_text_base_length) as usize {
                        let mut right_reminder_offset = right_start as i64;
                        let mut right_reminder_length = raw_text_base_length
                            - ((match_pos as u64 - raw_text_base_offset as u64)
                                + text.len() as u64);

                        // 如果需要去除右侧空白
                        if (data.attribute as u32 & TokenAttribute::RStrIp as u32) != 0 {
                            while right_reminder_length > 0 {
                                let next_char = raw_text
                                    .chars()
                                    .nth(right_reminder_offset as usize)
                                    .unwrap();
                                if !next_char.is_whitespace() {
                                    break;
                                }
                                right_reminder_offset += 1;
                                right_reminder_length -= 1;
                            }
                        }

                        // 插入右侧文本片段
                        if right_reminder_length > 0 {
                            cursor.insert_after(
                                FragmentBufferVariant::new_raw_text(
                                    raw_text.clone(),
                                    right_reminder_offset,
                                    right_reminder_length as i64,
                                )
                                .unwrap(),
                            );
                            cursor.move_next();
                        }

                        // 继续处理右侧文本
                        raw_text_base_offset = right_reminder_offset as u64;
                        raw_text_base_length = right_reminder_length;
                    } else {
                        // 删除当前片段并退出循环
                        cursor.remove_current();
                        break;
                    }
                }
            }
            cursor.move_next();
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum FragmentBufferVariantType {
    Token,
    RawText,
}

#[derive(Debug, Clone)]
struct FragmentBufferVariant {
    variant_type: FragmentBufferVariantType,
    token: u32, // 假设 llama_token 是 i32 类型
    raw_text: String,
    offset: u64,
    length: u64,
}
impl FragmentBufferVariant {
    // 创建 Token 类型的变体
    fn new_token(token: u32) -> Self {
        Self {
            variant_type: FragmentBufferVariantType::Token,
            token,
            raw_text: String::new(),
            offset: 0,
            length: 0,
        }
    }

    // 创建 RawText 类型的变体
    fn new_raw_text(text: String, offset: i64, length: i64) -> Result<Self, &'static str> {
        // 参数验证
        if offset < 0 {
            return Err("offset must be non-negative");
        }
        if length < 1 {
            return Err("length must be positive");
        }
        if (offset + length) as usize > text.len() {
            return Err("offset + length exceeds text length");
        }

        Ok(Self {
            variant_type: FragmentBufferVariantType::RawText,
            token: NULL,
            raw_text: text,
            offset: offset as u64,
            length: length as u64,
        })
    }
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
            token_to_id: HashMap::new(),
            special_tokens: Vec::new(),
            id_to_token: todo!(),
        }
    }
}
#[derive(Debug, Clone)]
struct TokenData {
    pub text: String,
    pub score: f32,
    pub attribute: TokenAttribute,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
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
