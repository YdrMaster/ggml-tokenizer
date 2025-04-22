#![feature(linked_list_cursors)]
use ggus::{GGuf, GGufMetaMapExt};
use memmap2::Mmap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, LinkedList, VecDeque};
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
fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let file = File::open(path).unwrap();
    let file = unsafe { Mmap::map(&file) }.unwrap();
    load(file);
    let config = get_config();
    print!("{:?}", config)

    // println!("{}", gguf.general_architecture().unwrap())
}

// load 函数
fn load(file: Mmap) {
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
    config.bpe_ranks = bpe_ranks;
    GLOBAL_CONFIG.set(config).unwrap();
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
    bpe_ranks: HashMap<(String, String), usize>,
}
impl TokenizerConfig {
    /// 将文本字符串转换为标记 ID
    ///
    /// 如果文本在词汇表中存在，返回对应的标记 ID
    /// 否则返回 LLAMA_TOKEN_NULL
    pub fn text_to_token(&self, text: &str) -> TokenId {
        // 在 token_to_id 映射中查找文本
        if let Some(token_id) = self.token_to_id.get(text) {
            return *token_id;
        } else {
            NULL
        }
    }
    pub fn n_tokens(&self) -> u32 {
        self.id_to_token.len() as u32
    }
    pub fn get_token_data(&self, id: TokenId) -> TokenData {
        self.id_to_token[id as usize].clone()
    }
    /// 将单个字节转换为标记 ID
    pub fn byte_to_token(&self, ch: u8) -> TokenId {
        // 十六进制字符数组
        static HEX: &[u8; 16] = b"0123456789ABCDEF";

        match self.vocab_type {
            VocabType::Spm | VocabType::Ugm => {
                // 创建格式为 "<0xXY>" 的字符串，其中 XY 是字节的十六进制表示
                let buf = format!(
                    "<0x{}{}>",
                    HEX[(ch >> 4) as usize] as char,
                    HEX[(ch & 15) as usize] as char
                );

                // 尝试在词汇表中查找该字符串
                if let Some(token) = self.token_to_id.get(&buf) {
                    return *token;
                }

                // 如果找不到，尝试回退到仅将字节作为字符串
                let buf2 = String::from_utf8_lossy(&[ch]).to_string();

                // 使用 at 方法获取标记 ID，如果不存在则会 panic
                *self.token_to_id.get(&buf2).expect("无法找到字节对应的标记")
            }

            VocabType::Wpm | VocabType::Bpe => {
                // 对于 WPM 和 BPE 类型，使用 unicode_byte_to_utf8 函数
                let utf8_str = unicode_byte_to_utf8(ch);

                // 使用 at 方法获取标记 ID，如果不存在则会 panic
                *self
                    .token_to_id
                    .get(&utf8_str)
                    .expect("无法找到字节对应的标记")
            }

            _ => {
                // 对于其他类型，终止程序
                panic!("致命错误：不支持的词汇表类型")
            }
        }
    }
    fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        match self
            .bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
        {
            Some(rank) => *rank as i32,
            None => -1,
        }
    }
}

/// 将单个字节转换为 UTF-8 字符串
fn unicode_byte_to_utf8(ch: u8) -> String {
    String::from_utf8_lossy(&[ch]).to_string()
}
fn tokenize(raw_text: String, add_special: bool, parse_special: bool) {
    let config = get_config();
    let mut buffer = LinkedList::new();
    let mut output = Vec::new();
    tokenizer_st_partition(&mut buffer, parse_special);
    let mut fragment_buffer = if !raw_text.is_empty() {
        FragmentBufferVariant::new_raw_text(raw_text.clone(), 0, raw_text.len() as i64)
    } else {
        unreachable!()
    };
    match config.vocab_type {
        VocabType::None => todo!(),
        VocabType::Spm => {
            let mut is_prev_special = true; // prefix with space if first token
            if add_special && config.add_bos {
                output.push(config.bos);
                is_prev_special == true;
            }
            for fragment in buffer.iter_mut() {
                let substring = &fragment.raw_text
                    [(fragment.offset as usize)..(fragment.offset + fragment.length) as usize];
                let mut text = String::new();
                if fragment.variant_type == FragmentBufferVariantType::RawText {
                    if config.add_space_prefix && is_prev_special {
                        text.push(' ');
                    }
                    text.push_str(substring);

                    llama_escape_whitespace(&mut text);

                    todo!();
                    is_prev_special = false;
                } else {
                    output.push(fragment.token);
                    is_prev_special == true;
                }
                // 检查是否有重复的 BOS 标记
                if add_special && config.add_bos && output.len() >= 2 && output[1] == config.bos {
                    log::warn!(
                        " Added a BOS token to the prompt as specified by the model but the prompt"
                    );
                }

                // 添加 EOS 标记
                if add_special && config.add_eos {
                    output.push(config.eos);
                }
            }
        }
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
            id_to_token: Vec::new(),
            bpe_ranks: HashMap::new(),
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
/// 将字符串中的所有空格替换为特殊的 Unicode 字符 U+2581（下八分之一块）
pub fn llama_escape_whitespace(text: &mut String) {
    // 使用 Rust 的 replace_all 方法替换所有空格
    *text = text.replace(" ", "\u{2581}");
}
/// 符号结构体，表示文本中的一个符号
#[derive(Clone, Debug)]
pub struct LlmSymbol<'a> {
    /// 前一个符号的索引
    pub prev: i32,
    /// 下一个符号的索引
    pub next: i32,
    /// 指向原始文本的指针
    pub text: &'a str,
    /// 符号的长度
    pub n: usize,
}

/// 二元组结构体，用于表示两个相邻的符号
#[derive(Clone, Debug)]
pub struct LlmBigramSpm {
    /// 左侧符号的索引
    pub left: i32,
    /// 右侧符号的索引
    pub right: i32,
    /// 二元组的分数
    pub score: f32,
    /// 二元组的大小
    pub size: usize,
}

/// 为 LlmBigramSpm 实现 PartialEq
impl PartialEq for LlmBigramSpm {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}

/// 为 LlmBigramSpm 实现 Eq
impl Eq for LlmBigramSpm {}

/// 为 LlmBigramSpm 实现 PartialOrd
impl PartialOrd for LlmBigramSpm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 为 LlmBigramSpm 实现 Ord，用于优先队列
impl Ord for LlmBigramSpm {
    fn cmp(&self, other: &Self) -> Ordering {
        // 注意：这里是反向比较，因为我们需要最大堆
        // 首先比较分数，然后比较左侧索引
        match other.score.partial_cmp(&self.score) {
            Some(Ordering::Equal) => other.left.cmp(&self.left),
            Some(ord) => ord,
            None => Ordering::Equal, // 处理 NaN 情况
        }
    }
}

/// SPM 标记器会话结构体
pub struct LlmTokenizerSpmSession<'a> {
    /// 符号列表
    symbols: Vec<LlmSymbol<'a>>,
    /// 工作队列
    work_queue: BinaryHeap<LlmBigramSpm>,
    /// 反向合并映射
    rev_merge: HashMap<String, (i32, i32)>,
}

impl<'a> LlmTokenizerSpmSession<'a> {
    /// 创建一个新的 SPM 标记器会话
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    /// 标记化文本
    pub fn tokenize(&mut self, text: &'a str, output: &mut Vec<u32>) {
        // 将字符串分割为 UTF-8 字符
        let mut index = 0;
        let mut offs = 0;

        self.symbols.clear();

        while offs < text.len() {
            // 获取当前字符的 UTF-8 长度
            let len = unicode_len_utf8(text.as_bytes()[offs]);

            // 创建新的符号
            let sym = LlmSymbol {
                text: &text[offs..],
                n: std::cmp::min(len, text.len() - offs),
                prev: index - 1,
                next: if offs + len >= text.len() {
                    -1
                } else {
                    index + 1
                },
            };

            offs += sym.n;
            index += 1;
            self.symbols.push(sym);
        }

        // 用所有可能的 2 字符标记初始化工作队列
        for i in 1..self.symbols.len() {
            self.try_add_bigram(i as i32 - 1, i as i32);
        }

        // 持续替换频率最高的对，直到不能再替换
        while let Some(bigram) = self.work_queue.pop() {
            let left_idx = bigram.left as usize;
            let right_idx = bigram.right as usize;

            // 获取左右符号的可变引用
            // 注意：这里需要小心处理可变借用规则
            let left_sym_n = self.symbols[left_idx].n;
            let right_sym_n = self.symbols[right_idx].n;

            // 如果其中一个符号已经被合并，跳过它
            if left_sym_n == 0 || right_sym_n == 0 || left_sym_n + right_sym_n != bigram.size {
                continue;
            }

            // 将右符号合并到左符号中
            self.symbols[left_idx].n += right_sym_n;
            self.symbols[right_idx].n = 0;

            // 从链中移除右符号
            let right_next = self.symbols[right_idx].next;
            self.symbols[left_idx].next = right_next;

            if right_next >= 0 {
                self.symbols[right_next as usize].prev = bigram.left;
            }

            // 寻找更多替换
            self.try_add_bigram(self.symbols[left_idx].prev, bigram.left);
            self.try_add_bigram(bigram.left, self.symbols[left_idx].next);
        }

        // 处理最终的符号
        let mut i = 0;
        while i != -1 {
            let symbol = &self.symbols[i as usize];
            self.resegment(symbol, output);
            i = symbol.next;
        }
    }

    /// 尝试添加新的二元组
    fn try_add_bigram(&mut self, left: i32, right: i32) {
        let config = get_config();
        if left == -1 || right == -1 {
            return;
        }

        // 获取左右符号的文本
        let left_sym = &self.symbols[left as usize];
        let right_sym = &self.symbols[right as usize];

        // 构建完整的文本
        let left_text = &left_sym.text[..left_sym.n];
        let right_text = &right_sym.text[..right_sym.n];
        let text = format!("{}{}", left_text, right_text);

        // 查找标记
        let token = config.text_to_token(&text);

        if token == NULL {
            return;
        }

        if token as u32 >= config.n_tokens() {
            return;
        }

        // 获取标记数据
        let tok_data = config.get_token_data(token);

        // 创建新的二元组
        let bigram = LlmBigramSpm {
            left,
            right,
            score: tok_data.score,
            size: text.len(),
        };

        // 添加到工作队列
        self.work_queue.push(bigram);

        // 添加到反向合并映射
        self.rev_merge.insert(text, (left, right));
    }

    /// 重新分割符号
    fn resegment(&self, symbol: &LlmSymbol<'a>, output: &mut Vec<u32>) {
        let config = get_config();
        // 获取符号的文本
        let text = &symbol.text[..symbol.n];

        // 尝试将文本转换为标记
        let token = config.text_to_token(text);

        // 如果找到了标记，直接添加
        if token != NULL {
            output.push(token);
            return;
        }

        // 查找反向合并映射
        if let Some(&(left, right)) = self.rev_merge.get(text) {
            // 递归处理左右符号
            self.resegment(&self.symbols[left as usize], output);
            self.resegment(&self.symbols[right as usize], output);
            return;
        }

        // 如果没有找到映射，将每个字节作为单独的标记输出
        for i in 0..symbol.n {
            if let Some(byte) = symbol.text.as_bytes().get(i) {
                let id = config.byte_to_token(*byte);
                output.push(id);
            }
        }
    }
}

/// 标记常量
pub const LLAMA_TOKEN_NULL: i32 = -1;

/// BPE 标记器会话结构体
pub struct LlmTokenizerBpeSession<'a> {
    /// 标记器引用
    tokenizer: &'a LlmTokenizerBpe,
    /// 符号列表
    symbols: Vec<LlmSymbol<'a>>,
    /// 最终符号列表
    symbols_final: Vec<LlmSymbol<'a>>,
    /// 工作队列
    work_queue: LlmBigramBpe,
}

impl<'a> LlmTokenizerBpeSession<'a> {
    /// 创建一个新的 BPE 标记器会话
    pub fn new(tokenizer: &'a LlmTokenizerBpe) -> Self {
        Self {
            tokenizer,
            symbols: Vec::new(),
            symbols_final: Vec::new(),
            work_queue: LlmBigramBpe::new(),
        }
    }

    /// 添加标记到输出
    pub fn append(token_id: TokenId, output: &mut Vec<TokenId>) {
        output.push(token_id);
    }

    /// 添加 BOS 标记
    pub fn append_bos(&self, output: &mut Vec<TokenId>) -> bool {
        let config: &TokenizerConfig = get_config();
        if config.add_bos {
            output.push(config.bos);
            return true;
        }
        false
    }

    /// 添加 EOS 标记
    pub fn append_eos(&self, output: &mut Vec<TokenId>) -> bool {
        let config: &TokenizerConfig = get_config();
        if config.add_eos {
            output.push(config.eos);
            return true;
        }
        false
    }

    /// 标记化文本
    pub fn tokenize(&mut self, text: &'a str, output: &mut Vec<TokenId>) {
        let config: &TokenizerConfig = get_config();
        let mut final_prev_index = -1;
        let word_collection = unicode_regex_split(text, &self.tokenizer.regex_exprs);

        self.symbols_final.clear();

        for word in word_collection {
            self.work_queue = LlmBigramBpe::new();
            self.symbols.clear();

            let mut index = 0;
            let mut offset = 0;

            // 如果词汇表忽略合并且单词已经在词汇表中
            if config.ignore_merges && config.text_to_token(word) != NULL {
                //
                self.symbols.push(LlmSymbol {
                    prev: -1,
                    next: -1,
                    text: word,
                    n: word.len(),
                });
                offset = word.len();
            }

            // 将单词分割为 UTF-8 字符
            while offset < word.len() {
                let char_len =
                    (word.len() - offset).min(unicode_len_utf8(word.as_bytes()[offset]) as usize);
                let sym = LlmSymbol {
                    text: &word[offset..],
                    n: char_len,
                    prev: index - 1,
                    next: if offset + char_len == word.len() {
                        -1
                    } else {
                        index + 1
                    },
                };
                offset += sym.n;
                index += 1;
                self.symbols.push(sym);
            }

            // 添加所有可能的二元组
            for i in 1..(self.symbols.len() as i32) {
                self.add_new_bigram(i - 1, i);
            }

            // 构建标记
            while let Some(bigram) = self.work_queue.pop_move() {
                let left_idx = bigram.left as usize;
                let right_idx = bigram.right as usize;

                // 获取左右符号的引用
                let left_symbol = &self.symbols[left_idx];
                let right_symbol = &self.symbols[right_idx];

                // 如果其中一个符号已经被合并，跳过它
                if left_symbol.n == 0 || right_symbol.n == 0 {
                    continue;
                }

                // 创建左右标记的字符串
                let left_token =
                    String::from_utf8_lossy(&left_symbol.text.as_bytes()[..left_symbol.n])
                        .to_string();
                let right_token =
                    String::from_utf8_lossy(&right_symbol.text.as_bytes()[..right_symbol.n])
                        .to_string();

                // 检查二元组是否过时
                if left_token + &right_token != bigram.text {
                    continue;
                }

                // 合并右符号到左符号
                self.symbols[left_idx].n += self.symbols[right_idx].n;

                // 将右符号标记为已合并
                self.symbols[right_idx].n = 0;

                // 从链中移除右符号
                let right_next = self.symbols[right_idx].next;
                self.symbols[left_idx].next = right_next;

                if right_next >= 0 {
                    self.symbols[right_next as usize].prev = bigram.left;
                }

                // 寻找更多合并
                self.add_new_bigram(self.symbols[left_idx].prev, bigram.left);
                self.add_new_bigram(bigram.left, self.symbols[left_idx].next);
            }

            // 将完成的标记添加到最终列表，保持正确的顺序
            for sym in &self.symbols {
                if sym.n > 0 {
                    let mut new_sym = sym.clone();
                    new_sym.prev = final_prev_index;
                    new_sym.next = -1;

                    if final_prev_index != -1 {
                        self.symbols_final[final_prev_index as usize].next =
                            self.symbols_final.len() as i32;
                    }

                    self.symbols_final.push(new_sym);
                    final_prev_index = (self.symbols_final.len() - 1) as i32;
                }
            }
        }

        // 使用最终符号列表
        self.symbols = self.symbols_final.clone();

        // 处理所有符号
        if !self.symbols.is_empty() {
            let mut i = 0;
            while i != -1 {
                let symbol = &self.symbols[i as usize];
                if symbol.n > 0 {
                    // 创建符号的字符串
                    let str =
                        String::from_utf8_lossy(&symbol.text.as_bytes()[..symbol.n]).to_string();
                    let token = config.text_to_token(&str);

                    if token == NULL {
                        // 如果找不到标记，将每个字节作为单独的标记输出
                        for byte in str.bytes() {
                            let byte_str = String::from(byte as char);
                            let token_multibyte = config.text_to_token(&byte_str);
                            if token_multibyte != NULL {
                                output.push(token_multibyte);
                            }
                        }
                    } else {
                        // 添加找到的标记
                        output.push(token);
                    }
                }
                i = symbol.next;
            }
        }
    }

    /// 添加新的二元组
    fn add_new_bigram(&mut self, left: i32, right: i32) {
        let config: &TokenizerConfig = get_config();
        if left == -1 || right == -1 {
            return;
        }

        let left_token = &self.symbols[left as usize].text[..self.symbols[left as usize].n];
        let right_token = &self.symbols[right as usize].text[..self.symbols[right as usize].n];

        let mut rank_found = -1;

        rank_found = config.find_bpe_rank(left_token, right_token);

        if rank_found < 0 {
            return;
        }

        let bigram = LlmBigramBpeItem {
            left,
            right,
            text: format!("{}{}", left_token, right_token),
            size: left_token.len() + right_token.len(),
            rank: rank_found,
        };

        self.work_queue.push(bigram);
    }
}

/// BPE 二元组项结构体
#[derive(Clone, Debug)]
pub struct LlmBigramBpeItem {
    /// 左侧符号的索引
    pub left: i32,
    /// 右侧符号的索引
    pub right: i32,
    /// 二元组的文本
    pub text: String,
    /// 二元组的排名
    pub rank: i32,
    /// 二元组的大小
    pub size: usize,
}

/// BPE 二元组优先队列
pub struct LlmBigramBpe {
    /// 内部队列
    queue: VecDeque<LlmBigramBpeItem>,
}

impl LlmBigramBpe {
    /// 创建一个新的 BPE 二元组优先队列
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// 添加二元组到队列
    pub fn push(&mut self, item: LlmBigramBpeItem) {
        self.queue.push_back(item);
    }

    /// 弹出并移动二元组
    pub fn pop_move(&mut self) -> Option<LlmBigramBpeItem> {
        self.queue.pop_front()
    }

    /// 检查队列是否为空
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

///  BPE 标记器结构体
pub struct LlmTokenizerBpe {
    /// 正则表达式列表
    pub regex_exprs: Vec<String>,
}
/// 使用正则表达式分割 Unicode 文本
fn unicode_regex_split<'a>(text: &str, regex_exprs: &[String]) -> Vec<&'a str> {
    // 实际实现
    todo!()
}

/// 获取 UTF-8 字符的长度
fn unicode_len_utf8(byte: u8) -> usize {
    if byte & 0x80 == 0 {
        1
    } else if byte & 0xE0 == 0xC0 {
        2
    } else if byte & 0xF0 == 0xE0 {
        3
    } else if byte & 0xF8 == 0xF0 {
        4
    } else {
        1 // 无效的 UTF-8 序列，返回 1
    }
}
