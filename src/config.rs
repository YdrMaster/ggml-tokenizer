use std::collections::{HashMap, HashSet, LinkedList};

use ggus::{GGuf, GGufMetaMapExt};
use memmap2::Mmap;

use crate::{
    FragmentBufferVariant, FragmentBufferVariantType,
    common::{BPE_SESSION, GLOBAL_CONFIG, NULL, SPM_SESSION, TokenAttribute, TokenData, TokenId},
    session, tokenizer_st_partition,
    unicode::unicode_byte_to_utf8,
    untils::{get_bool, llama_escape_whitespace},
};

// 获取或初始化全局配置的函数
pub fn get_config() -> &'static TokenizerConfig {
    GLOBAL_CONFIG.get_or_init(|| {
        println!("Initializing TokenizerConfig for the first time..."); // 仅在首次调用时打印
        // 这里创建 TokenizerConfig 的实例
        TokenizerConfig::new()
    })
}

//  load 函数 默认都是gpt2
pub fn load(file: Mmap) {
    let gguf = GGuf::new(&file).unwrap();

    let model = gguf.tokenizer_ggml_model().unwrap();
    // 初始化session
    println!("tokenizer model = {model}");
    assert_eq!(model, "gpt2");

    let mut config = TokenizerConfig::new();
    // 此处等同于llama.cpp的合并
    let bpe_ranks = load_gpt2(&gguf);

    println!("gpt2 n rank = {}", bpe_ranks.len());
    // 设置预设字段
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
    config.vocab_type = VocabType::Bpe;
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
            let ids = config.tokenize("\n", false, false);
            if ids.is_empty() {
                config.linefeed = config.pad;
            } else {
                config.linefeed = ids[0];
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
    // 为特殊token值为null构建字符,高度类似，能偶抽象成方法或者声明宏
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

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VocabType {
    None = 0, // For models without vocab
    Spm = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
    Bpe = 2,  // GPT-2 tokenizer based on byte-level BPE
    Wpm = 3,  // BERT tokenizer based on WordPiece
    Ugm = 4,  // T5 tokenizer based on Unigram
    Rwkv = 5, // RWKV tokenizer based on greedy tokenization
}

#[derive(Clone)]
pub struct TokenizerConfig {
    pub vocab_type: VocabType,
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub eom: u32,
    pub unk: u32,
    pub sep: u32,
    pub pad: u32,
    pub fim_pre: u32,
    pub fim_suf: u32,
    pub fim_mid: u32,
    pub fim_pad: u32,
    pub fim_rep: u32,
    pub fim_sep: u32,
    pub linefeed: u32,
    pub mask: u32,
    pub add_space_prefix: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub ignore_merges: bool,
    pub clean_spaces: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
    pub token_to_id: HashMap<String, TokenId>,
    pub special_tokens: Vec<TokenId>,
    pub id_to_token: Vec<TokenData>,
    pub bpe_ranks: HashMap<(String, String), usize>,
}
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
    pub fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        match self
            .bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
        {
            Some(rank) => *rank as i32,
            None => -1,
        }
    }
    pub fn tokenize<'a>(
        &self,
        raw_text: &'a str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let mut buffer = LinkedList::new();
        let mut output = Vec::new();
        if !raw_text.is_empty() {
            buffer.push_front(
                FragmentBufferVariant::new_raw_text(raw_text.to_string(), 0, raw_text.len() as i64)
                    .unwrap(),
            );
            tokenizer_st_partition(&mut buffer, parse_special);
        }
        match self.vocab_type {
            VocabType::None => todo!(),
            VocabType::Spm => {
                let mut is_prev_special = true; // prefix with space if first token
                if add_special && self.add_bos {
                    output.push(self.bos);
                    is_prev_special = true;
                }
                for fragment in buffer.iter_mut() {
                    let substring = &fragment.raw_text
                        [(fragment.offset as usize)..(fragment.offset + fragment.length) as usize];
                    let mut text = String::new();
                    if fragment.variant_type == FragmentBufferVariantType::RawText {
                        if self.add_space_prefix && is_prev_special {
                            text.push(' ');
                        }
                        text.push_str(substring);

                        llama_escape_whitespace(&mut text);
                        todo!();
                        // SPM_SESSION.get_mut().unwrap()
                        //     .tokenize(&text, &mut output);
                        is_prev_special = false;
                    } else {
                        output.push(fragment.token);
                        is_prev_special = true;
                    }
                    // 检查是否有重复的 BOS 标记
                    if add_special && self.add_bos && output.len() >= 2 && output[1] == self.bos {
                        log::warn!(
                            " Added a BOS token to the prompt as specified by the model but the prompt"
                        );
                    }

                    // 添加 EOS 标记
                    if add_special && self.add_eos {
                        output.push(self.eos);
                    }
                }
            }
            VocabType::Bpe => {
                let mut session_ref = BPE_SESSION.lock().unwrap();
                for fragment in buffer.iter_mut() {
                    if fragment.variant_type == FragmentBufferVariantType::RawText {
                        let substring = &fragment.raw_text[(fragment.offset as usize)
                            ..(fragment.offset + fragment.length) as usize];
                        session_ref.tokenize(substring, &mut output);
                    } else {
                        session_ref.append_bos(&mut output);
                    }
                }
                if add_special {
                    session_ref.append_eos(&mut output);
                }
            }
            VocabType::Wpm => todo!(),
            VocabType::Ugm => todo!(),
            VocabType::Rwkv => todo!(),
        }
        output
    }
}
impl std::fmt::Debug for TokenizerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerConfig")
            // 这里只添加您想要显示的字段
            .field("vocab_type", &self.vocab_type)
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("add_bos", &self.add_bos)
            .field("add_eos", &self.add_eos)
            .field("add_space_prefix", &self.add_space_prefix)
            // 不添加您不想显示的字段：token_to_id, special_tokens, id_to_token, bpe_ranks
            .finish()
    }
}
