#![feature(linked_list_cursors)]
use common::{NULL, TokenAttribute};
use config::{VocabType, get_config, load};

use memmap2::Mmap;
use untils::llama_escape_whitespace;

use std::collections::{BinaryHeap, LinkedList, VecDeque};

use std::{collections::HashMap, fs::File};

mod common;
mod config;
mod session;
mod unicode;
mod untils;
fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let file = File::open(path).unwrap();
    let file = unsafe { Mmap::map(&file) }.unwrap();
    load(file);
    let config = get_config();
    print!("{:?}", config)

    // println!("{}", gguf.general_architecture().unwrap())
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
                is_prev_special = true;
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
                    is_prev_special = true;
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
