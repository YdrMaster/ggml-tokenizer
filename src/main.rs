#![feature(linked_list_cursors)]
use common::{GLOBAL_CONFIG, NULL, TokenAttribute};
use config::{VocabType, load};

use memmap2::Mmap;
use untils::llama_escape_whitespace;

use std::collections::LinkedList;

use std::fs::File;

mod common;
mod config;
mod session;
mod unicode;
mod untils;
fn main() {
    let prompt = "Hello my name is";
    let path = std::env::args_os().nth(1).unwrap();
    let file = File::open(path).unwrap();
    let file = unsafe { Mmap::map(&file) }.unwrap();
    load(file);
    let binding = GLOBAL_CONFIG.read().unwrap();
    let config = binding.as_ref().unwrap();
    // let tmp=config.tokenize(prompt, true, true);
    // print!("test {:?}", tmp);
    // println!("{}", gguf.general_architecture().unwrap())
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
