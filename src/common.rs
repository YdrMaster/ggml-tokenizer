use std::sync::OnceLock;

use crate::config::TokenizerConfig;

pub const NULL: u32 = u32::MAX;
pub type TokenId = u32;
pub static GLOBAL_CONFIG: OnceLock<TokenizerConfig> = OnceLock::new();
#[derive(Debug, Clone)]
pub struct TokenData {
    pub text: String,
    pub score: f32,
    pub attribute: TokenAttribute,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum TokenAttribute {
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
