/// 将单个字节转换为 UTF-8 字符串
pub fn unicode_byte_to_utf8(ch: u8) -> String {
    String::from_utf8_lossy(&[ch]).to_string()
}

/// 使用正则表达式分割 Unicode 文本
pub fn unicode_regex_split<'a>(text: &str, regex_exprs: &[String]) -> Vec<&'a str> {
    // 实际实现
    todo!()
}

/// 获取 UTF-8 字符的长度
pub fn unicode_len_utf8(byte: u8) -> usize {
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
