# 分词器

## 加载流程

### load

首先要模型的`tokenizer.ggml.model`初始化模型的一些字段。 [第一次初始化](https://github.com/YdrMaster/ggml-tokenizer/blob/5466304df0f80ab380d9504bd29a69b87931e9f9/src/config.rs#L38)

读取模型的词汇表和相关属性，构建词汇表和bpe_ranks。[加载词汇表](https://github.com/YdrMaster/ggml-tokenizer/blob/5466304df0f80ab380d9504bd29a69b87931e9f9/src/config.rs#L100?)

通过词汇表自动纠正错误的特殊词汇 [矫正特殊词汇的id](https://github.com/YdrMaster/ggml-tokenizer/blob/5466304df0f80ab380d9504bd29a69b87931e9f9/src/config.rs#L146)

构建终止字符的id列表[special_eog_ids](https://github.com/YdrMaster/ggml-tokenizer/blob/5466304df0f80ab380d9504bd29a69b87931e9f9/src/config.rs#L300)

收集特殊字符的id列表[special_tokens](https://github.com/YdrMaster/ggml-tokenizer/blob/5466304df0f80ab380d9504bd29a69b87931e9f9/src/config.rs#L337)

