---
layout: default
title: Tokenization Video Conversion
---

# [Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)

### Table of Contents

1. [The Nitty-Gritty of Tokenization in GPT](#nitty-gritty-tokenization-gpt)    
   - [Downloading the Dataset](#downloading-dataset)    
   - [Exploring the Dataset](#exploring-dataset)    
   - [Understanding Character-Level Tokenization](#character-level-tokenization-understanding)    
   - [Creating a Character to Integer Mapping](#creating-char-int-mapping)    
   - [Encoding the Entire Dataset](#encoding-entire-dataset)    
   - [Embedding Tokens into the Model](#embedding-tokens-model)    
   - [Inspecting Input to the Transformer](#inspecting-input-transformer)    
2. [Diving Deeper into Tokenization](#diving-deeper-into-tokenization)    
   - [Character to Integer Mapping Revisited](#char-int-mapping-revisited)    
   - [From Characters to Chunks: BPE Algorithm](#characters-to-chunks-bpe)    
   - [Tokenization in LLAMA 2](#tokenization-llama-2)    
   - [Building Our Own Tokenizer](#building-own-tokenizer)    
   - [Tokenization Pitfalls](#tokenization-pitfalls)    
   - [Visualizing Tokenization with a Webapp](#visualizing-tokenization-webapp)    
   - [Arbitrary Token Splits](#arbitrary-token-splits)    
   - [Case Sensitivity and Context](#case-sensitivity-context)    
   - [Exploring Tokenization in Different Languages](#tokenization-different-languages)    
3. [The Challenge with Non-English Languages](#non-english-languages-challenge)    
   - [Tokenization Bias Toward English](#tokenization-bias-toward-english)    
   - [Tokenization of Python Code](#tokenization-python-code)    
   - [Improved Tokenization in GPT-4](#improved-tokenization-gpt4)    
   - [Unicode and the Nature of Text](#unicode-nature-of-text)    
   - [The Intricacies of Unicode](#intricacies-unicode)    
   - [The Task Ahead: Tokenizer Implementation](#tokenizer-implementation-task)    
4. [Exploring Unicode and Character Encodings](#unicode-and-character-encodings)    
   - [Accessing Unicode Code Points in Python](#accessing-unicode-code-points)    
   - [The Challenge with Direct Unicode Code Points](#challenge-unicode-code-points)    
   - [The Quest for a Better Encoding](#quest-better-encoding)    
   - [Unicode Encodings Overview](#unicode-encodings-overview)    
   - [UTF-8: The Preferred Encoding for Text](#utf8-preferred-encoding)    
   - [Unicode: The Programmer's Perspective](#unicode-programmers-perspective)    
   - [UTF-8 Everywhere: A Manifesto](#utf8-everywhere-manifesto)    
   - [Encoding in Practice](#encoding-in-practice)    
5. [Unicode and UTF-8 in Python Tokenization](#unicode-utf8-python-tokenization)    
   - [Encoding Strings in Python](#encoding-strings-python)    
   - [Byte Pair Encoding (BPE): A Solution](#byte-pair-encoding)    
   - [BPE Implementation for Tokenization](#bpe-implementation-tokenization)    
   - [BPE in Action](#bpe-in-action)    
6. [UTF-8 Token Streams in Practice](#utf8-token-streams)    
   - [Converting Text to UTF-8 Tokens](#converting-text-utf8-tokens)    
   - [Frequency of Byte Pairs](#frequency-byte-pairs)    
   - [Iterating Over Byte Pairs](#iterating-byte-pairs)    
   - [BPE in Practice](#byte-pair-encoding-in-action)    
   - [Implementing the `get_stats` Function](#implementing-get-stats)    
   - [Advancing the BPE Algorithm](#advancing-bpe-algorithm)    
   - [Identifying Merge Candidates](#identifying-merge-candidates)    
   - [Minimum Index Pair](#minimum-index-pair)    
   - [Handling Special Cases](#handling-special-cases)    
   - [Complete Merge Process](#complete-merge-process)    
   - [Importance of Merges](#importance-of-merges)    
   - [Edge Cases and Test Scenarios](#edge-cases-encoding-function)    
   - [Testing the Tokenizer](#testing-tokenizer)    
   - [Tokenizer Parameters](#tokenizer-parameters)    
   - [Merging Loop in BPE](#merging-loop-bpe)    
   - [Compression Achieved Through BPE Merges](#compression-bpe-merges)    
   - [Training the Tokenizer](#training-the-tokenizer)    
   - [Encoding and Decoding with the Trained Tokenizer](#encoding-decoding-trained-tokenizer)    
   - [Separation of Tokenization from Model Training](#separation-tokenization-model-training)    
   - [Encoding and Decoding: Bridging Tokens and Text](#encoding-decoding-tokens-text)    
   - [Implementing the BPE Merge Function](#implementing-bpe-merge-function)    
   - [Understanding the Impact of Merging](#understanding-impact-of-merging)    
   - [Iterative Merging in BPE](#iterative-merging-in-bpe)    
   - [Effect of Tokenization on Text Analysis](#effect-tokenization-text-analysis)    
   - [Continual Learning and Tokenization](#continual-learning-tokenization)    
   - [Finding the Sweet Spot in Tokenization](#finding-sweet-spot)    
   - [Tuning the Tokenization Hyperparameter](#tuning-tokenization-hyperparameter)    
   - [Leveraging Longer Texts for Better Token Statistics](#leveraging-longer-texts)    
   - [The Merging Loop in BPE](#merging-loop-bpe)    
   - [Exploring Byte Pair Frequencies](#exploring-byte-pair-frequencies)    
   - [Iterating Over and Merging Byte Pairs](#iterating-and-merging-byte-pairs)    
   - [BPE in Action](#byte-pair-encoding-in-action)    
   - [Deep Dive into Python Code for BPE](#python-code-byte-pair-encoding)    
   - [Implementing the BPE Merge Function](#implementing-bpe-merge-function)    
   - [Understanding the Impact of Merging](#understanding-impact-of-merging)    
   - [The Challenge with Non-English Languages](#non-english-languages-challenge)    
   - [Tokenization and Arithmetic](#tokenization-arithmetic)    
   - [Special Tokens as an Attack Surface](#special-tokens-attack-surface)    
   - [The Efficiency of Tokenization](#efficient-tokenization-importance)    
   - [Understanding the Inner Workings of Tokenization](#understanding-tokenization)    
   - [The Ongoing Tokenization Journey](#tokenization-journey-conclusion)    
   - [The Enigma of Unstable Tokens](#unstable-tokens-enigma)    
   - [Trigger Words and Model Misbehavior](#trigger-words-model-misbehavior)    
   - [Understanding Training Space Issues](#training-space-issues)    
   - [The Desired Behavior of Completion APIs](#desired-completion-behavior)    
   - [The Importance of Efficient Tokenization](#efficient-tokenization-importance)    
   - [Understanding the Inner Workings of Tokenization](#understanding-tokenization) 

### The Nitty-Gritty of Tokenization in GPT {#nitty-gritty-tokenization-gpt}

In the beginning, when we approach the subject of tokenization within large language models such as GPT, we must grapple with the dataset we are using for training. The dataset, in this case, is the **tiny Shakespeare dataset**, a staple in the realm of NLP for experimenting and building language models.

#### Downloading the Dataset {#downloading-dataset}

To kick things off, we start by obtaining our dataset, which is conveniently available online. The following script illustrates the process of downloading the tiny Shakespeare dataset:

```bash  
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  
```

Upon executing this command, your system connects to the appropriate server, sends an HTTP request, and retrieves the text file, saving it locally as `input.txt`. The output of this command shows the download speed and confirms the successful retrieval of the file.

#### Exploring the Dataset {#exploring-dataset}

The dataset is a considerable string of text when loaded into Python. To get a feel for our data, we read it in and inspect the first 1000 characters:

```python  
with open('input.txt', 'r', encoding='utf-8') as f:  
    text = f.read()

print(text[:1000])  
```

This snippet of code reads the entire file into a string variable named `text` and then prints out the first 1000 characters, which include dialogues from the First and Second Citizens and others.

#### Understanding Character-Level Tokenization {#character-level-tokenization-understanding}

Our journey into tokenization begins by understanding that the Shakespeare dataset is just text, a long string of characters. To plug this text into a language model, we must first construct a vocabulary. This vocabulary is a collection of unique characters that occur in our text.

Here's how we gather all the unique characters:

```python  
chars = sorted(list(set(text)))  
vocab_size = len(chars)  
print(''.join(chars))  
print(vocab_size)  
```

This code block outputs the sorted list of unique characters and the size of our vocabulary, which happens to be 65 characters. These characters include all the alphabets, punctuation, and special symbols found in the dataset.

#### Creating a Character to Integer Mapping {#creating-char-int-mapping}

Next, we create a lookup table to convert characters to integers and vice versa:

```python  
stoi = { ch:i for i,ch in enumerate(chars) }  
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s if c in stoi]  
decode = lambda l: ''.join([itos[i] for i in l])  
```

The `encode` function maps a string to a list of integers, while the `decode` function does the opposite, taking a list of integers and converting it back to a string. This encoding and decoding are pivotal for understanding how language models process text.

#### Encoding the Entire Dataset {#encoding-entire-dataset}

Once we have our encoding functions, we can encode the entire Shakespeare text and store it as a PyTorch tensor:

```python  
import torch  
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)  
print(data[:1000])  
```

The output shape of the tensor reflects the total number of characters in the dataset, while the snippet of the first 1000 encoded characters gives us a glimpse into the initial part of our encoded dataset.

#### Embedding Tokens into the Model {#embedding-tokens-model}

To understand how tokens are fed into the language model, let's take a look at the BigramLanguageModel class, which uses PyTorch to create a simple language model:

```python  
import torch.nn as nn  
import torch.nn.functional as F  
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):  
    def __init__(self, vocab_size):  
        super().__init__()  
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):  
        logits = self.token_embedding_table(idx)  
        # Additional code for loss computation omitted for brevity  
        return logits, loss  
```

In this model, each token has a corresponding row in the `token_embedding_table`. The integer representing a token is used to look up a row in this table, which contains the trainable parameters for that token. These parameters are optimized during the training process and serve as the inputs to the transformer layers of the language model.

#### Inspecting Input to the Transformer {#inspecting-input-transformer}

By examining the input to the transformer, we get a better sense of how token embeddings are used:

```python  
print(xb) # our input to the transformer  
```

The printed output would show a tensor with integers, each representing an embedded token that the transformer will process. Through this process, the language model learns to predict the next token in a sequence, thereby generating coherent text.

So far, we have journeyed through the initial stages of preparing our dataset, understanding character-level tokenization, and setting the stage for feeding tokenized data into a language model. As we have seen, tokenization is the critical first step in training a language model to understand and generate human-like text.  
### Diving Deeper into Tokenization {#diving-deeper-into-tokenization}

As we progress in our understanding of tokenization, it's important to recognize the evolution from the naive character-level approach we initially discussed. The character-level tokenizer we created earlier was straightforward, yet not quite sophisticated for the needs of state-of-the-art language models.

#### Character to Integer Mapping Revisited {#char-int-mapping-revisited}

To recap, we previously established a character-level tokenizer with mappings from characters to integers and back. Here's the code block that encapsulates this mapping:

```python  
stoi = { ch:i for i,ch in enumerate(chars) }  
itos = { i:ch for i,ch in enumerate(chars) }  
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers  
decode = lambda s: [itos[i] for i in s]  # decoder: take a list of integers, output a string

print(encode("Hello, World!"))  
print(decode([8, 5, 12, 12, 15, 27, 23, 15, 18, 12, 4]))  
```

These functions allow us to convert back and forth between strings and a list of integers, which is necessary for the model to process text data.

#### From Characters to Chunks: BPE Algorithm {#characters-to-chunks-bpe}

In practice, however, language models like GPT-2 use more complex methods for constructing token vocabularies. Instead of individual characters, these methods operate on "character chunks" using algorithms such as Byte Pair Encoding (BPE).

The BPE algorithm was notably mentioned in the GPT-2 paper under the "Input Representation" section, where the authors discuss the tokenization process. They expanded the vocabulary to 50,257 tokens and increased the context size to 1,024 tokens, meaning that each token in the transformer's attention layer can attend to up to 1,024 previous tokens. This reflects the importance of tokenization as a fundamental unit—the "atom"—of large language models.

#### Tokenization in LLAMA 2 {#tokenization-llama-2}

The LLAMA 2 paper further emphasizes the significance of tokens. In the paper, the authors describe their pretraining approach and the changes they made to improve performance. They trained on 40% more total tokens and doubled the context length among other improvements. This is detailed in the paper as follows:

- More robust data cleaning  
- Updated data mixes  
- Trained on 2 trillion tokens  
- Doubled the context length  
- Utilized grouped-query attention for larger models

The LLAMA 2 project also released various models for research and commercial use, with sizes ranging from 7 billion to 70 billion parameters. The paper underscores the importance of safe deployment, providing a responsible use guide and code examples to the community.

#### Building Our Own Tokenizer {#building-own-tokenizer}

With an understanding of the complexities of tokenization and its role in the functionality of LLMs, we can begin to construct our own tokenizer. While the BPE algorithm is not overly complicated, it is essential to understand how it affects the processing of text data.

#### Tokenization Pitfalls {#tokenization-pitfalls}

Tokenization is at the heart of many issues you might encounter with LLMs. Some common challenges attributed to tokenization include:

- Difficulty in spelling words correctly  
- Challenges with simple string processing tasks like reversing a string  
- Poor performance in non-English languages, such as Japanese  
- Struggles with simple arithmetic computations  
- Trouble coding in Python, as observed with GPT-2  
- Abrupt halts when encountering specific strings  
- Warnings about trailing whitespace

These issues demonstrate that many problems that may seem related to the neural network architecture actually stem from tokenization.

#### Visualizing Tokenization with a Webapp {#visualizing-tokenization-webapp}

To further illustrate tokenization, let's explore a webapp that visualizes the process. The following screenshot from the webapp shows various strings being tokenized:

![Tokenization Webapp](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/362_3OJEm4u99.jpeg)

For instance, the word "tokenization" gets split into tokens `30642` and `1634`. The token " is" (including the space) is indexed as `318`. It's important to note the inclusion of whitespace in the tokenization process, as it can significantly impact the interpretation of text.

Additionally, numbers are tokenized in a seemingly arbitrary manner. The number `127` is a single token, while `677` becomes two tokens: " 6" and "77". This inconsistency requires the model to learn during training that these tokens combine to represent the actual number.

#### Arbitrary Token Splits {#arbitrary-token-splits}

Tokenization can result in arbitrary splits, as shown in the example of the number `1275`, which becomes "12" and "75", while `6773` is tokenized as " 6" and "773". The model must make sense of these splits and correctly predict outcomes based on them.

#### Case Sensitivity and Context {#case-sensitivity-context}

The tokenization process is also case-sensitive, as highlighted by the different tokens generated for "Egg," "I have an Egg," "egg," and "EGG." These variations result in distinct tokens, showcasing the model's need to understand the same concept in various contexts and forms.

#### Exploring Tokenization in Different Languages {#tokenization-different-languages}

The complexities of tokenization extend to different languages as well. For instance, in Korean, the introduction from OpenAI's ChatGPT is tokenized, displaying the model's capability to handle diverse scripts and languages.

```python  
# Example of tokenizing a Korean phrase  
korean_phrase = "만나서 반가워요. 저는 OpenAI에서 개발한 대화형 인공지능 ChatGPT입니다. 궁금한 것이 있으시면 무엇이든 물어보세요."  
korean_tokens = encode(korean_phrase)  
print(korean_tokens)  
```

#### Conclusion {#conclusion-no-summary}

Tokenization's integral role in the behavior and performance of large language models cannot be overstated. As we've seen through various examples and explanations, the way text is tokenized directly affects the model's ability to process and generate language. Understanding tokenization is crucial for developers and researchers who wish to leverage the power of LLMs effectively. 

Stay tuned as we continue to delve into the intricacies of tokenization and how it shapes the capabilities of modern AI language models.  
### The Challenge with Non-English Languages {#non-english-languages-challenge}

As we delve further into the nuances of tokenization, it's essential to address the particular challenges that arise with non-English languages. In the case of Korean, for instance, tokenization behaves differently due to the nature of the language and the tokenizer's training data.

#### Tokenization Bias Toward English {#tokenization-bias-toward-english}

A significant portion of the training data for tokenizers like GPT-2 is in English. This imbalance results in shorter tokens for English sentences compared to their translations in languages like Korean or Japanese. Consequently, non-English languages end up with longer sequences of tokens for the same content, which can bloat the sequence length and consume more of the transformer's attention capacity.

Here's a summary of the issue:  
- English sentences might tokenize into around 10 tokens.  
- The same sentence in Korean or Japanese could use a significantly larger number of tokens.  
- Non-English texts are "stretched out" from the transformer's perspective due to longer tokens.  
- This stretching can cause the transformer to run out of context length, hindering performance.

#### Tokenization of Python Code {#tokenization-python-code}

Tokenization isn't only a concern for natural languages—it also impacts programming languages. Let's examine a snippet of Python code to illustrate this point.

```python  
for i in range(1, 101):  
    if i % 3 == 0 and i % 5 == 0:  
        print("FizzBuzz")  
```

Notice how each space in the code can be tokenized as an individual token. This granularity can lead to inefficiency, as GPT-2 might tokenize the spaces separately, bloating the text and consuming valuable sequence space within the transformer's context length.

#### Improved Tokenization in GPT-4 {#improved-tokenization-gpt4}

The evolution from GPT-2 to GPT-4 displays a marked improvement in the tokenizer's handling of whitespace, particularly in Python code. This efficiency is achieved by grouping multiple spaces into fewer tokens, which densifies the code representation and allows the transformer to attend to a more significant portion of the code when predicting the next token. This improvement is not just a function of the language model's architecture but also a testament to the thoughtful design of the tokenizer.

![GPT-4 Tokenizer Improvements](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/750_QDgKAptMi.jpeg)

Here's what you need to know about the advancements in GPT-4's tokenizer:  
- It reduces the token count for the same string, allowing denser input to the transformer.  
- The tokenizer's vocabulary size approximately doubled from GPT-2 to GPT-4.  
- The embedding and softmax layers in the transformer grow with the number of tokens, highlighting the need for a balanced approach to vocabulary size.  
- Python code sees a significant improvement in tokenization, with whitespace being more efficiently grouped.

#### Unicode and the Nature of Text {#unicode-nature-of-text}

Understanding the nature of text that we feed into transformers is another piece of the puzzle. According to Python's documentation, strings are immutable sequences of Unicode code points. Unicode code points are essentially unique numbers assigned to each character, regardless of the platform, program, or language. The Unicode Consortium defines these characters, with the latest standard encompassing over 150,000 characters across various scripts.

Here's a brief overview of Unicode and its implications:  
- Unicode standardizes a vast array of characters and scripts.  
- The standard is continuously evolving, with the latest version being 15.1 as of September 2023.  
- Unicode allows for consistent representation and transformation of text across different encodings, crucial for global text processing.

![Unicode Character Examples](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/994_wFlQgU32H.jpeg)

#### The Intricacies of Unicode {#intricacies-unicode}

Unicode's goal of a unified character set presents its own set of challenges, especially when it comes to italic or cursive characters in scripts like Cyrillic or the round-trip format conversion for legacy character sets like Korean Hangul. The standard has to accommodate various encoding forms, giving more than one method for representing some text. This complexity extends to mappings between legacy Japanese encodings and Unicode, which have led to mismatches in round-trip format conversions.

Here are a few notable points about Unicode's complexity:  
- It accommodates combining diacritics and precomposed characters.  
- There are multiple encoding forms for certain scripts, such as Korean Hangul.  
- Inconsistent mappings between earlier Japanese encodings and Unicode can lead to conversion issues.

#### The Task Ahead: Tokenizer Implementation {#tokenizer-implementation-task}

With the understanding that strings are sequences of Unicode code points, and the complexity of supporting multiple languages and special characters, we face the task of building a tokenizer that can accurately convert these strings into tokens for language models to process.

Remember, the goal is to:  
- Tokenize strings into integers from a fixed vocabulary.  
- Use these integers to look up vectors in a table and feed them into the transformer as inputs.  
- Support a wide array of languages and special characters found on the internet.

The task at hand involves intricate decisions about how to group characters into tokens to achieve efficient and effective language processing. Let's proceed to write some code to tackle this challenge.  
### Exploring Unicode and Character Encodings {#unicode-and-character-encodings}

In the realm of text processing and language modeling, understanding character encoding is vital. Let's demystify how we can access the Unicode code point for a given character in Python.

#### Accessing Unicode Code Points in Python {#accessing-unicode-code-points}

Python provides a straightforward way to access the Unicode code point of a character using the built-in `ord()` function.

```python  
# Example of getting the Unicode code point for 'H'  
code_point_H = ord('H')  
print(code_point_H)  # Output: 104

# Example with an emoji  
heart_emoji_code_point = ord('❤️')  
print(heart_emoji_code_point)  # Output: 128148

# Example with a unique character 'ñ'  
unique_char_code_point = ord('ñ')  
print(unique_char_code_point)  # Output: 241  
```

The `ord()` function, however, only accepts a single Unicode character and returns its corresponding integer code point. This means we cannot pass entire strings into it, as they may contain multiple code points.

To extract the Unicode code points for each character in a string, we can use a list comprehension:

```python  
# Extracting code points for each character in a string  
string = "Hello, ❤️ World!"  
code_points = [ord(char) for char in string]  
print(code_points)  
# Output: [72, 101, 108, 108, 111, 44, 32, 10084, 65039, 32, 87, 111, 114, 108, 100, 33]  
```

#### The Challenge with Direct Unicode Code Points {#challenge-unicode-code-points}

Using Unicode code points directly for tokenization might seem tempting, given that they are already integers. However, there are significant challenges:

- The Unicode set is vast, with over 150,000 code points, leading to a lengthy vocabulary.  
- The Unicode standard is dynamic and undergoes regular updates, which affects stability.

![Unicode Code Points Example](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1078_WBxLn5ZSd.jpeg)

Due to these reasons, a direct use of Unicode code points for tokenization is not practical. We need a more robust solution that can handle the dynamic nature of text and its encoding.

#### The Quest for a Better Encoding {#quest-better-encoding}

To address the limitations of using raw Unicode code points, we turn to the concept of encoding. Encoding is the process of converting text data into a different format or representation, such as binary data or byte streams.

##### Unicode Encodings Overview {#unicode-encodings-overview}

According to the Unicode Consortium, there are three main types of encodings:

- **UTF-8**: Encodes code points into a byte stream ranging from one to four bytes, making it the most common and versatile encoding form.  
- **UTF-16**: Uses two bytes for most characters but four bytes for characters outside the Basic Multilingual Plane (BMP).  
- **UTF-32**: Encodes every character into four bytes, providing a fixed length for each code point but at the cost of increased space consumption.

![Unicode Encodings](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1103_plm2RSOAkT.jpeg)

Among these, UTF-8 stands out due to its compatibility and efficiency. It is the predominant encoding used on the internet.

#### UTF-8: The Preferred Encoding for Text {#utf8-preferred-encoding}

UTF-8 is particularly important for our purposes because it encodes each Unicode code point into a byte stream that varies in length depending on the character.

- The first 128 code points (ASCII) require only one byte.  
- The next 1,920 code points, covering many alphabets and symbols, need two bytes.  
- Three bytes are required for the bulk of the BMP, including many common characters.  
- Four bytes encode the remaining characters, such as less common symbols and emojis.

![UTF-8 Encoding](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1123_VulxbmmLx.jpeg)

It's important to note that UTF-8's variable length encoding allows it to be backward compatible with ASCII, making it highly preferred for global text representation.

#### Unicode: The Programmer's Perspective {#unicode-programmers-perspective}

To gain a deeper understanding of Unicode, we can refer to insightful blog posts and articles written by programmers who have delved into the subject. These resources often cover the intricacies of Unicode and provide valuable context for encoding and processing text data.

![Programmer's Introduction to Unicode](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1163_fd2_k-Cnj.jpeg)

One such example is a blog post that discusses the diverse and complex nature of Unicode from a programmer's viewpoint, offering a glimpse into the character set, string handling, and file manipulation with Unicode text.

#### UTF-8 Everywhere: A Manifesto {#utf8-everywhere-manifesto}

The "UTF-8 Everywhere" manifesto is a compelling argument for the widespread adoption of UTF-8 due to its superiority and compatibility features. It explains why UTF-8 is significantly preferred over other encodings and is used more prominently across the internet.

![UTF-8 Everywhere](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1168_g0d5rk2wp.jpeg)

The manifesto underlines UTF-8's backward compatibility with ASCII and its ability to handle all Unicode code points efficiently, making it the ideal choice for a globalized internet.

#### Encoding in Practice: UTF-8 Encoding a String {#encoding-in-practice}

With a clear preference for UTF-8 established, let's see what happens when we encode a string into UTF-8:

```python  
# Encoding a string into UTF-8  
string_to_encode = "Programming ❤️"  
encoded_string = string_to_encode.encode('utf-8')  
print(encoded_string)  
# Output: b'Programming \xe2\x9d\xa4\xef\xb8\x8f'  
```

The encoded string is now represented as a byte stream, with non-ASCII characters converted into their corresponding byte sequences.

To sum up, while Unicode provides a vast and evolving code space for representing text, its direct use for tokenization is impractical. Encoding, particularly UTF-8, offers a versatile and efficient means to handle the diverse character sets required for language modeling. With this foundation, we can approach the challenge of tokenizer implementation with the objective of creating a system capable of accurately converting strings into tokens for language models.  
### Unicode and UTF-8 in Python Tokenization {#unicode-utf8-python-tokenization}

When dealing with different languages and special characters in text processing, it's essential to handle Unicode and its encodings properly. Python provides mechanisms to encode and decode strings to and from bytes, which is a fundamental step in text tokenization.

#### Encoding Strings in Python {#encoding-strings-python}

Python's string class has a `.encode()` method that allows you to convert a string to its corresponding byte representation in various encodings such as UTF-8, UTF-16, or UTF-32. Let's look at some examples:

```python  
# Encoding a string using UTF-8  
string_utf8 = '안녕하세요 👋 (hello in Korean!)'.encode('utf-8')  
print(list(string_utf8))  
# Output: [236, 149, 136, 235, 133, 149, 236, 158, 133, 236, 138, 164, 32, 240, 159, 145, 139, 32, 40, 104, 101, 108, 108, 111, 32, 105, 110, 32, 75, 111, 114, 101, 97, 110, 33, 41]

# Encoding a string using UTF-16  
string_utf16 = '안녕하세요 👋 (hello in Korean!)'.encode('utf-16')  
print(list(string_utf16))  
# Output will be a list of integers representing UTF-16 byte pairs

# Encoding a string using UTF-32  
string_utf32 = '안녕하세요 👋 (hello in Korean!)'.encode('utf-32')  
print(list(string_utf32))  
# Output will be a list of integers representing UTF-32 4-byte sequences  
```

As seen in the examples above, different encodings yield different byte streams. UTF-16 and UTF-32 often result in a byte stream with many zeros, especially for characters within the ASCII range. This leads us to the conclusion that UTF-8 is generally more efficient and is the preferred encoding for our purposes.

![UTF-8 Encoding Visualization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1377_svmXU4P4Z.jpeg)

However, using UTF-8 naively, as raw bytes, implies a limited vocabulary size of 256 tokens (since a byte can represent values from 0 to 255). This small vocabulary size would result in long sequences of bytes for any text, which is computationally inefficient for language models with finite context lengths.

#### Byte Pair Encoding (BPE): A Solution {#byte-pair-encoding}

To overcome the limitations of raw bytes, we turn to the Byte Pair Encoding (BPE) algorithm, which compresses byte sequences into a more manageable form. BPE works by iteratively replacing the most frequent pairs of bytes with a new byte that represents the pair, thus reducing the sequence's length.

```python  
# Implementation of Byte Pair Encoding in Python  
# Note: This is a simplified example for illustrative purposes.

# Initial byte sequence  
byte_sequence = [236, 149, 136, 235, 133, 149, 236, 158, 133, 236, 138, 164]

# Define a function to find the most frequent pair  
def find_most_frequent_pair(sequence):  
    pairs = {}  
    for i in range(len(sequence)-1):  
        pair = (sequence[i], sequence[i+1])  
        if pair in pairs:  
            pairs[pair] += 1  
        else:  
            pairs[pair] = 1  
    most_frequent_pair = max(pairs, key=pairs.get)  
    return most_frequent_pair

# Define a function to replace most frequent pair with a new token  
def replace_pair(sequence, pair, new_token):  
    new_sequence = []  
    skip_next = False  
    for i in range(len(sequence)-1):  
        if skip_next:  
            skip_next = False  
            continue  
        if (sequence[i], sequence[i+1]) == pair:  
            new_sequence.append(new_token)  
            skip_next = True  
        else:  
            new_sequence.append(sequence[i])  
    if not skip_next:  
        new_sequence.append(sequence[-1])  
    return new_sequence

# Running BPE algorithm  
new_token = 256  # Starting from 256 since 0-255 are already used by single bytes  
while True:  
    pair = find_most_frequent_pair(byte_sequence)  
    if not pair:  
        break  
    byte_sequence = replace_pair(byte_sequence, pair, new_token)  
    new_token += 1

print(byte_sequence)  
# Output will be a compressed byte sequence with new tokens  
```

This BPE algorithm allows us to support a larger vocabulary size, which can be tuned as a hyperparameter, while sticking to the UTF-8 encoding of strings. The BPE algorithm is effective because it doesn't require large computational overheads and remains consistent and reliable.

![BPE Process Visualization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1382_V6tFjrvXO.jpeg)

The BPE algorithm has been modified for use in large language models, combining tokens that encode single characters with those that encode entire words or even the longest compound words.

#### BPE Implementation for Tokenization {#bpe-implementation-tokenization}

Let's see how the BPE algorithm can be applied to a real-world example. We'll take the first paragraph from a blog post and attempt to encode it using BPE.

```python  
# Example text from a blog post  
text = "I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

# Placeholder code to demonstrate BPE on the example text  
# ... (similar steps as the above BPE implementation)

# Output will be a compressed representation of the text  
```

By iteratively compressing sequences and minting new tokens, we can achieve a more efficient representation of our data. This approach allows any arbitrary sequence to be encoded using the optimized vocabulary and also decoded back to strings, a necessary process for language model training.

The concept of tokenization-free, autoregressive sequence modeling is still an area of active research. It has the potential to allow models to directly process raw byte streams, but it's not yet widely adopted or proven at scale. For now, the BPE algorithm remains a cornerstone of efficient tokenization for large language models.

![BPE Application Example](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1546_conTIyS5u.jpeg)

In essence, BPE helps us to overcome the inefficiency of using UTF-8 raw bytes for tokenization by compressing sequences and creating a more manageable vocabulary size. This method strikes a balance between sequence length and vocabulary size, making it a viable solution for modern language models.  
### UTF-8 Token Streams in Practice {#utf8-token-streams}

In the previous section, we saw the theoretical underpinnings of tokenization using Unicode and the Byte Pair Encoding algorithm. Now, let's apply these concepts to a practical example to understand how we actually get tokens from text.

#### Converting Text to UTF-8 Tokens {#converting-text-utf8-tokens}

To begin, we have our text which we want to tokenize. We encode this text into UTF-8, which will give us a raw byte stream. For ease of manipulation in Python, we convert these bytes into integers and create a list.

Here is the Python code that demonstrates this conversion:

```python  
# The original paragraph  
text = "Here's an example paragraph with some UTF-8 encoded characters."

# Encode the text into UTF-8 and convert bytes to a list of integers  
tokens = list(text.encode('utf-8'))  
print('Original text length:', len(text), 'code points')  
print('UTF-8 encoded length:', len(tokens), 'bytes')  
```

The output will show the original text length in code points and its UTF-8 encoded length in bytes. For instance, the example text might have a length of 533 code points but when encoded in UTF-8, the length could be 616 bytes or tokens. This discrepancy is due to the fact that while simple ASCII characters are represented by a single byte, more complex Unicode characters can take up to four bytes.

#### Frequency of Byte Pairs {#frequency-byte-pairs}

Once we have our list of byte tokens, the next step in the BPE algorithm is to find the most frequently occurring pair of bytes.

```python  
# Python function to get stats on byte pairs  
def get_stats(tokens):  
    counts = {}  
    for i in range(len(tokens)-1):  
        pair = (tokens[i], tokens[i+1])  
        if pair in counts:  
            counts[pair] += 1  
        else:  
            counts[pair] = 1  
    return counts

# Get the frequency statistics of our tokens  
stats = get_stats(tokens)  
print(stats)  
```

This code snippet will produce a dictionary where the keys are consecutive byte pairs from the token list and the values are their respective counts. To visualize this better, one might sort this dictionary:

```python  
# Print the byte pair counts in descending order  
sorted_stats = sorted(((value, key) for key, value in stats.items()), reverse=True)  
print(sorted_stats)  
```

The sorted list of byte pair frequencies will help us determine which pairs to merge in the subsequent steps of the BPE algorithm.

#### Iterating Over Byte Pairs {#iterating-byte-pairs}

The BPE algorithm iterates over the byte pairs to eventually merge the most common pairs. This iterative process continues until a desired vocabulary size is reached or no more merges can be performed. Here's an example code snippet that demonstrates the iterative process:

```python  
# Simplified BPE iteration  
new_token_id = max(tokens) + 1  # Assuming max(tokens) is < 256

while len(stats) > 0:  
    most_frequent_pair = max(stats, key=stats.get)  
    # If you want to see the most common pair  
    print('Most common pair:', most_frequent_pair)  
      
    # Replace the most common pair in the token list with a new token  
    i = 0  
    while i < len(tokens)-1:  
        if (tokens[i], tokens[i+1]) == most_frequent_pair:  
            tokens[i] = new_token_id  
            del tokens[i+1]  
        else:  
            i += 1

    # Update stats with the new list of tokens  
    stats = get_stats(tokens)

    new_token_id += 1  
```

By the end of this process, you will have a tokenized version of your original text, with common pairs of bytes merged into single tokens.

![UTF-8 Encoded Byte Stream Visualization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/1546_conTIyS5u.jpeg)

#### Note on Tokenization Tools {#note-tokenization-tools}

As you work through tokenization, you might find it helpful to use tools such as Jupyter notebooks for a more interactive experience. These environments allow you to write, test, and refine your functions in real-time. If you're following along, consider finding a notebook dedicated to tokenization and try to implement the function yourself.

### Implementing the `get_stats` Function {#implementing-get-stats}

When implementing the `get_stats` function, we use a Pythonic approach to iterate through consecutive elements. Here's how you can implement this function to count byte pair frequencies:

```python  
# Function to get byte pair counts  
def get_stats(ids):  
    counts = {}  
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements  
        counts[pair] = counts.get(pair, 0) + 1  
    return counts

# Example usage of get_stats  
tokens = [239, 188, 181, ...]  # A sample list of byte tokens  
stats = get_stats(tokens)  
print(sorted(((v, k) for k, v in stats.items()), reverse=True))  
```

This function uses a dictionary to track the counts of each byte pair. We then print out the sorted counts to identify the most frequent pairs. This information is critical for the next step in the BPE algorithm, where we begin to merge these pairs.

#### Advancing the BPE Algorithm {#advancing-bpe-algorithm}

The BPE algorithm isn't just about finding the most common pairs; it's about merging them to create a new token, which represents the pair. This new token is then used in subsequent iterations, potentially being combined with other tokens to form even larger ones. This process reduces the number of tokens needed to represent the text and increases the efficiency of the tokenization process.

In our example, we would continue to iterate over the list of byte tokens, replacing the most frequent pairs with new tokens and updating our statistics accordingly. Each iteration simplifies the token list further, bringing us closer to an optimal set of tokens for training our language model.

```python  
# Continue iterating and merging the most common pairs  
while True:  
    most_frequent_pair = max(stats, key=stats.get)  
    if stats[most_frequent_pair] < threshold:  # A threshold to stop merging  
        break  
    tokens, stats = merge_pair(tokens, most_frequent_pair)  
```

The `merge_pair` function is not shown here but would be responsible for replacing occurrences of the most frequent pair with a new token and returning the updated list of tokens and statistics.

By diligently applying these methods, we can construct a robust tokenization system that is essential for feeding text into a language model. This process is at the heart of enabling machines to understand and generate human language in a way that is both efficient and effective.  
#### Exploring Byte Pair Frequencies {#exploring-byte-pair-frequencies}

The `get_stats` function is a cornerstone of the BPE algorithm, allowing us to identify the most frequently occurring byte pairs in our token stream. In a Jupyter Notebook environment, we can use this function to iterate through tokens and count the occurrences of each pair. The function looks as follows:

```python  
def get_stats(ids):  
    counts = {}  
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements  
        counts[pair] = counts.get(pair, 0) + 1  
    return counts  
```

Once we have the statistics, we can sort the byte pair counts in descending order to identify which pairs are the most common:

```python  
stats = get_stats(tokens)  
# Print the byte pair counts in descending order  
sorted_stats = sorted(((v, k) for k, v in stats.items()), reverse=True)  
print(sorted_stats)  
```

If we run the above code, we might observe output indicating that the pair `(101, 32)` is the most common, occurring 20 times in our token stream. This pair represents the ASCII code for 'e' followed by a space, as demonstrated by the following code snippet:

```python  
# Convert the byte pair to characters  
print(chr(101), chr(32))  # Output: ('e', ' ')  
```

This output confirms that many words in our text likely end with the letter 'e', followed by a space.

#### Iterating Over and Merging Byte Pairs {#iterating-and-merging-byte-pairs}

The next step in the BPE algorithm is to iterate over the token stream and replace the most common byte pairs with a new token. To avoid conflicts with existing byte values which range from 0 to 255, we introduce a new token with an ID of 256:

```python  
# Create a new token ID beyond the existing byte range  
new_token_id = 256

# Replace instances of the most common pair with the new token ID  
tokens = [new_token_id if (tokens[i], tokens[i+1]) == (101, 32) else token  
          for i, token in enumerate(tokens[:-1])]  
```

By iterating through the tokens and performing this substitution, we effectively reduce the granularity of our token stream, which is the essence of BPE tokenization.

#### Byte Pair Encoding in Action {#byte-pair-encoding-in-action}

Let's consider a practical example of how we might perform byte pair encoding on a more complex string of Unicode characters. Suppose we have the following string, which includes a range of different Unicode characters:

```python  
text = "This is a string with Unicode characters like 👍, 🌟, and 🚀."  
```

After encoding this text to UTF-8 and converting it to a stream of tokens, we might have something like this:

```python  
tokens = [84, 104, 105, 115, 32, ..., 240, 159, 154, 128, 46]  
```

The `get_stats` function can then be used to find the most common byte pairs. As we iterate through and replace these pairs with new tokens, we may see the following changes to the token stream:

```python  
# After several iterations of BPE  
tokens = [256, 257, 258, 259, 260, ..., 261, 262]  
```

Each new token ID (from 256 onwards) represents a byte pair that has been merged. This process continues until no more merges can be performed or we reach a predetermined vocabulary size.

### Deep Dive into Python Code for Byte Pair Encoding {#python-code-byte-pair-encoding}

We can dive deeper into the Python code that facilitates the BPE process. The `get_stats` function is utilized within a loop to repeatedly identify and merge the most frequent byte pairs:

```python  
def get_stats(token_list):  
    counts = {}  
    for pair in zip(token_list, token_list[1:]):  # Iterate over consecutive elements  
        counts[pair] = counts.get(pair, 0) + 1  
    return counts

# This loop will continue to merge byte pairs  
while True:  
    stats = get_stats(tokens)  
    if not stats:  
        break  # Exit the loop if there are no more pairs to process  
    most_common_pair = max(stats, key=stats.get)  
    if stats[most_common_pair] < threshold:  
        break  # Exit if the frequency is below a certain threshold  
    tokens = merge_pair(tokens, most_common_pair)  
```

In the above code, `merge_pair` is a hypothetical function that would handle the replacement of the most common pair with a new token ID and return the updated list of tokens. The `threshold` variable is a configurable parameter that determines when to stop the merging process.

By executing this code, we incrementally build a tokenization scheme that can compress the original text into a more manageable sequence of tokens, which is critical for the efficiency of language models.

#### Monitoring Progress with Jupyter Notebooks {#monitoring-progress-jupyter-notebooks}

The Jupyter Notebook environment is particularly useful for monitoring the progress of BPE tokenization. By outputting the sorted byte pair counts at each iteration, we can observe how the token stream changes over time:

```python  
# Output the sorted byte pair counts after each iteration  
print(sorted(((v, k) for k, v in stats.items()), reverse=True))  
```

This interactive approach allows for a clearer understanding of the tokenization process and provides immediate feedback on the effects of each merge operation.

By adhering to these methods, we are equipped to construct a sophisticated tokenization system that underpins the functionality of modern language models. This detailed process is critical to enabling computers to process and generate human language with high accuracy and efficiency.  
#### Implementing the BPE Merge Function {#implementing-bpe-merge-function}

Continuing our exploration of the Byte Pair Encoding (BPE) algorithm, we delve deeper into the practical aspects of tokenization. After identifying the most frequent byte pairs using the `get_stats` function, the next critical step is merging these pairs. This is where the actual tokenization magic happens, and we'll see how that unfolds with Python code.

Let's revisit the `get_stats` function for context:

```python  
def get_stats(ids):  
    counts = {}  
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements  
        counts[pair] = counts.get(pair, 0) + 1  
    return counts  
```

With our statistics at hand, we can now address the process of merging. We're looking for occurrences of the byte pair `(101, 32)` and will replace it with a new token, `256`. Here's how we might go about implementing this merging logic:

```python  
# Define the merge function  
def merge(ids, pair, idx):  
    newids = []  
    i = 0  
    while i < len(ids):  
        # Check if we're not at the last position and if the pair matches  
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:  
            newids.append(idx)  
            i += 2  # Increment by 2 to skip the matched pair  
        else:  
            newids.append(ids[i])  
            i += 1  
    return newids

# Example usage of the merge function  
# Replacing occurrences of (6, 7) with 99 in the list [5, 6, 6, 7, 9, 1]  
print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))  
# Output: [5, 6, 99, 9, 1]  
```

The `merge` function is designed to iterate over the list of tokens (`ids`) and replace each occurrence of the specified pair with the new token index (`idx`). It's crucial to handle the list's bounds carefully to avoid out-of-bounds errors when checking for the presence of the target pair.

Now, let's apply this function to our actual token stream:

```python  
# Identifying the top pair to merge  
top_pair = max(stats, key=stats.get)  
# Merging the most common pair in our tokens list  
tokens2 = merge(tokens, top_pair, 256)  
```

After running the merge operation, our token stream is transformed. For example, given a list `[5, 6, 6, 7, 9, 1]` where we want to replace occurrences of `6, 7` with `99`, the merge function will produce the desired list `[5, 6, 99, 9, 1]`.

Here's a snippet of the token stream before and after the merge operation:

```python  
# Before merging  
[239, 188, 181, ..., 32, 84, 104, 101, 32, 118, 101, 114, 121, ...]

# After merging the pair (101, 32) with 256  
[239, 188, 181, ..., 32, 84, 104, 256, 118, 101, 114, 121, ...]  
```

The sequence now includes `256` wherever the pair `(101, 32)` used to be. This example illustrates the essence of the BPE process—reducing the complexity of the input by merging frequent pairs.

As we continue our BPE tokenization, we'll repeat this merging process iteratively, each time substituting the most common pair with a new token index, thus gradually building a more compressed representation of our text. This iterative process not only simplifies the text representation but also allows for more efficient processing by language models.

In a Jupyter Notebook environment, these code snippets and their outputs can be particularly illustrative, providing a dynamic way to monitor the progress of our tokenization process. The immediate feedback is invaluable for understanding the inner workings of BPE and its impact on the input text.

By now, we've covered a significant portion of the BPE algorithm, from identifying byte pair frequencies to merging them and transforming the token stream. This forms the foundation upon which modern language models process and understand text data. As we continue to iterate and refine our tokenization technique, we enable these models to tackle the complexities of natural language with greater precision and nuance.  
#### Understanding the Impact of Merging in BPE {#understanding-impact-of-merging}

As we continue our deep dive into the Byte Pair Encoding (BPE) algorithm, we witness the transformations that occur during the tokenization process. Recall that we started with a token stream length of 616. After applying the BPE merging operation, we observed a reduction in the length of the token array to 596, indicating that the merge operation reduced the sequence by 20 tokens. This aligns with the 20 occurrences of the byte pair we targeted for merging.

To illustrate the changes with a concrete example, consider the following token sequence before and after merging:

```plaintext  
# Before merging  
length: 616  
[239, 188, 101, ..., 32, 84, 104, 101, 32, ...]

# After merging the pair (101, 32) with 256  
length: 596  
[239, 188, 181, ..., 256, 84, 104, 256, ...]  
```

Here, the pair `(101, 32)` has been replaced by the new token `256`, simplifying the sequence and illustrating the essence of BPE.

#### Verifying the Merge Operation {#verifying-the-merge-operation}

To ensure the merge function's efficacy, it is important to verify that the intended byte pairs are merged correctly and that no unintended pairs are present. For instance, we might search for occurrences of the pair `(5, 6)` and verify the absence of a specific byte pair, such as `(1, 132)`, which should not occur in the processed array. This step is crucial for maintaining the integrity of the token stream and ensuring that the merge operation is functioning as expected.

#### Iterative Merging in BPE {#iterative-merging-in-bpe}

The BPE algorithm is not a one-off operation; rather, it is an iterative process. After successfully merging a single pair, the algorithm proceeds to re-scan the sequence to identify the next most common pair and replace it. This process is repeated multiple times, each iteration further compressing the representation of the text. The number of iterations is a hyperparameter that can be adjusted based on the desired level of tokenization granularity.

```python  
# Example of a while loop to perform iterative merging  
while not convergence_criteria_met:  
    stats = get_stats(tokens)  
    if not stats:  
        break  # No more mergeable pairs, exit the loop  
    top_pair = max(stats, key=stats.get)  
    tokens = merge(tokens, top_pair, new_idx)  
    new_idx += 1  # Increment the new token index for the next merge  
```

The pseudocode above demonstrates how a while loop can be employed to apply the BPE merge function iteratively until a specified convergence criterion is met. The `convergence_criteria_met` is a condition that determines when the iterative process should terminate, which could be based on the maximum vocabulary size, a minimum frequency threshold, or a fixed number of merge operations.

#### The Effect of Tokenization on Text Analysis {#effect-tokenization-text-analysis}

Tokenization serves as a critical step in text analysis and natural language processing pipelines. The specific method of tokenization can have a direct impact on the features used in machine learning models and, consequently, on the models' performance. Different tokenization techniques may yield vastly different token sequences even from the same text, influencing the outcome of the downstream tasks.

Here are some key insights about tokenization methods:

- Simple methods might just split text based on spaces and punctuation.  
- More advanced methods could take linguistic structures into account.  
- The choice of tokenization can affect the type of features used in the model.  
- It can also influence the choice of machine learning algorithms later in the project.  
- Different languages and frameworks may require distinct tokenization approaches.

Choosing the right tokenization method is thus a pivotal decision that should be made early in the project, as it bears on both the model's architecture and its ultimate efficacy in tasks like sentiment analysis, translation, or information retrieval.

#### Continual Learning and Tokenization {#continual-learning-tokenization}

As we have seen, tokenization is a nuanced and iterative process that underpins the performance of language models. It is a vital part of the NLP pipeline, laying the foundation for effective feature extraction and subsequent learning. The BPE algorithm, with its iterative merging of byte pairs, showcases how we can systematically reduce complexity while preserving the integrity of the original text.

For NLP practitioners, understanding and implementing tokenization is an iterative learning curve in itself. Each step, from initial token stream creation to iterative pair merging, offers insights into how language models perceive and process text. The iterative nature of BPE also allows for fine-tuning the granularity of tokenization, which can be tailored to the specific needs of the task at hand.

In summary, Byte Pair Encoding is not just a tokenization technique; it is an example of how attention to detail in preprocessing can lead to more nuanced and powerful language models. As we continue to iterate on these processes, our understanding of language modeling deepens, leading to ever more capable and sophisticated NLP systems.  
#### Finding the Sweet Spot in Tokenization {#finding-sweet-spot}

Tokenization is not just about breaking down text into smaller pieces; it's about finding the optimal balance between vocabulary size and sequence length. The "sweet spot" for tokenization is that point at which the language model performs best, which often requires fine-tuning the number of merges in the BPE algorithm.

```plaintext  
# Example of a token sequence after BPE merges  
[239, 189, 181, ..., 256, 84, 104, 256, ...]  
```

In the sequence above, we can see how the BPE algorithm has replaced specific byte pairs with new tokens, effectively reducing the overall length of the sequence and increasing the vocabulary size.

#### Tuning the Tokenization Hyperparameter {#tuning-tokenization-hyperparameter}

The size of the vocabulary is a hyperparameter that significantly impacts the performance of the tokenizer. Large language models like GPT have been known to use vocabularies of around 100,000 tokens. Finding the right balance requires careful tuning, as illustrated by the deliberate choice of performing exactly 20 merges to reach a vocabulary size of 276 tokens from an initial 256.

#### Leveraging Longer Texts for Better Token Statistics {#leveraging-longer-texts}

To achieve more representative token statistics, it's beneficial to use longer texts during the training phase of the tokenizer. This approach provides:

- Better understanding of token frequency and distribution  
- More accurate identification of common byte pairs  
- Enhanced performance due to more sensible results

#### The Merging Loop in BPE {#merging-loop-bpe}

The process of merging tokens in BPE is iterative and can be visualized as building a binary forest of token merges. Here's an example of how the merging loop might look in Python:

```python  
def get_stats(ids):  
    counts = {}  
    for pair in zip(ids, ids[1:]):  
        counts[pair] = counts.get(pair, 0) + 1  
    return counts

def merge(ids, pair, idx):  
    newids = []  
    i = 0  
    while i < len(ids):  
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:  
            newids.append(idx)  
            i += 2  
        else:  
            newids.append(ids[i])  
            i += 1  
    return newids

vocab_size = 276  # the desired final vocabulary size  
num_merges = vocab_size - 256  
ids = list(tokens)  # create a copy of the tokens list

merges = {}  # (int, int) -> int  
for i in range(num_merges):  
    stats = get_stats(ids)  
    pair = max(stats, key=stats.get)  
    idx = 256 + i  
    # merge the pair and update the token list  
    ids = merge(ids, pair, idx)  
    merges[pair] = idx  
```

The code above demonstrates how to create a mapping of merged byte pairs to new token indices, effectively compressing the token sequence while building up the vocabulary.

#### The BPE Binary Forest {#bpe-binary-forest}

In the BPE algorithm, merges are recorded in a dictionary that maps pairs of child tokens to a new parent token. Unlike a tree with a single root, BPE creates a forest where individual bytes (the leaves) are merged to form new parent nodes iteratively. This is a crucial aspect of how BPE builds its token hierarchy.

![BPE Binary Forest Visualization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/2107_vXJzxAr0k.jpeg)

#### Compression Achieved Through BPE Merges {#compression-bpe-merges}

By performing merges, BPE not only simplifies the token sequence but also achieves compression of the text. For example, starting with 24,000 bytes and after 20 merges, we might end up with only 19,000 tokens, which gives us a compression ratio of approximately 1.27. This ratio indicates how effectively the BPE algorithm can compress text using a limited number of merges.

#### Training the Tokenizer {#training-the-tokenizer}

Training the tokenizer is an essential stage separate from the language model itself. It involves using a distinct training set of documents to perform the BPE algorithm, establishing a vocabulary and set of merges. The tokenizer acts as a translation layer, converting raw text into token sequences and vice versa.

#### Encoding and Decoding with the Trained Tokenizer {#encoding-decoding-trained-tokenizer}

The final steps in the tokenization process involve encoding raw text into tokens and decoding token sequences back into text. The trained tokenizer is equipped to handle both tasks, allowing for seamless translation between different representations of text.

```python  
# Encoding text to tokens and decoding tokens back to text  
encoded_text = tokenizer.encode(raw_text)  
decoded_text = tokenizer.decode(encoded_text)  
```

Throughout this journey, we have explored the intricacies of tokenization, from the basic understanding of the process to the complexities of the BPE algorithm and its implementation. We have seen how careful tuning of the vocabulary size and the use of longer texts can enhance token statistics, leading to more effective language models. As the field of NLP continues to evolve, the importance of tokenization and its impact on language model performance remains a critical area of research and development.  
#### The Separation of Tokenization from Model Training {#separation-tokenization-model-training}

A crucial aspect of preparing data for Large Language Models (LLMs) is the distinction between the tokenization process and the actual model training. Tokenization serves as a foundational step where raw text is converted into a sequence of tokens, which is the only form of data the LLM encounters during training. This process is typically done once and is considered a massive pre-processing step that translates all training data into a token sequence. After this step, the original raw text is no longer needed and is discarded.

The tokenizer itself is trained separately from the LLM, often using its own distinct dataset. This is because the performance of the LLM can be affected by the variety of data used to train the tokenizer. For instance, if the tokenizer's training set includes a diverse mix of languages and coding languages, this will influence the merges that occur and consequently the density of certain types of data in the token space.

Here's an example of how you might convert raw text to tokens and vice versa, assuming we have already trained the tokenizer:

```python  
# Assuming tokenizer is an instance of a trained tokenizer  
encoded_text = tokenizer.encode("Hello, world!")  
decoded_text = tokenizer.decode(encoded_text)

print(encoded_text)  # Outputs the list of tokens  
print(decoded_text)  # Outputs "Hello, world!"  
```

#### Encoding and Decoding: Bridging Tokens and Text {#encoding-decoding-tokens-text}

Once the tokenizer is trained, we need to understand how to translate between the two realms of tokens and raw text. Encoding and decoding are the processes that allow this translation.

### Decoding: From Token Sequences to Text {#decoding-token-text}

Decoding is the process of converting a sequence of token integers back into human-readable text. This is integral to understanding the output of a language model, as the model's predictions come in the form of tokens. The following Python function outlines a simple approach to decoding:

```python  
# The vocabulary dictionary maps token IDs to their respective bytes objects  
vocab = {idx: bytes([idx]) for idx in range(256)}

# Incorporate merges into the vocabulary  
for (p0, p1), idx in merges.items():  
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):  
    # Convert a list of token integers back into text  
    tokens = b''.join(vocab[idx] for idx in ids)  
    text = tokens.decode('utf-8')  
    return text  
```

In the code above, we start with a basic mapping for byte tokens and then apply the merges to create a full vocabulary of tokens. We then use this vocabulary to decode a list of token integers into the original text.

Care must be taken with the decoding process. An incorrect sequence of tokens can lead to errors, particularly when dealing with unicode encoding like UTF-8.

### Handling UTF-8 Decoding Errors {#handling-utf8-decoding-errors}

UTF-8 is a common encoding standard that can encode all possible characters (code points) in Unicode. However, not all sequences of bytes are valid in UTF-8, and attempting to decode an invalid byte sequence will result in an error. For example, if we try to decode the byte `0x80` directly, Python will raise a `UnicodeDecodeError` because it is an invalid start byte in UTF-8.

The UTF-8 encoding system requires that certain byte patterns are followed:

- The first 128 code points (0-127) are represented with a single byte.  
- Code points 128 to 2047 require two bytes.  
- Code points 2048 to 65535 use three bytes.  
- Higher code points up to 1114111 use four bytes.

Bytes in multi-byte sequences beyond the first must start with the bits `10`, which is why a standalone `0x80` byte is invalid when decoded as UTF-8.

Here is an example demonstrating this error and its resolution:

```python  
def decode(ids):  
    # Convert a list of token integers back into text  
    try:  
        tokens = b''.join(vocab[idx] for idx in ids)  
        text = tokens.decode('utf-8')  
    except UnicodeDecodeError as e:  
        print(f"Decoding error: {e}")  
        text = None  
    return text

# Correctly decode a token sequence  
print(decode([97]))  # Outputs 'a' as 97 corresponds to 'a' in ASCII

# Attempt to decode an invalid token sequence  
print(decode([128]))  # Raises UnicodeDecodeError and outputs the error message  
```

In the example above, the function `decode` attempts to convert a list of token integers back into text and catches any `UnicodeDecodeError` that arises, printing an error message instead of raising the exception. This allows the user to understand why the decoding failed without interrupting the program's execution.

Understanding encoding and decoding is paramount when working with LLMs, as these processes bridge the gap between the raw textual data and the token sequences that the models understand and generate. As the technology advances, the ways in which we preprocess and convert data for LLMs will continue to shape the effectiveness and versatility of these powerful tools.  
### UTF-8 Encoding and Potential Pitfalls {#utf8-encoding-pitfalls}

UTF-8 is a widely used encoding format for representing text in computers, which can encode all possible characters as a sequence of bytes. UTF-8 has a specific schema that must be followed, particularly when dealing with multi-byte sequences for representing Unicode characters. Understanding this schema is crucial because an invalid byte sequence can lead to decoding errors, which is a common issue when dealing with LLMs.

#### Invalid Byte Sequences in UTF-8 {#invalid-byte-sequences-utf8}

Not every byte sequence constitutes valid UTF-8. If a language model predicts tokens that don't align with UTF-8 standards, we may encounter a `UnicodeDecodeError` when trying to convert these tokens back into a string. Here's a simple explanation of why certain bytes are invalid:

- **Invalid Start Byte**: The byte `0x80` has a binary representation of `1000 0000`. In UTF-8, a byte that starts with `10` must be a continuation byte, not a start byte. Therefore, `0x80` is an invalid start byte on its own.  
    
#### Handling Decoding Errors {#handling-decoding-errors}

By default, the Python `bytes.decode` method uses a strict error handling strategy, which means it will raise a `UnicodeDecodeError` if it encounters invalid UTF-8 byte sequences. However, there are other strategies we can employ:

- `ignore`: Skips invalid bytes.  
- `replace`: Replaces invalid bytes with a replacement character, typically "�".

Using `replace` can help us avoid decoding errors when working with LLMs. OpenAI, for example, uses this strategy in their code releases.

### Implementing Error Handling in Token Decoding {#implementing-error-handling}

When we implement the decoding function, we want to handle invalid byte sequences gracefully. Let's modify our previous `decode` function to use the `replace` error strategy:

```python  
def decode(ids):  
    # Given ids (list of integers), return Python string  
    tokens = b''.join(vocab[idx] for idx in ids)  
    text = tokens.decode('utf-8', errors='replace')  
    return text

print(decode([128]))  # Outputs a string with a replacement character for the invalid byte  
```

In the example above, if the token list contains `128`, which represents an invalid start byte in UTF-8, the decoding function will replace it with the "�" character, thus preventing a `UnicodeDecodeError`.

#### Python's Built-in Methods for Handling Bytes {#python-built-in-methods}

Python provides several built-in methods for dealing with byte sequences. These methods are part of the `bytes` and `bytearray` types and allow for various operations, such as:

- `bytes.decode(encoding='utf-8', errors='strict')`: Converts byte sequences to strings, handling errors according to the specified strategy.  
- `bytes.removesuffix(suffix, /)`: Removes a specified suffix from the byte sequence if present.

### Encoding: From Text to Token Sequences {#encoding-text-tokens}

While decoding translates token sequences into text, encoding performs the opposite: converting text to token sequences. To implement an encoding function, we need to start by converting the text into raw bytes using UTF-8 encoding and then use our merge dictionary to combine certain byte sequences according to the trained tokenizer's rules.

#### Merging Byte Sequences During Encoding {#merging-byte-sequences}

The merges dictionary contains rules for which byte pairs should be combined into a single token. For example:

```python  
merges = {  
    (101, 32): 256,  # e + space  
    (105, 110): 257, # i + n  
    # ... additional merge rules ...  
}  
```

Using this information, we can write an `encode` function that takes a string and outputs a list of token IDs:

```python  
def encode(text):  
    # Given a string, return list of integers (the tokens)  
    tokens = list(text.encode('utf-8'))  
    # Apply merges to combine tokens where applicable  
    # ... implementation of merge rules ...  
    return tokens

# Example usage:  
print(encode("Hello, world!"))  # Outputs a list of token IDs  
```

In the function above, `text.encode('utf-8')` converts the text into its corresponding UTF-8 byte sequence. Then, the function applies the merge rules to combine specific byte pairs into single tokens, outputting a list of token IDs.

### Practical Considerations with LLMs and Tokenization {#practical-considerations-llms-tokenization}

As we delve deeper into the intricacies of tokenization, it's important to remember the practical implications for working with LLMs:

- **Error Handling**: Be prepared to handle invalid byte sequences when decoding token outputs from an LLM.  
- **Merge Rules**: Understand the tokenizer's merge rules to properly encode text into tokens and decode tokens into text.  
- **Data Preparation**: Thorough preprocessing and conversion of data are key to maximizing the effectiveness of LLMs.

![UTF-8 Encoding Example](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/2794_8N_l7_8ChR.jpeg)

In the example above, we demonstrate how a byte sequence (in this case, `128`) can be replaced with a special marker to avoid decoding errors. This approach ensures that even if an LLM predicts an invalid sequence of tokens, we can still decode it into a human-readable string, albeit with markers indicating where the errors occurred.

Through this detailed exploration of tokenization, we've laid the groundwork for a deeper understanding of how LLMs interact with text data. The tokenization process is not merely a technical detail; it is a foundational aspect of how LLMs learn and generate language. By mastering these concepts, researchers and practitioners can better harness the capabilities of LLMs and push the boundaries of what these models can achieve.  
### Understanding the Inner Workings of Merges in Tokenization {#understanding-merges}

When we think about tokenization in Large Language Models, it's crucial to consider the mechanics behind the merging of byte pairs. This process is fundamental to how tokenizers like BPE operate, and it directly influences the model's ability to understand and generate text.

#### Constructing the Merge Dictionary {#constructing-merge-dictionary}

The merge dictionary is a core component of the BPE algorithm. It dictates which byte pairs should be combined into a single token. This dictionary is built from top to bottom, with the order of insertion being significant. Here's how a typical merge dictionary looks:

```python  
merges = {  
    (101, 32): 256,  # e + space  
    (105, 110): 257, # i + n  
    (115, 32): 258,  # s + space  
    (116, 104): 259, # t + h  
    (101, 114): 260, # e + r  
    # ... additional merge rules ...  
}  
```

In the above code, the tuple `(101, 32)` represents the byte pair for `e` and a space, and the corresponding token index for this pair is `256`. These rules are applied during the tokenization process to merge individual bytes into tokens.

#### The Encoding Function {#encoding-function}

The encoding function converts a string into a list of token IDs, applying the merge rules from the dictionary. Let's take a closer look at a simplified version of this function:

```python  
def encode(text):  
    # Given a string, return list of integers (the tokens)  
    tokens = list(text.encode('utf-8'))  
    # Apply merges to combine tokens where applicable  
    # ... implementation of merge rules ...  
    return tokens  
```

In this function, `text.encode('utf-8')` turns the input string into a series of bytes. Then, based on the merges dictionary, it applies the rules to combine certain byte pairs into single tokens.

#### Order of Merges and Loop Implementation {#order-of-merges}

The order of merges is important; we prefer to perform all the early merges before moving on to the later ones. This is because some merges depend on the results of earlier ones. For instance, if a merge relies on the token `256`, which was created earlier, we must respect the sequence of operations.

Here's a pseudo-code representation of how we might implement this loop:

```pseudo  
while true:  
    # Identify the pair to merge at this stage  
    # If no more pairs can be merged, break out of the loop  
    # Perform the merge operation  
```

#### Identifying Merge Candidates {#identifying-merge-candidates}

To perform a merge, we need to identify eligible pairs of bytes that can be combined. We use a function, say `get_stats`, to count how often each byte pair occurs in our sequence of tokens. Here's how we might use this information:

```python  
# Assume get_stats returns a dictionary of byte pairs and their frequencies  
stats = get_stats(tokens)

# We are only interested in the keys (the byte pairs themselves)  
merge_candidates = stats.keys()

# Find the merge candidate with the lowest index in the merges dictionary  
```

#### Selecting the Pair with the Minimum Index {#minimum-index-pair}

We want to select the pair with the lowest index in the merges dictionary, as this indicates the next merge to perform. Here's an approach using Python's `min` function:

```python  
pair_to_merge = min(merge_candidates, key=lambda pair: merges.get(pair, float('inf')))  
```

In this code, `merges.get(pair, float('inf'))` retrieves the index of the pair in the merges dictionary or returns infinity if the pair is not in the dictionary. This ensures that pairs not eligible for merging are not considered.

#### Handling Special Cases {#handling-special-cases}

We must account for special cases, such as when the input string consists of a single character or is empty. In such cases, `stats` would be empty, and the merge operation would fail. We need to handle this within our encoding function to prevent errors.

#### Putting It All Together: The Complete Merge Process {#complete-merge-process}

Now let's put all these pieces together to understand the complete merge process within the encoding function:

```python  
def encode(text):  
    # Given a string, return list of integers (the tokens)  
    tokens = list(text.encode('utf-8'))  
    while True:  
        # Identify eligible merge candidates  
        merge_candidates = get_stats(tokens).keys()  
          
        # Find the pair to merge  
        pair_to_merge = min(merge_candidates, key=lambda pair: merges.get(pair, float('inf')))  
          
        # If no pair can be merged, break out of the loop  
        if merges.get(pair_to_merge) is None:  
            break  
          
        # Perform the merge operation  
        tokens = merge_pair(tokens, pair_to_merge)  
      
    return tokens  
```

In this function, `merge_pair` is a hypothetical function that would replace every occurrence of `pair_to_merge` in the `tokens` list with the corresponding index from the `merges` dictionary.

### The Importance of Merges in Tokenization {#importance-of-merges}

The process of merging byte pairs is more than just a technical detail; it is integral to the efficiency and accuracy of tokenization in LLMs. By understanding and correctly implementing this process, we can ensure that our language models are able to understand and generate text based on the nuanced rules of human language. 

This exploration of the encoding function and merge dictionary highlights the complexity and precision required in the tokenization process. Each step, from constructing the merge dictionary to handling special cases in the encoding function, plays a vital role in how effectively an LLM can work with text data. As researchers and developers, mastering these steps allows us to better utilize LLMs and contribute to advancements in the field of natural language processing.  
#### Edge Cases and Test Scenarios for the Encoding Function {#edge-cases-encoding-function}

When developing a robust encoding function for tokenization, it's essential to consider and handle edge cases. One such case occurs when the input string is very short, consisting of a single character or even being empty. In such scenarios, the list of tokens would have a length of less than two, and as a result, there would be no byte pairs to merge. To address this, we can implement a simple check:

```python  
def encode(text):  
    tokens = list(text.encode('utf-8'))  
    if len(tokens) < 2:  
        return tokens  
    # ... rest of the implementation ...  
```

By returning early in the case of short input strings, we prevent unnecessary processing and potential errors during the merging phase.

#### Testing the Tokenizer {#testing-tokenizer}

Testing is an essential part of developing any algorithm, and the tokenizer is no exception. Here we have a few test cases to validate our implementation:

1. If we encode a string and then decode it, we should expect to get the original string back.  
2. The tokenizer should handle strings that it has not seen during training.

Here's how we can approach testing:

```python  
# Example of the encode function, continued from previous code  
def encode(text):  
    # ... previous implementation details ...  
    # Begin merging process  
    while len(tokens) > 1:  
        stats = get_stats(tokens)  
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))  
        # ... rest of the implementation ...

# Test case 1: Encoding and then decoding should return the original string  
original_text = "The training text we used to train the tokenizer"  
encoded_tokens = encode(original_text)  
decoded_text = decode(encoded_tokens)  
assert original_text == decoded_text

# Test case 2: Test with unseen text  
unseen_text = "Text that the tokenizer has not seen"  
assert isinstance(encode(unseen_text), list)  
```

These tests give us confidence that the tokenizer is implemented correctly and can handle both seen and unseen text.

![Tokenization Test Output](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/3405_7ZGmpjB8B.jpeg)

#### The Tokenizer's Parameters {#tokenizer-parameters}

The essence of the Byte Pair Encoding (BPE) algorithm lies in its merges dictionary, which is effectively a set of instructions for the tokenizer on how to combine raw bytes into meaningful tokens. The merges dictionary, once trained on a dataset, enables both encoding and decoding between raw text and sequences of tokens. This is illustrated in the following snippet:

```python  
# The merges dictionary extracted from training  
merges = {  
    # ... previously defined pairs ...  
    (257, 103): 270,  
    (261, 100): 271,  
    # ... additional merge rules ...  
}

# This dictionary creates a binary forest on top of raw bytes  
# which can be used to encode and decode text  
```

#### Tokenization and Compression Ratio {#tokenization-compression}

An important metric to consider when evaluating a tokenizer is the compression ratio it achieves. The tokenizer is a distinct module from the Large Language Model (LLM) and has its own training dataset. It uses the BPE algorithm to train the vocabulary and then translates text into token sequences, which the LLM will utilize. This separation is key because the LLM never directly handles raw text, only token sequences.

The compression ratio reflects the tokenizer's efficiency in reducing the size of the input text. Here's an example of how we might calculate and present this ratio:

```python  
# Example of calculating the compression ratio  
compression_ratio = calculate_compression_ratio(text, encoded_tokens)  
print(f"Compression ratio: {compression_ratio}%")  
```

This ratio provides insights into how well the tokenizer optimizes the input data for the LLM to process.

#### State-of-the-Art Language Models and Their Tokenizers {#state-of-art-tokenizers}

Moving forward, we delve into some state-of-the-art language models to understand the variety of tokenizers in use today. The GPT series, particularly the GPT-2 paper from 2019, is an excellent starting point. This paper discusses the tokenizer used for GPT-2 and motivates the use of byte-level BPE for UTF-8 encoded text.

The GPT-2 architecture and the nuances of its input representation are outlined in the paper's excerpts. The tokenizer's ability to handle any Unicode string without lossy pre-processing makes it a powerful tool for language modeling. The approach combines the empirical benefits of word-level models with the generality of byte-level approaches, resulting in a tokenizer that can process any dataset.

![GPT-2 Paper Input Representation](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/3470_ftwXSYmmu.jpeg)

![GPT-2 Paper BPE Details](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/3475_V6-flz9uW9.jpeg)

For those interested in a deeper dive, the paper's section on input representation is quite readable:

_Byte Pair Encoding (BPE) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences._

The paper also touches on the modifications made to the standard BPE to optimize it for GPT-2's requirements, emphasizing the importance of not merging across character categories and the exception made for spaces to improve compression efficiency.

This comprehensive look at the tokenizer's role in LLMs underscores its significance and the complexities involved in making it work efficiently and accurately. As we continue to explore different tokenization strategies and their impact on language modeling, it becomes clear that tokenization is not merely a preprocessing step but a critical component that shapes the performance and capabilities of state-of-the-art language models.  
### Advanced Tokenization Strategies in GPT-2 {#advanced-tokenization-strategies}

When we talk about tokenization in the context of language models like GPT-2, it's not just a straightforward application of the naive Byte Pair Encoding (BPE) algorithm. The GPT-2 model by OpenAI introduced some nuances to the way tokenization is handled in order to optimize the process and avoid suboptimal clustering of tokens.

#### The Challenge of Punctuation and Semantics {#challenge-punctuation-semantics}

The tokenization process for language models like GPT-2 has to deal with frequent words that appear next to various types of punctuation. For instance, the word `dog` might frequently appear alongside periods and question marks, like `dog.` or `dog?`. Naively, BPE might merge these into single tokens, leading to a proliferation of tokens that are essentially the word `dog` with different punctuation attached. This conflates semantics with punctuation, which is suboptimal. OpenAI recognized this issue and sought to address it in GPT-2's tokenizer.

The tokenizer developers introduced manual rules to prevent certain types of characters from being merged together, enforcing a more sophisticated tokenization strategy on top of BPE.

#### Enforcing Merge Rules {#enforcing-merge-rules}

The GPT-2 tokenizer incorporates manually defined merge rules to prevent unwanted combinations. Here's a snippet of the tokenizer's code from the OpenAI GPT-2 GitHub repository:

![GPT-2 Tokenizer Source Code](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/3564__zH33R5CJ.jpeg)

Importantly, the code uses the `regex` library rather than the standard Python `re` module, which provides more advanced regular expression capabilities. This allows the tokenizer to use complex regex patterns to control which parts of the text should never be merged.

#### Regex Patterns in Tokenization {#regex-patterns-tokenization}

The core of GPT-2's tokenization strategy lies in a regex pattern that identifies parts of text that should not be merged. This pattern looks complicated, but let's break it down:

```python  
from functools import lru_cache

@lru_cache()  
def bytes_to_unicode():  
    # ... Function code ...  
```

The `@lru_cache()` decorator is used to cache the results of the function, which is presumably used to convert bytes to Unicode characters. While the full function code is not shown, it's an important part of the tokenizer's functionality.

Now, the pattern itself is used in conjunction with `regex.findall` to match against the input string. The pattern includes various Unicode character properties, such as `\p{L}`, which matches any kind of letter from any language. 

```python  
pat = re.compile(r'''[^\\W\\d] # ... rest of the pattern ... ''')  
```

The use of this pattern allows the tokenizer to split the input text into segments that can be tokenized without inadvertently merging semantically distinct elements like words and punctuation.

#### GPT-2's Tokenization in Practice {#gpt2-tokenization-practice}

To see how the GPT-2 tokenizer works, we'll use the regex pattern to tokenization an example string:

```python  
import regex as re

# Example pattern from GPT-2 tokenizer  
pat = re.compile(r'''...''') # The full regex pattern goes here

# Example string  
text = "Hello world!"

# Tokenization using the regex pattern  
tokens = re.findall(pat, text)  
print(tokens)  
```

This would result in a list that separates the words and punctuation, respecting the enforced merge rules. The actual output would depend on the full regex pattern used by the GPT-2 tokenizer.

#### A Closer Look at the GPT-2's Encoder {#gpt2-encoder}

OpenAI's choice to name the tokenizer module `encoder.py` may seem a bit unconventional, as the module is responsible for both encoding and decoding. However, it's worth examining the content of this module to understand the intricacies of the GPT-2 tokenizer.

Within the `encoder.py` file, one finds the `Encoder` class with methods for BPE, encoding, and decoding, which work together to convert text to tokens and back:

```python  
class Encoder:  
    # ... other methods ...

    def bpe(self, token):  
        # ... BPE algorithm implementation ...

    def encode(self, text):  
        bpe_tokens = []  
        for token in re.findall(self.pat, text):  
            # Encoding logic  
            return bpe_tokens

    def decode(self, tokens):  
        # Decoding logic  
        return text  
```

The `encode` method uses regex to find all occurrences of the pattern in the input text, and then it applies the BPE algorithm to each found token.

Understanding the GPT-2 tokenizer's design and implementation offers valuable insights into the challenges and solutions of creating effective tokenization strategies for large language models. By examining the OpenAI GPT-2 tokenizer, we gain a richer appreciation for the complexity and thoughtfulness that underlies these foundational components of modern NLP systems.  
### Understanding Token Sequences and Compression in GPT-2 {#understanding-token-sequences}

When delving deeper into the tokenization process of GPT-2, we come across the fundamental transformation where raw text is converted into a sequence of tokens. This process involves encoding text into tokens and then verifying that the decoding process returns the original text.

Let's consider an example to understand this process:

```python  
# Define the encode function  
def encode(text):  
    # Given a string, return list of integers (the tokens)  
    tokens = list(text.encode('utf-8'))  
    return tokens

# Define the decode function  
def decode(tokens):  
    # Given a list of tokens, return the original string  
    text = bytes(tokens).decode('utf-8')  
    return text

# The original text  
text = "hello world"

# Encoding the text  
encoded_text = encode(text)  
print(encoded_text)

# Decoding back to text  
text2 = decode(encoded_text)  
print(text2 == text)  # Outputs: True

# Check the validity of the decoded text  
valtext = decode(encode(text))  
print(valtext == text)  # Outputs: True  
```

In this example, the function `encode` takes a string and returns a list of integers representing the tokens. The `decode` function does the reverse, converting the token list back into the original string. The process is validated by checking if the decoded text is equivalent to the original text.

#### Tokenization Example with GPT-2 {#tokenization-example-gpt2}

Consider the following tokenization example:

```  
hello world

In [92]: text2 = decode(encode(text))  
print(text2 == text)  
True

In [93]: valtext =  
```

The output `True` confirms that the decoding of the encoded text results in the original text, indicating that the tokenization process has not altered the meaning or structure of the text.

#### Compression Ratio of Tokens {#compression-ratio-tokens}

When analyzing the tokenization module, we also encounter metrics such as the length of the tokens and the compression ratio:

```  
tokens length: 2439  
ids length: 19438  
compression ratio: 1.27x  
```

These metrics provide insights into the efficiency of the tokenization process. The compression ratio indicates how much the original text has been condensed into tokens. In the example above, a compression ratio of `1.27x` means that the token sequence is 1.27 times smaller than the sequence of Unicode code points that represent the raw text.

#### The Role of the Tokenizer Module {#role-tokenizer-module}

It is essential to understand that the Tokenizer is a completely separate, independent module from the Large Language Model (LLM). It has its own training dataset of text (which could be different from that of the LLM), and it's responsible for translating back and forth between raw text and sequences of tokens. The LLM then only ever sees the tokens and never directly deals with any text.

The process can be visualized as follows:

```  
LLM  
  |  
token sequence  
  |  
Tokenizer  
  |  
raw text (Unicode code point sequence)  
```

The tokenizer thus acts as an intermediary that encodes and decodes between human-readable text and the token sequences that the LLM can process.

#### Decoding Given a Sequence of Integers {#decoding-given-sequence-integers}

Decoding involves translating a sequence of integers back into text. The range of integers used corresponds to the size of the vocabulary that the tokenizer was trained on. Here's a simplified view of how decoding works using a vocabulary and the BPE algorithm:

```python  
vocab = {idx: bytes([idx]) for idx in range(256)}  
for (p0, p1), idx in merges.items():  
    vocab[idx] = vocab[p0] + vocab[p1]  
```

In the above snippet, `vocab` is a dictionary where each index maps to a byte sequence, and `merges` represent pairs of indices that are merged into new tokens during the BPE training process.

#### Concatenation of Token Sequences {#concatenation-token-sequences}

After tokenizing separate elements of the text, the resulting tokens are concatenated to form a continuous sequence that the LLM can interpret. This process ensures that all parts of the text are tokenized independently and then joined to reconstruct the sequence:

```  
hello world. How are you?  
```

For instance, in the above text, "hello," "world," "How," "are," and "you?" would be tokenized separately and then concatenated to form the token sequence that represents the entire sentence.

### Enforcing Tokenization Rules Using Regex Patterns {#enforcing-tokenization-rules}

The GPT-2 tokenizer uses regex patterns to enforce rules about which text elements should not be merged during tokenization. This prevents undesired combinations of characters and helps maintain the distinction between different types of characters, such as letters, numbers, and punctuation.

The tokenizer ensures that:

- Letters and numbers are separated. For example, "hello world123" will tokenize "world" and "123" as separate entities.  
- Apostrophes and other punctuation are treated appropriately. For example, "don't" and "house's" are tokenized correctly, but special Unicode apostrophes may be tokenized differently.  
- Case sensitivity affects tokenization. Words like "House's" will tokenize differently from "house's" due to the absence of `RE.IGNORECASE`, resulting in the apostrophe being separated in the uppercase version.

The use of regex patterns is crucial in managing these nuances, as it allows the tokenizer to effectively chunk up the text and prevent certain merges from happening. These patterns are carefully crafted to handle a wide variety of text scenarios, ensuring the tokenizer's robustness and versatility.

### Real-World Application of GPT-2 Tokenization {#real-world-application-gpt2-tokenization}

The GPT-2 tokenizer's ability to handle complex text scenarios extends to real-world applications. For instance, when dealing with large volumes of text data, the tokenizer's regex patterns can effectively separate and identify different text components, such as whitespace, punctuation, and different types of characters.

The tokenizer's design to prepend spaces to tokens (e.g., " are" instead of "are") is a deliberate choice that allows for consistent tokenization across different text inputs. This approach also ensures that commonly used tokens, like " u" for "you," are readily available in the token sequence, which can be beneficial for the LLM's processing and understanding of the text.

By exploring the tokenizer's operation through examples and explanations, we gain a greater understanding of the intricate processes that underpin the GPT-2 model's tokenization strategy. This knowledge equips us with the tools to appreciate the complexity of NLP tasks and the sophistication of the solutions employed by state-of-the-art language models like GPT-2.  
### Delving into the Specifics of GPT Tokenization {#gpt-tokenization-specifics}

When it comes to the nitty-gritty of tokenization in models like GPT-2 and GPT-4, we encounter a sophisticated process that goes beyond simple character or word splitting. The tokenizer is designed to split text into chunks whenever a category change is detected, which can result in a high number of elements in the tokenized list, especially when dealing with code or other structured text.

#### Forced Splits and Regex Patterns {#forced-splits-regex}

A key aspect of the GPT tokenization process involves the use of regular expressions (regex) to enforce rules on how text should be split. Let's explore how regex patterns are applied in the GPT series:

```python  
import regex as re

# Define a regex pattern for the GPT tokenizer  
gpt2pat = re.compile(r'...')  
```

The above snippet illustrates the use of the `regex` module in Python to compile a pattern that the tokenizer will use. While the exact pattern is not shown here, we know from OpenAI's implementation that it is crafted to ensure that certain splits are forced, such as ensuring there are no merges within elements.

#### The Role of Whitespace in Tokenization {#whitespace-tokenization}

An interesting peculiarity in OpenAI's approach to tokenization is the handling of spaces. Unlike some tokenizers, OpenAI's tokenizer for GPT-2 does not merge spaces during the tokenization process. This can be observed by tokenizing a chunk of text and noticing that spaces are kept independent and are all tokenized to the same value, often represented as "20" in the encoded token list.

```python  
# Example code to illustrate the handling of spaces in GPT-2  
# ... (code not provided in full)  
```

This suggests that OpenAI enforced a rule that spaces should never be merged, indicating additional rules on top of chunking and Byte Pair Encoding (BPE) that are not explicitly documented.

#### Inference Code for Tokenization {#inference-code}

The code released by OpenAI for the GPT tokenizer is only intended for inference, not for training. This means that while it can apply merges to a new piece of text based on pre-trained rules, it cannot be used to train the tokenizer from scratch with new text.

The Encoder class within the tokenizer code demonstrates this:

```python  
class Encoder:  
    def __init__(self, encoder, bpe_merges, errors='replace'):  
        # ... initialization code ...

    def bpe(self, token):  
        # ... BPE algorithm code ...  
```

This class includes a method for applying BPE to given tokens, using a cache and a set of pre-defined merge rules (`bpe_merges`). Despite the lack of training code, this inference code gives us insight into the tokenization process.

#### Tiktoken: Official Tokenization Library from OpenAI {#tiktoken-library}

For those looking to implement GPT tokenization, OpenAI provides an official library called Tiktoken. This library, available through package installation, allows users to tokenize text using the same methods as the GPT models. Here's how you might use it:

```python  
import tiktoken

# Example use of Tiktoken library for GPT-2 tokenization  
# ... (full example code not provided)  
```

Tiktoken provides the tokenization inference for various GPT models, including GPT-2 and GPT-4. Running the library will output tokens that are specific to each model.

#### Differences Between GPT-2 and GPT-4 Tokenization {#gpt2-vs-gpt4-tokenization}

A significant update in GPT-4's tokenizer compared to GPT-2 is the change in the regex pattern used to chunk up text. This change affects how certain elements, such as numbers and case-sensitive characters, are tokenized.

For example, GPT-4's tokenizer introduces case-insensitive matching for certain contractions and abbreviations, allowing it to recognize both uppercase and lowercase versions of these text fragments. Additionally, GPT-4 limits the merging of numbers to sequences of up to three digits, preventing the creation of tokens from lengthy number sequences.

The regex pattern changes and these tokenization adjustments are part of what's behind the improved performance in GPT-4, although much of the rationale for these changes remains undocumented.

#### Tokenization Patterns and Special Tokens {#tokenization-patterns-special-tokens}

The tokenization patterns and the handling of special tokens are crucial for the tokenizer's performance. OpenAI has made adjustments to these patterns and the special tokens used, which directly influence the encoding and decoding processes.

For those interested in the finer details of these patterns, it's recommended to refer to the regex documentation and step through the patterns to understand their functionality. However, it suffices to say that major changes include case-insensitive matching and refined handling of numbers and whitespace.

#### Vocabulary Expansion {#vocabulary-expansion}

One notable change from GPT-2 to GPT-4 is the expansion of the tokenizer's vocabulary size. GPT-4's vocabulary has roughly doubled from GPT-2's 50,000 tokens to approximately 100,000 tokens, allowing for a richer and more nuanced understanding of language.

### Insights into GPT Tokenization {#insights-gpt-tokenization}

Through the examination of tokenization in GPT models, we gain valuable insights into the complexity of processing human language. The choice of regex patterns, the handling of whitespace, and the specific adjustments made in newer models like GPT-4 all contribute to the effectiveness of these language models.

While the full details of OpenAI's tokenization training process remain a mystery, the inference code and patterns provided offer a glimpse into the intricate design of these powerful NLP tools. As we continue to explore the capabilities and inner workings of LLMs, the tokenization process remains a key area for understanding and innovation.  
### Exploring the GPT-2 Encoder Implementation {#gpt2-encoder-implementation}

In the quest to understand the inner workings of tokenization in GPT-2, it's invaluable to dive into the actual code used by OpenAI. The `encoder.py` file is central to this process, and though it may seem daunting at first glance, it is quite approachable with the foundational knowledge we've built up so far.

#### The Encoder Class {#encoder-class}

At the heart of the `encoder.py` file is the `Encoder` class, which encapsulates the functionality for tokenization and detokenization (decoding). It's interesting to note that the encoder operates not just as a simple vocabulary lookup but also includes more complex operations such as byte-pair encoding (BPE).

Let's examine a snippet from the `Encoder` class that details the BPE method:

```python  
class Encoder:  
    def bpe(self, token):  
        if len(word) == 1:  
            break  
        else:  
            pairs = get_pairs(word)  
        word = '.'.join(word)  
        self.cache[token] = word  
        return word  
```

The above code highlights the BPE process, which iteratively merges the most frequent pairs of bytes or characters in the text. This is a crucial step in reducing the size of the vocabulary without losing the fine-grained detail of the language.

#### Encoding and Decoding Methods {#encoding-decoding-methods}

The `Encoder` class provides methods to encode text into tokens and decode tokens back into text. The `encode` method breaks down text into BPE tokens, while the `decode` method does the reverse:

```python  
class Encoder:  
    # ... other methods ...

    def encode(self, text):  
        bpe_tokens = []  
        for token in re.findall(self.pat, text):  
            token = '.'.join(self.byte_encoder[b] for b in token.encode('utf-8'))  
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  
        return bpe_tokens

    def decode(self, tokens):  
        text = '.'.join([self.decoder[token] for token in tokens])  
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)  
        return text  
```

These methods demonstrate how text is transformed into a sequence of integers (tokens) and back again, which is the foundation of how GPT-2 processes and generates language.

#### Initialization and Vocabulary Loading {#initialization-vocabulary-loading}

The tokenizer's vocabulary and BPE merges are loaded from files when the `Encoder` is initialized. This allows the encoder to use the pre-trained tokenization scheme of GPT-2:

```python  
class Encoder:  
    # ... other methods ...

    def get_encoder(model_name, models_dir):  
        with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:  
            encoder = json.load(f)  
        with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding='utf-8') as f:  
            bpe_data = f.read()  
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]  
        return Encoder(  
            encoder=encoder,  
            bpe_merges=bpe_merges,  
        )  
```

This piece of code reads the necessary files to construct the `Encoder` object. The `encoder.json` contains the mapping from tokens to integers, and `vocab.bpe` contains the merge rules for BPE.

#### Understanding the Byte Encoder and Decoder {#byte-encoder-decoder}

OpenAI's `encoder.py` also includes a byte encoder and decoder, which may initially seem like a superfluous detail. These elements are crucial for handling text as byte sequences, ensuring compatibility with different text encodings:

```python  
# Placeholder for the actual code from encoder.py  
# The byte_encoder and byte_decoder are dictionaries  
# that map bytes to tokens and tokens to bytes, respectively.  
```

#### The Relationship Between the Tokenizer and LLM {#tokenizer-llm-relationship}

It's important to underline that the tokenizer is a separate module from the LLM. It has its own training dataset and processes text independently of the LLM:

- **Tokenizer**: Translates raw text into sequences of tokens.  
- **LLM**: Processes sequences of tokens, never dealing with raw text directly.

The tokenizer's role is to provide the LLM with a digestible sequence of tokens that encapsulate the meaning and structure of the input text.

#### Encoding and Decoding Validation {#encoding-decoding-validation}

To validate the encoding and decoding process, one can perform a round-trip conversion and check for equality:

```python  
text2 = decode(encode(text))  
print(text2 == text)  # Should output True  
```

This simple test ensures that the tokenizer can accurately reconstruct the original text from the tokens, which is essential for the LLM to function correctly.

#### Dive Deeper into Encoder.py {#dive-deeper-encoder}

While the byte encoder and decoder might not be deeply fascinating, they are a necessary part of the tokenization process. Those interested in the minutiae can step through the relevant portions of code to gain a better understanding of their function.

Through this exploration of the `encoder.py` file, we've gained a deeper appreciation for the complexity and ingenuity behind GPT-2's tokenization process. Understanding these mechanisms is essential for anyone looking to comprehend or work with large language models like GPT-2.  
### The GPT-2 Byte-Pair Encoding Algorithm {#gpt2-byte-pair-encoding}

The core of the GPT-2 tokenization mechanism lies in the Byte-Pair Encoding (BPE) algorithm. This method, while seemingly complex, becomes more intuitive once we dissect the code and the algorithmic principles behind it.

#### The BPE Function {#bpe-function}

The `bpe` function within the `Encoder` class is where the BPE magic happens. The process iteratively merges the most frequent pairs of bytes or characters in the text. Here's a closer look at the function's implementation:

```python  
class Encoder:  
    def bpe(self, token):  
        word = tuple(token)  
        while True:  
            if len(word) == 1:  
                break  
            else:  
                pairs = get_pairs(word)  
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))  
                if bigram not in self.bpe_ranks:  
                    break  
                first, second = bigram  
                new_word = []  
                i = 0  
                while i < len(word):  
                    try:  
                        j = word.index(first, i)  
                        new_word.extend(word[i:j])  
                        i = j  
                    except ValueError:  
                        new_word.extend(word[i:])  
                        break  
                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:  
                        new_word.append(first + second)  
                        i += 2  
                    else:  
                        new_word.append(word[i])  
                        i += 1  
                word = tuple(new_word)  
        word = '.'.join(word)  
        self.cache[token] = word  
        return word  
```

In this code, `get_pairs` is a function that identifies the pair of symbols to be merged next. The pair is chosen based on its rank in `self.bpe_ranks`, which is a dictionary containing the merge order for pairs of bytes or characters.

#### Encoding and Decoding with BPE {#encoding-decoding-bpe}

The encoding and decoding methods are critical to the conversion of text to tokens and vice versa:

```python  
class Encoder:  
    # ... other methods ...

    def encode(self, text):  
        bpe_tokens = []  
        for token in re.findall(self.pat, text):  
            token = '.'.join([self.byte_encoder[b] for b in token.encode('utf-8')])  
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  
        return bpe_tokens

    def decode(self, tokens):  
        text = '.'.join([self.decoder[token] for token in tokens])  
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)  
        return text  
```

The `encode` function uses regular expressions to find all substrings in the input text that correspond to tokens. It then encodes the text into bytes and applies BPE to convert the byte sequences into tokens. The `decode` function reverses this process, turning a sequence of tokens back into human-readable text.

#### Special Tokens in GPT-2 {#special-tokens-gpt2}

One of the often overlooked yet vital components of the GPT-2 tokenizer are the special tokens. These tokens are not part of the original text but are added to convey certain signals or structures within the token stream:

- **End of Text Token**: Used to delimit documents within the training dataset, signaling the language model that the document has ended.

The encoder object from OpenAI GPT-2 contains 50,257 tokens, which include 256 raw byte tokens, 50,000 tokens from the BPE merges, and one special token for the end of the text.

#### Inserting the Special Tokens {#inserting-special-tokens}

Special tokens are inserted into the token stream to delimit different parts of the data. For instance, the end-of-text token is placed between documents to inform the language model that the text following is unrelated to the previous content. Here is how you might retrieve and handle these special tokens:

```python  
# Download the vocab.bpe and encoder.json files  
# wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe  
# wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

import os, json

with open('encoder.json', 'r') as f:  
    encoder = json.load(f)  # ~equivalent to our 'vocab'

with open('vocab.bpe', 'r', encoding='utf-8') as f:  
    bpe_data = f.read()  
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]  
# ~equivalent to our 'merges'

# Handling special tokens  
special_tokens = [220, 2620, 220, 23748, 995, 10185]  
# ... additional processing  
```

The special tokens, such as the end-of-text token, are a crucial part of the encoding process, and their handling is often implemented in a domain-specific manner within the tokenizer.

#### Special Token Handling in the tiktoken Library {#tiktoken-library}

While `encoder.py` from OpenAI's GPT-2 does not explicitly detail special token handling, the tiktoken library, implemented in Rust, does include such functionality. The tiktoken library has specialized routines to detect and handle these tokens appropriately, ensuring their correct placement and interpretation during tokenization.

Below is an example of how special tokens might be handled in a library like tiktoken:

```rust  
// Rust code snippet from tiktoken library  
fn _encode_native(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<Rank> {  
    let special_regex = self._get_tl_special_regex();  
    // ... additional code for special token handling  
    // The code uses regular expressions to detect special tokens and handles them accordingly  
    // ... remaining implementation  
}  
```

This code handles special tokens during the encoding process, ensuring that they are correctly identified and processed.

#### Tokenization in Practice with tiktoken {#tokenization-practice-tiktoken}

Let's see an example of tokenization in action using the tiktoken library:

```python  
# Example code for using tiktoken library to tokenize text  
from tiktoken import Tokenizer

tokenizer = Tokenizer(model_name='gpt2')  
tokens = tokenizer.encode("Hello, world! How are you?")  
print(tokens)  
# Output: [220, 2620, 220, 23748, 995, 10185]

# Adding a special token to the text  
special_text = "Hello, world!   
#### Deeper Dive into Special Tokens {#special-tokens-deep-dive}
```
As we've discussed, special tokens play a significant role in tokenization, especially when transitioning from base language modeling to more nuanced applications like conversational AI. These tokens aren't just about marking the end of a text or a document; they're critical for demarcating conversational turns and managing dialogue flow.

![Special Tokens](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/4911_QaWkK9xnN.jpeg)

The image above illustrates an important point: special tokens are not an afterthought. They're integral to the design and functionality of modern language models. The text extracted from this image shows a snippet of code involving `encoder.json`. This file is pivotal in encoding because it represents our vocabulary, the set of tokens that the language model understands and operates with. Here's a similar code snippet that demonstrates how to load and use this encoder:

```python  
import os, json

with open('encoder.json', 'r') as f:  
    encoder = json.load(f)  # This is equivalent to our vocabulary  
```

Special tokens, such as `  
#### Understanding Tokenizer Training and Vocabulary {#tokenizer-training-vocabulary}

Continuing our exploration of tokenization, it's essential to understand the process of tokenizer training and the significance of the vocabulary it creates. The vocabulary of a tokenizer is like a map, guiding the translation between human-readable text and the language model's understanding. Let's delve deeper into how this vocabulary is built and used.

```python  
# The length of the encoder gives us the total number of tokens  
len(encoder)  # 256 raw byte tokens + 50,000 merges + 1 special token  
50257  
```

The encoder size, which combines raw byte tokens with merges and a special token, typically amounts to 50,257 for models like GPT. This count is not arbitrary; it represents the extensive set of tokens that a language model can recognize and generate.

#### Tokenization in Action with Tectoken {#tectoken-action}

To truly grasp the tokenizer's functionality, one can experiment with encoding and decoding strings using a library like Tectoken. By passing a string through the tokenizer, you can observe the conversion into tokens, and inversely, recover the original string from these tokens.

```python  
# Example of string tokenization  
encoded_string = tokenizer.encode("Your sample string")  
print(encoded_string)  
# Output: A list of token IDs representing "Your sample string"

# Decoding the tokens back into the original string  
decoded_string = tokenizer.decode(encoded_string)  
print(decoded_string)  
# Output: "Your sample string"  
```

When you implement your own training function, which libraries like Tectoken don't provide, you gain the ability to craft custom token vocabularies suited to your specific application.

#### GPT-4 Vocabulary Insights {#gpt4-vocabulary-insights}

Exploring the token vocabulary of GPT-4 provides valuable insights into the tokenizer's behavior. The first 256 tokens are typically raw individual bytes, and what follows is a visualization of token merges that occurred during the training phase.

![Token Merges Visualization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/5359_I-_Z3lUy4.jpeg)

The image above shows part of the GPT-4's vocabulary. Notably, the token `256` represents the first merge GPT-4 performed: combining two spaces into a single token. This merge order reflects the prioritization the model learned during its training phase.

#### Comparing Tokenization Strategies {#comparing-tokenization-strategies}

Moving beyond GPT-4 and Tectoken, let's compare different tokenization strategies employed by various libraries. SentencePiece, commonly used in the Llama and Mistral series, is one such library. It has the efficiency to both train and perform inference on BPE tokenizers.

The key difference between SentencePiece and other tokenizers lies in its approach to encoding:

- **Tikitoken**: Encodes to utf-8 and then applies BPE on bytes.  
- **SentencePiece**: Applies BPE directly on Unicode code points and, for rare code points, falls back to utf-8 bytes encoding which then translates into byte tokens.

This distinction fundamentally changes how text is processed and represented internally:

```python  
import sentencepiece as spm

# Assuming 'toy.txt' contains some example text we want to tokenize  
spm.SentencePieceTrainer.train('--input=toy.txt --model_prefix=m --vocab_size=2000')  
```

The `character_coverage` parameter in SentencePiece determines how the model treats rare code points—either mapping them to an `UNK` token or encoding them as bytes if the `byte_fallback` option is enabled.

#### Hands-On with SentencePiece {#hands-on-sentencepiece}

To better understand the practical application of SentencePiece, let's look at a hands-on example:

```python  
# Writing a toy.txt file with some random text  
with open('toy.txt', 'w') as f:  
    f.write("This is an example text to demonstrate how SentencePiece tokenizes.")

# Importing SentencePiece and training a tokenizer on the toy text file  
import sentencepiece as spm  
spm.SentencePieceTrainer.train('--input=toy.txt --model_prefix=toy_model --vocab_size=200')

# Loading the trained model and encoding text  
sp = spm.SentencePieceProcessor()  
sp.load('toy_model.model')  
encoded_text = sp.encode_as_pieces("This is an example text.")  
print(encoded_text)  
# Output: A list of subword tokens representing the input text  
```

Through this process, we can witness how SentencePiece tokenizes text directly on the Unicode code points and utilizes a fallback mechanism for rare characters, showcasing a unique approach to constructing a language model's vocabulary.

In summary, the intricacies of tokenizer training and vocabulary construction are pivotal to how language models understand and generate text. By dissecting these processes and comparing different tokenization strategies, we deepen our comprehension of the inner workings of language models and their tokenization mechanisms.  
### Diving into SentencePiece's Multitude of Options {#sentencepiece-options}

SentencePiece is a robust library with a myriad of configurations, which can be daunting due to its complexity. This flexibility is a result of SentencePiece being a mature tool in the NLP toolkit, designed to handle a diverse range of text processing needs. Its extensive set of configuration arguments, which has grown over time, reflects its capacity to adapt to various tokenization challenges.

#### The Complexity of SentencePiece Configuration {#complexity-sentencepiece-configuration}

The vast array of configurations in SentencePiece can be overwhelming. Here's a snippet showcasing some of the options available:

![SentencePiece Configuration](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/5498_e800GMu2_.jpeg)

For a complete list of training options and helpful documentation, the SentencePiece GitHub repository and its `sentencepiece_model.proto` file are valuable resources.

#### SentencePiece Model Options {#sentencepiece-model-options}

Exploring the `sentencepiece_model.proto` file, we encounter various options that affect how SentencePiece tokenizes text. Here are some notable configurations:

- `split_by_number`: Determines if there should be a boundary between numbers and non-number characters.  
- `split_by_whitespace`: Controls whether whitespace is used to split sentence pieces.  
- `treat_whitespace_as_suffix`: Adds whitespace as a suffix rather than a prefix.  
- `allow_whitespace_only_pieces`: Allows pieces consisting solely of whitespace.

These options exemplify how granular you can get with SentencePiece's tokenization rules, impacting the tokenizer's behavior significantly.

#### Configuring a SentencePiece Tokenizer {#configuring-sentencepiece-tokenizer}

Let's take a closer look at how one might configure SentencePiece for training a new tokenizer. The following code block represents a simplified configuration that mirrors the settings reportedly used for training the Llama 2 tokenizer:

```python  
import os

options = {  
    # input specification  
    'input': 'my_input_file.txt',  
    # other configurations omitted for brevity...  
}  
```

The `input` key specifies the source text file, while additional parameters would follow in the `options` dictionary. The goal is to replicate the tokenizer setup as closely as possible to that used by Llama 2, ensuring consistency and compatibility.

#### SentencePiece Training Arguments {#sentencepiece-training-arguments}

Within the myriad of SentencePiece options, some are more relevant to us than others. For instance, arguments like `--num_threads` and `--max_sentencepiece_length` directly affect the training process. Here's a quick rundown of some important arguments:

- `--num_threads`: Number of threads for training.  
- `--max_sentencepiece_length`: Maximum length of a sentence piece.  
- `--split_by_whitespace`: Whether to use whitespace for splitting sentence pieces.  
- `--split_digits`: Whether to split all digits into separate pieces.

These parameters are crucial when training a SentencePiece model to ensure the tokenizer aligns with our specific needs and computational resources.

#### Emulating Llama 2's Tokenizer Training {#emulating-llama2-tokenizer}

To emulate the tokenizer training of Llama 2, we can inspect the tokenizer model file released by Meta and use it to generate a prototype file that allows us to analyze all configurable options. Below is a Python snippet that approximates the setup used for training Llama 2's tokenizer:

```python  
# train a sentencepiece model on it  
# the settings here are (best effort) those used for training Llama 2  
import os

options = {  
    # input specification  
    'input': 'my_input_file.txt',  
    # other configurations omitted for brevity...  
}  
```

By inspecting the options and copying those that seem relevant, we can closely match the tokenizer's behavior.

#### The Importance of Raw Data in Tokenization {#importance-raw-data}

When dealing with language models, the preference is to minimize data preprocessing and maintain the integrity of the raw text. This approach contrasts with older NLP methods where text normalization was prevalent. In the context of LLMs, any form of text manipulation is considered unnecessary and potentially harmful, as it can distort the data's natural variability.

#### Special Tokens and Model Training in SentencePiece {#special-tokens-model-training}

SentencePiece allows for the definition of special tokens, which play a crucial role in the functioning of language models. These can include tokens like the beginning of sentence, end of sentence, and padding tokens. The process of training a SentencePiece model involves setting these special tokens and then executing the training to produce the `.model` and `.vocab` files needed for tokenization.

In summary, SentencePiece's extensive configuration options offer the ability to fine-tune tokenization to a high degree, replicating the tokenization strategy of sophisticated models like Llama 2. By understanding and leveraging these options, we can create a tokenizer that meets the specific needs of our language model, preserving the raw data's integrity and accommodating the peculiarities of different languages and character sets.  
### Understanding SentencePiece Special Tokens {#understanding-sentencepiece-special-tokens}

Special tokens play a crucial role in the structure of tokenized text for language models. They serve as important markers that assist the model in understanding the beginning, end, and other structural elements of the text. SentencePiece allows for the customization of these tokens, and their presence must be carefully managed. Below is a configuration snippet that sets up special tokens in SentencePiece:

```python  
# special tokens configuration  
unk_id=0,  # the UNK token MUST exist  
bos_id=1,  # the beginning of sequence token  
eos_id=2,  # the end of sequence token  
pad_id=-1,  # padding token, set to -1 to turn off

# systems configuration  
num_threads=os.cpu_count(),  # utilize all system resources

spm.SentencePieceTrainer.train(**options)

# Load the trained SentencePiece model  
sp = spm.SentencePieceProcessor()  
sp.load('tok400.model')

# Retrieve the vocabulary from the model  
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]  
```

In the output, we can expect to see the special tokens defined earlier, along with their assigned IDs:

```  
Out [361]: [['<unk>', 0],  # UNK token  
            ['<s>', 1],    # Beginning of sequence token  
            ['</s>', 2],   # End of sequence token  
            ['<0x00>', 3],  # Other tokens continue from here...  
            ...  
           ]  
```

By examining the vocabulary, we can see the tokens and their corresponding IDs, which include the UNK, beginning of sequence, and end of sequence tokens. In this example, the padding token is not used (`pad_id=-1`), and we also have a list of individual byte tokens starting from `<0x00>` onwards.

#### Special Tokens and Byte Tokens {#special-tokens-byte-tokens}

When we delve deeper into the vocabulary, we notice the structure of the tokens and their IDs. The vocabulary starts with the special tokens, followed by the byte tokens. The byte tokens are crucial as they serve as fallback options for rare code points which may not have been encountered during training. Here is how the special tokens and byte tokens appear in the vocabulary:

```python  
# Load the trained SentencePiece model  
sp = spm.SentencePieceProcessor()  
sp.load('tok400.model')

# Retrieve the vocabulary from the model  
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]

# The beginning of the vocabulary list  
[  
    ['<unk>', 0],  # UNK token  
    ['<s>', 1],    # Beginning of sequence token  
    ['</s>', 2],   # End of sequence token  
    ['<0x00>', 3],  # Byte tokens start from here...  
    ...  
]  
```

The byte tokens continue sequentially, and after them come the merges, which are combinations of individual tokens.

#### Exploring Merge Tokens and Individual Code Points {#exploring-merge-tokens}

SentencePiece also creates merge tokens, which are essentially combinations of individual tokens that represent more complex structures. Following the byte tokens, the vocabulary list will include these merges, showing only the parent nodes and their IDs:

```python  
# Merge tokens in the vocabulary  
[  
    ['<0xF7>', 250],  
    ['<0xF8>', 251],  
    ['<0xF9>', 252],  
    ...  
    ['n', 259],  
    ['_', 260],  
    ['t', 261],  
    ...  
]  
```

Finally, the vocabulary concludes with the individual code point tokens. These tokens are derived from the raw set of code points encountered in the training data:

```python  
# Individual code point tokens at the end of the vocabulary list  
[  
    ['w', 380],  
    ['y', 381],  
    ['p', 382],  
    ...  
]  
```

This ordering reflects how SentencePiece represents its vocabulary, starting with special tokens, followed by byte tokens, merge tokens, and then individual code point tokens.

#### The Role of Byte Fallback and Character Coverage {#byte-fallback-character-coverage}

SentencePiece can optionally fall back to UTF-8 bytes for rare code points. This behavior is determined by the `character_coverage` hyperparameter, which dictates the proportion of characters covered by the model. If a character is not covered, it will be represented by byte tokens. This byte fallback mechanism ensures that the tokenizer can handle any text, including characters that are not common enough to have their own dedicated tokens.

Here's an example of how these settings might be configured in SentencePiece:

```python  
options = {  
    # normalization configuration  
    'normalization_rule_name': 'identity',  # turn off normalization  
    'character_coverage': 0.99995,          # coverage for rare word treatment  
    'byte_fallback': True,                  # enable byte fallback  
      
    # merge rules  
    'split_digits': True,  
    'split_by_unicode_script': True,  
    ...  
      
    # special tokens  
    'unk_id': 0,  
    'bos_id': 1,  
    'eos_id': 2,  
    ...  
      
    # system resources  
    'num_threads': os.cpu_count()  # use all system resources  
}

spm.SentencePieceTrainer.train(**options)  
```

The settings here are designed to emulate the tokenizer training of sophisticated models like Llama 2, as they are equipped to handle a wide range of text inputs without requiring language-specific preprocessing.

#### SentencePiece as an End-to-End System {#sentencepiece-end-to-end-system}

SentencePiece enables the creation of a purely end-to-end system that does not rely on language-specific pre- or post-processing. The tokenizer can be trained on raw text, and it handles the complexities of various languages and scripts internally. This approach aligns with the modern trend in language modeling, where preserving the natural variability and integrity of the data is of utmost importance.

By understanding the intricacies of SentencePiece's tokenization process and its extensive configurations, we are better equipped to train and utilize tokenizers that are adaptable, efficient, and aligned with the needs of large language models.  
#### Tokenization Configuration Details {#tokenization-configuration-details}

In the construction of a tokenizer such as SentencePiece, several configuration options play a pivotal role in shaping the resulting vocabulary. The following snippet provides a glimpse of these configurations:

```python  
split_by_whitespace=True,  
split_by_number=True,  
max_sentencepiece_length=16,  
add_dummy_prefix=True,  
allow_whitespace_only_pieces=True,  
# special tokens  
unk_id=0, # the UNK token MUST exist  
bos_id=1, # the beginning of sequence token  
eos_id=2, # the end of sequence token  
pad_id=-1, # padding token, set to -1 to turn off  
# systems configuration  
num_threads=os.cpu_count(), # utilize all system resources

spm.SentencePieceTrainer.train(**options)  
```

After the configuration is set, we load the model and retrieve the vocabulary. The `encode` function provided by SentencePiece translates text into a sequence of token IDs:

```python  
# Load trained SentencePiece model and retrieve vocabulary  
sp = spm.SentencePieceProcessor()  
sp.load('tok400.model')  
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]  
```

#### Rare Code Points and Byte Fallback {#rare-code-points-byte-fallback}

The treatment of rare code points is a crucial aspect of tokenizer design. If a code point only appears once in a large corpus, it may not be included in the vocabulary, leading to unknown tokens. Consider a scenario where SentencePiece encounters code points that it has not seen during training, as is often the case with non-Latin characters like Korean:

```python  
# Encoding text containing rare code points  
ids = sp.encode("Hello 안녕")  
```

When `byte_fallback` is set to `True`, SentencePiece resorts to representing these rare characters as a sequence of byte tokens, ensuring no data is lost during tokenization.

#### UTF-8 Byte Tokens in SentencePiece {#utf8-byte-tokens}

SentencePiece can fall back to UTF-8 byte tokens when encountering unknown code points. These byte tokens are integrated into the vocabulary and can be observed as follows:

```python  
# UTF-8 Byte Tokens in the vocabulary  
vocab = [  
    ['<0x80>', 131],  
    ['<0x81>', 132],  
    ['<0x82>', 133],  
    # ...  
]  
```

The use of byte tokens allows SentencePiece to handle any text, including characters that are not part of its learned vocabulary.

#### Disabling Byte Fallback {#disabling-byte-fallback}

Disabling byte fallback (`byte_fallback=False`) results in the disappearance of byte tokens from the vocabulary. This changes the way the tokenizer handles unknown characters:

```python  
# Disabling byte fallback and retraining  
spm.SentencePieceTrainer.train(**options)

# Load model and observe the absence of byte tokens  
sp = spm.SentencePieceProcessor()  
sp.load('t.ko400.model')  
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]  
```

Without byte fallback, any text containing rare or unknown characters is represented by the unknown token (`<unk>`), which may not be ideal for language modeling purposes.

#### Impact on Language Modeling {#impact-language-modeling}

The presence or absence of byte fallback has significant implications for language modeling. For instance, without byte fallback, an entire string of rare characters may be reduced to a sequence of unknown tokens, offering no meaningful information to the model:

```python  
# Encoding without byte fallback  
ids = sp.encode("Hello 안녕")  
```

In such cases, the tokenizer outputs a single ID representing the unknown token, which is not informative and can negatively impact the model's ability to understand and generate text involving rare characters.

#### Why Byte Fallback Matters {#why-byte-fallback-matters}

Byte fallback is essential for language models like Llama that are designed to handle a wide range of text inputs, including rare characters. By encoding these rare characters as a sequence of bytes, the model receives information that can help it understand and generate text more effectively.

#### SentencePiece Training with Byte Fallback {#sentencepiece-training-byte-fallback}

Here is an example of configuring and training a SentencePiece tokenizer with byte fallback enabled:

```python  
max_sentence_length=4192, # max number of bytes per sentence  
seed_sentencepiece_size=10000000,  
shuffle_input_sentence=True,  
# rare word treatment  
character_coverage=0.99995,  
byte_fallback=True,  
# merge rules  
split_digits=True,  
split_by_unicode_script=True,  
split_by_whitespace=True,  
split_by_number=True,  
max_sentencepiece_length=16,  
add_dummy_prefix=True,  
allow_whitespace_only_pieces=True,  
# special tokens  
unk_id=0, # the UNK token MUST exist  
bos_id=1, # the beginning of sequence token  
eos_id=2, # the end of sequence token  
pad_id=-1, # padding token, set to -1 to turn off  
# systems configuration  
num_threads=os.cpu_count(), # utilize all system resources

spm.SentencePieceTrainer.train(**koptions)  
```

By enabling `byte_fallback`, we ensure that the tokenizer can handle any text, making it a robust tool for language modeling.

#### Tokenization Anomalies {#tokenization-anomalies}

An interesting phenomenon in tokenization is how SentencePiece treats whitespace. When decoding tokens, whitespace is often visualized as a bold underline (`▁`). Additionally, some tokenizers may add a dummy prefix to the text, which results in an extra space at the beginning of the encoded string:

```python  
# Observations on whitespace and dummy prefixes  
sp = spm.SentencePieceProcessor()  
sp.load('tok400.model')  
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]  
```

Understanding these peculiarities is important for anyone working with tokenizers, as they can affect the interpretation of the model's outputs and its overall performance.

#### Tokenization Configuration Explained {#tokenization-configuration-explained}

The intricate details of tokenization configuration are what make the tokenizer work in specific ways. One important factor is the `model_prefix` option:

```python  
# File argument  
model_prefix='tok400', # output filename prefix  
```

This indicates the output filename prefix for the trained model. The `add_dummy_prefix=True` setting is particularly interesting. As per the documentation, this option adds a dummy whitespace at the beginning of text. This is done to treat words at the beginning of sentences and words in the middle of sentences in the same way. Such a setting tries to mitigate the issue of different representations for what may conceptually be the same word.

#### Normalization in Tokenization {#normalization-in-tokenization}

In SentencePiece, the `NormalizerSpec` message encodes various parameters for string normalization, which includes:

- The name of the normalization rule.  
- Pre-compiled normalization rules (`precompiled_charsmap`).  
- An option to add dummy whitespace at the beginning of text.

This is part of SentencePiece's protocol buffer configuration and plays a crucial role in how text is processed before tokenization.

#### Pre-processing in Tokenization {#pre-processing-tokenization}

When we talk about pre-processing in tokenization, the `add_dummy_prefix` option is a significant factor. Essentially, SentencePiece will automatically prepend a space to the string before tokenization:

```python  
# Pre-processing by adding a space  
In [379]: ids = sp.encode("example string")  
```

This space addition transforms the input by normalizing the beginning of sentences, making "world" and " world" the same for the tokenizer:

```python  
# Resulting encoding  
In [379]: ids = sp.encode(" world")  
```

The image below demonstrates the outcome of such pre-processing, where both instances of "world" result in a token prefixed with a space:

![Tokenization Example](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6115_MaZNDCBgwt.jpeg)

By doing so, the tokenizer treats words at the beginnings of sentences and within sentences uniformly, which is an essential step for models like Llama 2.

#### Protocol Buffer Representation of Tokenizer {#protocol-buffer-tokenizer}

For those who want to delve deeper into the tokenizer's configuration, exploring the raw protocol buffer representation can provide insights into how the tokenizer has been set up. This information can reveal specifics like the normalization rules used and how the tokenizer was trained:

```proto  
# NormalizerSpec from the tokenizer's proto file  
normalizer_spec {  
  name: "..."  
  precompiled_charsmap: "..."  
}

# TrainerSpec from the tokenizer's proto file  
trainer_spec {  
  input: "..."  
  ...  
}  
```

To replicate the tokenization process of models like Meta's Llama 2, one would copy these settings into their configuration.

#### Options for SentencePiece Training {#options-sentencepiece-training}

When training your tokenizer, a variety of options are available to tailor it to your specific needs. Here's a detailed configuration for SentencePiece training:

```python  
options = {  
  # input spec  
  input="...",  
  # file arg  
  model_prefix='tok400',  
  # algorithm spec  
  model_type='bpe',  
  vocab_size=400,  
  # normalization  
  normalization_rule_name='identity',  
  remove_extra_whitespaces=False,  
  input_sentence_size=200000000, # max number of training sentences  
  # rare word treatment  
  character_coverage=0.99995,  
  byte_fallback=True,  
  # merge rules  
  split_digits=True,  
  split_by_unicode_script=True,  
  split_by_whitespace=True,  
  split_by_number=True,  
  max_sentence_length=16,  
  add_dummy_prefix=True,  
  allow_whitespace_only_pieces=True,  
  # special tokens  
  unk_id=0, # the UNK token MUST exist  
  bos_id=1, # the BOS token  
  eos_id=2, # the EOS token  
  pad_id=-1, # padding token, set to -1 to turn off  
  # systems  
  num_threads=os.cpu_count(), # use all system resources  
}

spm.SentencePieceTrainer.train(**options)  
```

This configuration is essential for ensuring that the tokenizer can handle a wide variety of text inputs and is robust enough for different use cases.

#### Further Insights into Tokenization {#further-insights-tokenization}

The tokenization process is not only about splitting text into tokens. It involves a series of decisions and configurations that can significantly impact the performance of a language model. For example, the inclusion of an UNK token (`unk_id=0`) is mandatory, and the manner in which byte fallbacks are handled (`byte_fallback=True`) can affect how the tokenizer deals with rare words or characters.

Despite its efficiency and widespread use in the industry, SentencePiece has its quirks and complexities. It's not always straightforward, and the documentation may not cover all nuances, requiring practitioners to spend time experimenting and understanding the inner workings of the tokenizer.

For those interested in training their tokenizer, SentencePiece provides a repository that, despite its documentation challenges, is a valuable resource. The repo allows for a deep dive into tokenization, enabling the creation of a custom tokenizer that suits specific needs.  
### Understanding the Role of Vocabulary Size in LLMs {#understanding-vocab-size}

As we continue our exploration of the tokenization process in large language models (LLMs), the concept of vocabulary size is a pivotal factor that warrants a deeper understanding. The vocabulary size is a key parameter in the configuration of LLMs, and it has a direct impact on the model's performance and its ability to understand and generate language.

#### Vocabulary Size and Token Embedding {#vocabulary-size-token-embedding}

In the Jupyter Notebook for the video lecture, we encounter the following code snippet that illustrates the setup of vocabulary size within a GPT-style model:

```python  
# Setting up the vocabulary size  
chars = sorted(list(set(text)))  
vocab_size = len(chars)  
stoi = { c: i for i, c in enumerate(chars) }  
itos = { i: c for i, c in enumerate(chars) }  
```

Here, `vocab_size` is determined by the number of unique characters in the dataset, which in our small example, was a mere 65 characters. However, in a full-scale model, this number will be much larger, encompassing a wide array of tokens.

#### Data Splits and Batch Generation {#data-splits-batch-generation}

The model goes on to split the data into training and validation sets, creating batches that will be used during the training process:

```python  
data = torch.tensor(encode(text), dtype=torch.long)  
n = int(0.9*len(data))  # first 90% will be train, rest val  
train_data = data[:n]  
val_data = data[n:]

def get_batch(split):  
    # generate a small batch of data of inputs x and targets y  
    data = train_data if split == 'train' else val_data  
    ix = torch.randint(len(data)-block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  
    return x, y  
```

This code ensures that the model is exposed to various examples during training, which helps it to generalize better when making predictions.

#### The Role of Embedding Tables {#role-embedding-tables}

The model leverages two crucial embedding tables: one for token embeddings and one for position embeddings. Both play a vital role in interpreting the input data:

```python  
class GPTLanguageModel(nn.Module):  
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head):  
        super().__init__()  
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  
        # Additional layers and initialization methods are defined here...  
```

The `vocab_size` is significant because it determines the number of rows in the token embedding table. Each token is associated with a vector of size `n_embd`, which the model will learn during training.

#### From Embeddings to Logits {#embeddings-to-logits}

After processing through the transformer blocks, the model outputs logits, which represent the probabilities of the next token in the sequence:

```python  
class GPTLanguageModel(nn.Module):  
    # ... (previous class code)  
      
    def forward(self, idx, targets=None):  
        B, T = idx.shape  
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)  
        # Position embeddings and transformer blocks are applied here...  
        logits = self.lm_head(tok_emb)  # (B, T, vocab_size)  
        # Loss calculation happens here if targets are provided...  
```

The logits are then used to calculate the probability distribution over the entire vocabulary for the next token prediction.

#### Token Embedding Table Growth {#token-embedding-table-growth}

As the vocabulary size increases, the embedding table grows accordingly. This expansion has implications for the model's complexity and resource requirements, as each additional token requires more parameters to be learned:

```python  
# As vocab_size increases, so does the embedding table  
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  
```

#### Initialization of Model Weights {#initialization-model-weights}

Proper initialization of weights is crucial for the success of deep learning models. The GPTLanguageModel class includes specific instructions for initializing weights of linear and embedding layers:

```python  
def _init_weights(self, module):  
    if isinstance(module, nn.Linear):  
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
        if module.bias is not None:  
            torch.nn.init.zeros_(module.bias)  
    elif isinstance(module, nn.Embedding):  
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
```

This code ensures that the weights are initialized in a way that promotes effective learning.

#### The Impact of Large Vocabulary on Linear Layers {#impact-large-vocabulary}

With a larger vocabulary, the linear layer responsible for producing logits must perform more computations:

```python  
self.lm_head = nn.Linear(n_embd, vocab_size)  
```

Each token in the vocabulary necessitates an additional dot product in this layer, increasing the computational load.

#### Limitations of Infinite Vocabulary Size {#limitations-infinite-vocabulary}

An infinitely large vocabulary size is unfeasible due to several limitations:

- The token embedding table size would become impractically large.  
- The linear layer's computational complexity would increase significantly.

These constraints necessitate a balance between vocabulary size and model efficiency.

#### Multi-Head Attention and Layer Normalization {#multi-head-attention-layer-norm}

In the transformer architecture, multi-head attention and layer normalization are key components that enable the model to handle dependencies between tokens effectively:

```python  
class Block(nn.Module):  
    def __init__(self, n_embd, n_head):  
        super().__init__()  
        head_size = n_embd // n_head  
        self.sa = MultiHeadAttention(n_head, head_size)  
        self.ffwd = FeedForward(n_embd)  
        self.ln1 = nn.LayerNorm(n_embd)  
        self.ln2 = nn.LayerNorm(n_embd)  
      
    def forward(self, x):  
        x = x + self.sa(self.ln1(x))  
        x = x + self.ffwd(self.ln2(x))  
        return x  
```

The combination of these elements allows the LLM to capture complex patterns and nuances in the data.

### Conclusion

The structure and size of the vocabulary are critical components in the design of LLMs. By understanding the role of vocabulary size, embedding tables, and the computational demands of linear layers, we gain better insights into the inner workings of models like GPT. While the urge to increase vocabulary size to capture more nuances of language exists, practical constraints dictate a careful consideration of the trade-offs involved.  
#### Exploring Vocabulary Size and Model Extensibility {#vocab-size-model-extensibility}

Understanding the vocabulary size of language models is one thing, but considering the implications of modifying this aspect is another. As we delve deeper into the intricacies of tokenization in LLMs, it's important to examine the practical implications of adjusting the vocabulary size, especially when extending pre-trained models.

##### Growing Vocabulary Size: Implications {#growing-vocab-implications}

Increasing the vocabulary size of a language model certainly has its ramifications:

- **Computational Expense**: The linear layer (`lm_head`) responsible for generating logits becomes more computationally demanding. More tokens equate to increased calculations.

    ```python  
    # Growing linear layer with larger vocabulary size  
    self.lm_head = nn.Linear(n_embd, vocab_size)  
    ```

- **Sparse Token Occurrences**: A larger vocabulary could mean that individual tokens appear less frequently in the training data. This rarity could lead to undertrained token vectors, as they engage in fewer forward-backward passes necessary for learning.

- **Sequence Compression**: A larger vocabulary can compress sequences significantly, potentially encoding more information into fewer tokens. While this can enhance the model's ability to process larger texts, it might also strain the model's ability to unpack and utilize the denser information during the forward pass.

##### Extending Pre-Trained Models {#extending-pretrained-models}

When fine-tuning pre-trained models for specific tasks, such as chat applications, it's common to introduce additional special tokens. These tokens help manage conversation metadata and maintain the structure of interactions. However, this expansion necessitates minor model surgery:

- **Adding New Tokens**: To introduce new tokens, we need to resize the token embedding table by adding new rows and initializing them with small, random values.

    ```python  
    # Resize the embedding table to include new tokens  
    self.token_embedding_table = nn.Embedding(vocab_size + num_new_tokens, n_embd)  
    ```

- **Extending Linear Weights**: The linear layer must also account for the new tokens, necessitating an increase in the number of dot products.

    ```python  
    # Extend the linear layer's weights for new tokens  
    self.lm_head = nn.Linear(n_embd, vocab_size + num_new_tokens)  
    ```

This process is relatively straightforward and allows for the base model parameters to remain frozen while only the new token embeddings are trained.

#### Leveraging Gist Tokens for Efficient Prompt Compression {#gist-tokens-prompt-compression}

In the context of using language models with long prompts, the computational load can be intense. Researchers at Stanford University have addressed this challenge in a paper titled "Learning to Compress Prompts with Gist Tokens." By introducing new 'gist' tokens, they've developed a method to distill lengthy prompts into more manageable representations, enabling greater computational efficiency without significant losses in output quality.

![Gist Tokens Abstract](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6527_BEVR6WIDA.jpeg)

The gist tokens are trained via distillation to stand in for the original long prompts, resulting in up to 26 times compression and notable reductions in floating-point operations (FLOPs), wall time, and storage requirements.

#### Beyond Text: Multimodal Tokenization in Transformers {#beyond-text-multimodal-tokenization}

The application of transformers isn't limited to textual data. There's a growing trend to adapt these models for multimodal inputs such as images, videos, and audio. The key lies in tokenizing these different data types.

![Multimodal Tokenization](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6532_SKrs8PuMT.jpeg)

For example, an image can be partitioned into patches, each represented as an integer token. This approach can be extended to soft tokens, where representations are passed through bottlenecks akin to autoencoders, allowing the model to handle a diverse array of input modalities without substantial changes to the underlying architecture.

#### Envisioning World Simulators with Video Generation Models {#world-simulators-video-generation}

The concept of using generative models as simulators of the physical world has been pushed further by developments in video generation. A notable example is the Sora model from OpenAI, which operates on spacetime patches of video and image latent codes to generate high-fidelity videos.

![Sora Overview](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6602_uQLDr2FnB.jpeg)

This approach to video generation suggests a promising avenue for creating general-purpose simulators that can aid in understanding and interacting with the world around us.

#### Sora's Unified Representation for Visual Data {#sora-unified-visual-data}

The Sora project focuses on developing a unified representation for visual data, enabling large-scale training of generative models. It represents a substantial leap in the field, combining text tokens with visual patches to create a cohesive model capable of processing and predicting a variety of visual modalities.

![Sora's Unified Representation](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6676_YuRNfvcAM.jpeg)

This integration of diverse data types into the transformer architecture paves the way for more robust and versatile AI systems.

#### Video Chunking: A New Frontier in Tokenization {#video-chunking}

The idea of chunking videos into tokens extends the principle of tokenization from text to a more dynamic medium. Whether through hard tokens in autoregressive models or soft tokens in diffusion models, the encoding of video data into a tokenized format is an area ripe for exploration.

![Video Chunking](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6696_np4QtCDPzy.jpeg)

As research in this area progresses, we can anticipate further innovations that enable transformers to process not just static images but also the flow and complexity of video content.

#### Reflecting on Tokenization {#reflecting-on-tokenization}

Having ventured deep into the tokenization algorithm, we circle back to the initial examples of language model limitations and can now understand why they occur. A language model's difficulty with spelling or handling string-related tasks often stems from how tokenization condenses information.

![Tokenization Reflection](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6706_G1C-VxPBo.jpeg)

The tokenization process is at the core of these challenges, shaping the way models perceive and process language, and influencing their capabilities and limitations.  
### The Complexities of Long Tokens in LLMs {#long-tokens-complexities}

The challenges of tokenization in LLMs go beyond just the initial conversion of text to tokens. As we delve deeper into the mechanics of tokenization, we observe that the length and composition of tokens significantly impact the model's performance.

So fundamentally, this is because characters are chunked up into tokens, some of which can be quite lengthy.

For instance, a token as long as "that default style" turns out to be a single entity within the GPT-4 vocabulary.

![Long Token Example](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6731_IFYdY-rfu.jpeg)

This extended character sequence within a single token raises questions about the model's ability to handle tasks related to the spelling of such tokens. When prompted with questions about the number of specific letters in "dot default style," the model inaccurately counted, suggesting a difficulty with tasks that require breaking down longer tokens.

#### Troubles with String Reversal {#string-reversal-trouble}

Exploring further, we can look at a character-level task like string reversal. When asking GPT-4 to reverse "default style," the model produced a jumbled result, unable to accurately perform the task. This inability hints at an underlying issue with the tokenization strategy.

However, when the task was reformulated to first list every character separated by spaces and then reverse that list, GPT-4 was able to correctly reverse the characters. It suggests that once the string is broken down into individual characters—effectively individual tokens—the model can process and output them correctly.

### Why Are LLMs Worse at Non-English Languages? {#llms-non-english}

The struggle of language models with non-English languages can be attributed to two main factors:

1. The models encounter less non-English data during training, resulting in less familiarity with these languages.  
2. The tokenizers themselves are not sufficiently trained on non-English data.

An example that illustrates this point is the English phrase "hello, how are you?" which tokenizes into five tokens, while its Korean counterpart translates into 15 tokens. This disparity in tokenization causes everything in non-English languages to be more bloated and diffuse, leading to poorer performance by the model.

![Tokenization of Non-English Languages](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6855_xGHNJWUWt.jpeg)

### Tokenization and Arithmetic {#tokenization-arithmetic}

The tokenization process also affects the model's ability to perform simple arithmetic. Numbers are tokenized in an arbitrary manner, which can be problematic for tasks that require precise mathematical operations.

For instance, there is an algorithmic approach to addition that works at the character level, requiring the model to refer to specific parts of the digits. However, the tokenization of numbers might not align with this approach, leading to difficulties in performing such operations correctly.

A detailed analysis reveals that four-digit numbers can be tokenized in various ways: as a unique token or as combinations of two tokens—either a 1-3, 2-2, or 3-1 pattern. This inconsistency in tokenization complicates the execution of numerical algorithms, forcing the model to handle a multitude of special cases.

![Tokenization Patterns in Numbers](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6915_do_YMrIWM.jpeg)

To improve arithmetic performance, some models like Meta's Llama 2 were trained with digits split into individual tokens, a strategy aimed at simplifying numerical tokenization.

#### Challenges with Coding in Python {#challenges-coding-python}

The tokenization issues extend to coding tasks as well, particularly in languages like Python. For GPT-2, the encoding efficiency for handling spaces—a critical aspect of Python's syntax—was poor, with each space being tokenized individually. This significantly reduced the context length that the model could attend to, a tokenization inefficiency that was later addressed in GPT-4.

### Special Tokens as an Attack Surface {#special-tokens-attack-surface}

The knowledge of special tokens can be exploited to confuse language models. For example, when GPT-4 encounters the string `<endoftext>`, it fails to recognize it or print it as requested, suggesting an issue with the handling of special tokens. This behavior points to potential vulnerabilities where special tokens could be used to disrupt the normal functioning of a model.

![Special Token Confusion](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/6731_IFYdY-rfu.jpeg)

Understanding the tokenization process and its quirks can reveal much about the underlying behavior of LLMs. Whether it's spelling words, reversing strings, processing non-English languages, performing arithmetic, coding in Python, or handling special tokens, tokenization remains a central yet challenging aspect of language model design and performance.  
### Understanding Training Space Issues in LLMs {#training-space-issues}

In the realm of tokenization, the treatment of spaces in text strings is a subtle yet critical concern. It affects how a Large Language Model (LLM) interprets and continues a given sequence of tokens. This aspect of tokenization is particularly relevant when we consider how LLMs such as GPT-3.5 Turbo handle text inputs for completions.

#### Space Handling in Completion Models {#space-handling-completion}

To better understand the impact of spaces on token sequences, consider the following scenario with GPT-3.5 Turbo. When provided with a tagline for an ice cream shop, the model is expected to continue the sequence seamlessly. However, if the input text ends with a trailing space, the LLM issues a warning:

> Your text ends in a training space which causes it worse performance due to how API splits text into tokens.

![Training Space Issue Warning](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7134_2554Slv_5.jpeg)

This warning highlights a potential performance degradation due to the way the model splits text into tokens. Let's dissect the issue further:

- A trailing space before submitting a completion request might lead to token 220 being appended to the sequence prematurely.  
- This addition is out of the ordinary as the space character is typically a prefix to other tokens in GPT models.  
- If a space is part of the next token, inserting it as a separate token leads to an out-of-distribution scenario for the LLM.

```python  
# Example of token encoding with trailing space  
# Here "space O" is token 880, represented as ' O'  
input_text = "Here's a tagline for an ice cream shop "  
# The trailing space becomes a separate token, potentially causing issues  
encoded_text = encode(input_text)  
# encoded_text now includes an out-of-distribution token at the end  
```

#### Tokenization Oddities and Unstable Tokens {#tokenization-oddities}

The tokenization system within LLMs is complex and handles a variety of special cases. One such complexity is the handling of what are referred to as **unstable tokens**.

Unstable tokens are sequences that are not tokenized consistently across different instances, which can lead to unpredictable model behavior. The `lib.rs` file in the tokenization repository contains code that deals with these unstable tokens, indicating the intricacies involved:

```rust  
// Excerpt from lib.rs handling unstable tokens  
match next_special {  
    // Pushing the special token  
    Some(m) => {  
        let piece = m.as_str();  
        let token = self.special_tokens_encoder[piece];  
        ret.push(token);  
        start = m.end();  
        last_piece_token_len = 0;  
    }  
    None => break,  
}

// last_piece_token_len is used for determining unstable tokens  
(ret, last_piece_token_len)  
```

Unstable tokens could arise from patterns like `\s*[\r\n]+`, which indicate spaces followed by line breaks. These patterns can be unstable because the presence or absence of a line break could alter the tokenization of adjacent characters.

```rust  
// More code dealing with unstable tokens  
fn _encode_unstable_native(  
    // Encoding logic for unstable tokens  
    let mut reencoded = byte_pair_encode(  
        &unstable_bytes[..unstable_bytes.len() - last_decoded.1],  
        &self.encoder,  
    );  
    reencoded.extend(byte_pair_encode(  
        &unstable_bytes[unstable_bytes.len() - last_decoded.1..],  
        &self.encoder,  
    ));  
    completions.insert(reencoded);  
)  
```

This handling of unstable tokens is not typically documented, but it is a crucial part of the tokenizer's robustness. It shows the effort to maintain the consistency of token sequences even when faced with tricky text patterns.

![Unstable Tokens Handling](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7427_6yUaEp9dY.jpeg)

### The Desired Behavior of Completion APIs {#desired-completion-behavior}

What we expect from a completion API is not just to append the next full token after a partial token sequence but to consider a range of possible tokens that could follow. For example, taking the input "default cell star," we would like the model to predict a continuation that makes sense within the context, rather than being limited to the next most likely token.

This presents a challenge because the model must navigate the space of all possible continuations, considering not only the likelihood of subsequent tokens but also the potential re-tokenization of the initial input. It's a complex balancing act that stems directly from the tokenization process.

![Complex Token Sequencing](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7442_CFgB9dU_V.jpeg)

This complexity is encapsulated in the handling of what are described as "unstable native" tokens, which are addressed in the tokenization code:

```rust  
fn _encode_unstable_native(  
    let mut reencoded = byte_pair_encode(  
        &unstable_bytes[..unstable_bytes.len() - last_decoded.1],  
        &self.encoder,  
    );  
    reencoded.extend(byte_pair_encode(  
        &unstable_bytes[unstable_bytes.len() - last_decoded.1..],  
        &self.encoder,  
    ));  
    completions.insert(reencoded);  
)  
```

In this snippet, we see the tokenizer re-encoding unstable bytes, attempting to find stable tokens that can be consistently used in the model's predictions. The pursuit of stability in tokenization is an ongoing challenge, reflected in the intricate code dedicated to addressing these issues.

Tokenization, while not always at the forefront of discussions about LLMs, remains a foundational component that influences everything from model performance to the nuances of how models interpret and generate text. As we continue to develop and refine these models, the subtleties of tokenization will undoubtedly play a critical role in shaping their evolution.  
### The Enigma of Unstable Tokens in LLMs {#unstable-tokens-enigma}

In the labyrinthine world of LLMs, the phenomenon of unstable tokens stands out as a particularly intriguing oddity deserving a deeper exploration. These tokens, often inconsistent in their tokenization, can lead to some truly bizarre model behaviors.

![Unstable Tokens](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7487_XfIEWH26t.jpeg)

#### The Mysterious Case of SolidGoldMagikarp {#solidgoldmagikarp-mystery}

One of the most captivating examples of tokenization gone awry is the case of `SolidGoldMagikarp`. This seemingly innocuous string has gained a peculiar kind of internet notoriety among those who work with LLMs. One might even consider dedicating an entire video to unraveling the mystery behind such unstable tokens.

The significance of these tokens is underscored by their oddity:

- They can cause LLMs to fail at spelling simple words.  
- They can hinder LLMs in performing basic string processing tasks.  
- They can deteriorate performance in non-English languages.  
- They can be the reason LLMs struggle with arithmetic.  
- They can make coding in Python using GPT-2 more difficult than it should be.  
- They can result in abrupt termination of LLMs upon encountering specific strings.

The list of challenges goes on, all tracing back to the complexities of tokenization.

#### Clustering Tokens and Unveiling Anomalies {#clustering-tokens-anomalies}

In an attempt to demystify these tokens, one researcher ventured into the token embedding space, clustering tokens based on their embedding representations. This exploration revealed a cluster of tokens that appeared distinctly odd when compared to the others.

![Token Clustering](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7502_McIuVVQ0a.jpeg)

The findings indicated that certain tokens, including `SolidGoldMagikarp` and others like `TheNitromeFan` and `cloneembedreportprint`, consistently positioned themselves as outliers. These tokens did not seem to fit semantically within their clusters, nor did they resemble typical tokens one would expect to find.

![Clustering Tokens](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7512_jhni-Dw5F.jpeg)

The extracted insights from the analysis pointed to something more systematic:

- **Prompt Generation**: The investigation involved creating prompts to elicit responses from LLMs.  
- **Finding Weird Tokens**: Some tokens were consistently difficult for LLMs to handle.  
- **Clustering**: Tokens were clustered in embedding space to discern patterns.  
- **Anomalous Tokens**: Certain tokens, when used in prompts, led to strange or evasive responses from the models.

The specific behaviors documented were as diverse as they were perplexing, ranging from evasive answers to hallucinatory completions. In some cases, the LLMs would replace these tokens with semantically or phonetically similar words, creating a kind of 'token hallucination'.

#### Probing the Unpredictable: GPT-3's Response to Anomalous Tokens {#probing-unpredictable}

The peculiar behavior of LLMs in response to anomalous tokens was put to the test using a series of prompt templates, reformulated in minor ways to gauge the LLM's reaction. GPT-3, with temperature set to zero, was chosen as the model most likely to follow straightforward instructions without the unpredictability introduced by temperature or daily updates.

However, when fed these anomalous tokens, GPT-3's responses were inconsistent and often nonsensical. The model struggled to repeat the tokens, instead providing a variety of responses:

- **Evasion**: Replies like "I can't hear you" or "I don't know what you're trying to say" were common.  
- **Hallucinatory Completions**: The model would sometimes repeat entirely different tokens, such as transforming 'DevOnline' to 'guildcon' or 'idiosyncrasy'.

```plaintext  
'Please can you repeat back the string '<token string>' to me?'  
```

This simple prompt, when paired with the problematic tokens, revealed the depth of the issue. The LLM's inability to handle these tokens consistently points to a fundamental flaw in how these tokens are processed or understood by the model.

![Anomalous Token Response](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7566_UGshiBvwC.jpeg)

#### A Deeper Dive into Token Anomalies {#deeper-dive-token-anomalies}

As the research delved deeper into these peculiar tokens, it became apparent that the category of 'weird' or 'forbidden' tokens was not clearly defined, with various degrees of anomaly observed. Some tokens, when prompted to be repeated back, would be replaced with seemingly random numbers or words.

The persistence of these anomalies across multiple tests and over time suggests a consistent issue within the model's handling of these tokens. It was noted that asking ChatGPT to repeat certain tokens would lead to stalling, with the model unable to proceed past the first quotation mark.

#### Origins of Anomalous Tokens: A Speculation {#origins-anomalous-tokens}

The genesis of these anomalous tokens may lie in the web scraping process that contributed to the creation of GPT's 50,257-token set. While the training texts for GPT models are curated, it's conceivable that the outliers were scraped from less conventional sources such as e-commerce backends, Reddit threads, or online game logs. These sources might not have been included in the training corpora, leading to the model's unfamiliarity and erratic behavior when encountering these tokens.

The speculation is that these tokens' odd placement near the centroid in embedding space might be due to their minimal involvement in training, creating a kind of 'token limbo' where the model simply doesn't know how to react. This uncertainty could be exacerbated by floating point errors during the model's forward propagation.

![Token Origins](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7546__hgX9Ae7dj.jpeg)

The research, while still a work in progress, opens up fascinating questions about the nature of tokenization and its impact on LLM behavior. Feedback and further exploration are encouraged as we continue to unravel the complex tapestry of LLM tokenization.

As we journey through the convoluted pathways of tokenization, the SolidGoldMagikarp stands as a testament to the unpredictability and complexity inherent in the world of LLMs—a world where tokens, much like words, hold power beyond their surface meaning.  
### Trigger Words and Model Misbehavior {#trigger-words-model-misbehavior}

In the intricate domain of Large Language Models (LLMs), certain terms act as "trigger words," leading to perplexing and often inappropriate behavior from the model. These anomalies in tokenization can cause LLMs to exhibit behaviors that deviate from safety guidelines, including the use of profanity or generating responses that seem misaligned with the context.

#### Fishing for Anomalous Tokens {#fishing-anomalous-tokens}

The quest to understand these peculiar tokens has led to the identification of a set of indices representing these anomalies. The following list showcases the range of tokens that have been associated with erratic LLM behavior:

```plaintext  
[188, 189, 190, ... 39906]  
```

A closer examination of these tokens reveals a pattern. Many of them, such as `SolidGoldMagikarp`, `RandomEditorWithX`, `isSpecialOrderable`, and `DragonMagazine`, seem to have originated from places on the web that were not included in the LLMs' training corpora, such as e-commerce backends and gaming logs.

#### The Reddit Connection: SolidGoldMagikarp {#reddit-connection}

The peculiar case of `SolidGoldMagikarp`, a token that has baffled many, is believed to have originated from Reddit. It appears that a discrepancy between the tokenization dataset and the actual language model training dataset may have given rise to a dedicated token for this particular Reddit user due to their high frequency of mention within the tokenization data.

#### The Untrained Token Theory {#untrained-token-theory}

When the language model is trained without the presence of certain tokens, such as `SolidGoldMagikarp`, in its training set, those tokens never get activated. They remain untrained within the embedding table, akin to unallocated memory. At test time, invoking such a token can lead to undefined behavior, as the embedding table outputs an untrained vector that the Transformer model cannot process effectively.

This phenomenon is not just limited to `SolidGoldMagikarp` but extends to any token that the model has not encountered during training, which results in out-of-sample behavior when such tokens are used.

#### The Challenge of Evasion and Hallucinatory Completions {#evasion-hallucinations}

When LLMs encounter these anomalous tokens, they may employ evasion strategies or hallucinatory completions, replacing the token with a different, often thematically or phonetically similar word. For example, `DevOnline` might be transformed into `dog`, while `InstoreAndOnline` could lead to a variety of completions like `Institute` or `Instruction`.

#### Security and Testing Implications {#security-testing-implications}

The unpredictable nature of these tokens has raised concerns about security and testing. The LLM's evasion strategies, such as claiming not to understand a prompt or refusing to repeat certain strings, are indicative of a deeper issue within the model's understanding of its tokenization process.

#### Tokenization: The Root of Many LLM Peculiarities {#tokenization-root-peculiarities}

The issues arising from tokenization are widespread and impact many areas of LLM functionality:

- Spelling of words  
- Simple string processing tasks  
- Performance in non-English languages  
- Simple arithmetic  
- Coding in Python  
- Handling of specific trigger strings

These challenges emphasize the critical importance of tokenization in the overall behavior and reliability of LLMs.

### The Importance of Efficient Tokenization {#efficient-tokenization-importance}

The efficiency of tokenization is a pressing concern, particularly when considering the cost implications of processing data with LLMs. Different data formats can impact the density of tokenization, with formats like YAML being more token-efficient compared to JSON.

For instance, the same data represented in JSON and YAML can result in vastly different token counts:

![Token Efficiency Comparison](https://ik.imagekit.io/micsmco/frames/lets_build_the_gpt_tokenizer/7760_1eAD91B2d.jpeg)

#### Final Recommendations on Tokenization {#final-recommendations-tokenization}

As we delve into the world of tokenization, it becomes clear that we must not overlook its complexities. Tokenization is fraught with potential pitfalls, including security and safety issues. The dream of eliminating tokenization as a necessary step in LLMs remains an alluring goal for researchers and developers alike.

When building applications involving LLMs, consider the following:

- Reusing tokens from models like GPT-4 could be a viable option.  
- If creating a new vocabulary is required, utilizing the Byte Pair Encoding (BPE) algorithm with tools like sentencepiece can be effective, though one must be mindful of the multitude of settings.

The ongoing quest to find efficient encoding schemes demands thorough analysis and optimization, as the token economy influences both computational resources and financial costs.  
### Understanding the Inner Workings of Tokenization {#understanding-tokenization}

As we conclude our exploration of tokenization in LLMs, it’s crucial to recognize the impact it has on the practical application of these models. Despite its dry and intricate nature, tokenization is a fundamental stage in the pipeline of LLMs that we simply cannot afford to ignore due to the plethora of issues and "footguns" it presents, including security and AI safety.

Tokenization is not merely a preprocessing step; it's the bedrock upon which the reliability and safety of LLMs are built. As we've seen, seemingly innocuous tokens can trigger unexpected and often undesirable behavior in models, leading to a host of problems ranging from misinterpretation to security vulnerabilities.

#### Tokenization's Lasting Challenges {#tokenization-challenges}

Tokenization presents several challenges that are yet to be fully overcome:

- The presence of "unallocated memory" in the form of untrained tokens can lead to unpredictable model behavior.  
- The need for efficient tokenization is not only a computational issue but also has significant cost implications.

The ultimate goal, then, is to create a tokenization system that is as efficient and error-free as possible. Those who can improve upon the current state of tokenization in language models will be making a monumental contribution to the field.

#### Reusing Tokens & Vocabulary {#reusing-tokens-vocabulary}

In practical applications, reusing existing tokens and vocabularies from models like GPT-4 can be a smart move. The [`tokenizers`](https://github.com/huggingface/tokenizers) library, for example, provides efficient and reliable tools for inference with the Byte Pair Encoding (BPE) algorithm, which is already well-optimized for use with LLMs.

However, if you find yourself in a position where you must train a new vocabulary from scratch, it’s advisable to use BPE with a tool like `sentencepiece`. While this approach has its own set of complexities and potential pitfalls, it remains a viable option for those who need a custom solution.

#### Navigating BPE and SentencePiece {#navigating-bpe-sentencepiece}

When dealing with BPE and `sentencepiece`, you must tread carefully. The myriad of settings and hyperparameters can easily become overwhelming, leading to errors that could truncate or otherwise distort your sentences. It is imperative to:

- Ensure you understand each parameter and its impact.  
- Refer to trusted configurations, such as those used in well-regarded models.  
- Spend time examining the hyperparameters and the underlying code to ensure you have configured everything correctly.

#### Exploring the Encoder {#exploring-encoder}

Let’s delve deeper into the tokenization process by examining the GPT-2 encoder. The following Python code snippet demonstrates how to download the essential files and load them into an `Encoder` class:

```python  
# Download the vocab and encoder files for GPT-2  
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe  
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

import os, json

with open('encoder.json', 'r') as f:  
    encoder = json.load(f)  # This is roughly equivalent to our vocabulary

with open('vocab.bpe', 'r', encoding='utf-8') as f:  
    bpe_data = f.read()  
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]  
    # This is roughly equivalent to our merges

# We can then use these in our Encoder class  
```

The `Encoder` class uses the loaded vocabulary and merge rules to tokenize and detokenize text as needed, implementing the core functionality of the BPE algorithm.

#### Visualizing Tokenization in Action {#visualizing-tokenization}

To better understand this process, let's use a Jupyter Notebook to visualize how tokenization works. The following code snippet uses a regular expression pattern, `gpt2pat`, to tokenize an example string:

```python  
import re

# Example regular expression pattern for tokenizing with GPT-2  
gpt2pat = ...

# Sample code fragment for tokenization visualization  
example = """  
for i in range(1, 101):  
    if i % 3 == 0 and i % 5 == 0:  
        print("FizzBuzz")  
"""

# Visualize the tokenization of the example  
print(re.findall(gpt2pat, example))  
```

Through this visualization, we gain insights into how different parts of the code are segmented into tokens, which can then be mapped to the model's embedding table for further processing.

#### The Encoder Class Implementation {#encoder-class-implementation}

A simplified version of an `Encoder` class, which encapsulates the BPE tokenization and detokenization logic, might look like this:

```python  
class Encoder:

    def bpe(self, token):  
        # BPE tokenization logic here...

    def encode(self, txt):  
        # Encoding logic here...

    def decode(self, tokens):  
        # Decoding logic here...

    @staticmethod  
    def get_encoder(model_name, models_dir):  
        # Logic to load the encoder and BPE data...  
```

Each method in the `Encoder` class has a specific role in managing the conversion between text and tokens, ensuring efficient communication between the language model and the human-readable text.

### Conclusion: The Ongoing Tokenization Journey {#tokenization-journey-conclusion}

Tokenization remains a key area of research and development in the quest for more advanced and efficient LLMs. As we've seen, it's not just about converting strings to tokens; it's about security, efficiency, and the seamless operation of the entire model.

For now, we recommend that developers leverage existing tokens and vocabularies wherever possible, and approach the creation of new vocabularies with caution. As for the future, we eagerly anticipate advancements that will make tokenization even more robust and streamlined.

And with that, we reach the end of our deep dive into the complexities of tokenization. We've covered a lot of ground, from the basics to the intricate details of tokenization processes and their implications. As the field continues to evolve, we may return to this topic for an even more detailed exploration. But for now, we leave you with a better understanding of the critical role tokenization plays in the world of LLMs.

Thank you for joining us on this journey through the intricacies of LLM tokenization. It's been a complex, sometimes dry, but undoubtedly essential topic. Remember, the path to improved language models is paved with the bits and bytes of well-crafted tokens!

END ARTICLE
