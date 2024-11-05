# Comparative Analysis of GloVe, ELMo, and BERT
# SRN: PES2UG22CS228
# NAME: ISHAAN NENE

<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    padding: 10px;
    text-align: left;
    border: 1px solid #ddd;
  }
  th {
    background-color: #333;
    color: white;
  }
  td.glove {
    background-color: #e6f7ff;
    color: #005073;
  }
  td.elmo {
    background-color: #fcefe3;
    color: #b36b00;
  }
  td.bert {
    background-color: #e8f5e9;
    color: #1b5e20;
  }
</style>

| Feature                  | <span class="glove">GloVe</span>                | <span class="elmo">ELMo</span>                | <span class="bert">BERT</span>                   |
|--------------------------|-------------------------------------------------|------------------------------------------------|---------------------------------------------------|
| **Type of Embedding**    | <span class="glove">Static Embeddings</span>    | <span class="elmo">Contextualized Embeddings</span> | <span class="bert">Contextualized Embeddings</span> |
| **Training Technique**   | <span class="glove">Trained on co-occurrence statistics of words in a corpus using matrix factorization (unsupervised)</span> | <span class="elmo">Trained using bidirectional LSTM to capture context-dependent word representations</span> | <span class="bert">Trained using Transformer-based masked language model, capturing bidirectional context</span> |
| **Input Format**         | <span class="glove">Pre-tokenized text corpus</span> | <span class="elmo">Tokenized text, processed through bidirectional LSTM layers</span> | <span class="bert">Tokenized text (subword tokens) for masked language modeling</span> |
| **Embedding Generation** | <span class="glove">Provides a single vector per word, independent of context</span> | <span class="elmo">Provides different vectors for each word based on surrounding context</span> | <span class="bert">Provides contextual embeddings based on sentence structure and position</span> |
| **Model Output**         | <span class="glove">Word embeddings (static, context-independent)</span> | <span class="elmo">Word embeddings (contextual, dynamic for each word instance)</span> | <span class="bert">Word embeddings (contextual, dynamic, sensitive to sentence structure)</span> |
| **Dimensionality**       | <span class="glove">Fixed dimensionality (e.g., 50, 100, 200, 300)</span> | <span class="elmo">Variable based on the LSTM layers used; typically large (e.g., 1024)</span> | <span class="bert">Typically large dimensionality; configurable but often 768 or 1024</span> |
| **Context Awareness**    | <span class="glove">No, provides the same vector regardless of sentence context</span> | <span class="elmo">Yes, embeddings vary based on the sentence context</span> | <span class="bert">Yes, embeddings vary based on sentence context and word position</span> |
| **Handling of Polysemy** | <span class="glove">Does not distinguish between multiple meanings of the same word</span> | <span class="elmo">Captures polysemy by providing different vectors for each word instance</span> | <span class="bert">Handles polysemy by providing unique embeddings based on the entire sentence context</span> |
| **Applications**         | <span class="glove">- Sentiment analysis <br> - Named entity recognition <br> - Information retrieval <br> - Text classification in resource-constrained environments</span> | <span class="elmo">- Question answering <br> - Coreference resolution <br> - Named entity recognition <br> - Semantic similarity tasks</span> | <span class="bert">- Text classification <br> - Machine translation <br> - Question answering <br> - Text summarization <br> - Conversational AI <br> - Named entity recognition</span> |
| **Advantages**           | <span class="glove">Efficient and interpretable, suitable for static tasks with large-scale corpora</span> | <span class="elmo">Captures rich semantic context for words, handling polysemy and nuanced meaning</span> | <span class="bert">Strong performance on diverse NLP tasks, highly flexible, leverages transfer learning</span> |
| **Limitations**          | <span class="glove">Cannot handle polysemy (same vector for all contexts)</span> | <span class="elmo">Higher computational cost compared to GloVe, context is limited by LSTM architecture</span> | <span class="bert">Computationally intensive, requires significant resources for fine-tuning</span> |
| **Common Usage Scenarios** | <span class="glove">Ideal for applications requiring fast processing with limited hardware, like search engines and basic NLP tasks</span> | <span class="elmo">Suitable for applications needing nuanced understanding, such as complex document processing and coreference resolution</span> | <span class="bert">Ideal for applications demanding complex language understanding, including chatbots, translation services, and question-answering systems</span> |
| **Implementation Complexity** | <span class="glove">Relatively simple; uses pre-trained embeddings and straightforward implementation</span> | <span class="elmo">Moderate; requires specialized handling for LSTM layers and may need additional configuration</span> | <span class="bert">High; Transformer models require GPU/TPU for training and careful handling for fine-tuning and deployment</span> |
| **Pretrained Models Available** | <span class="glove">Yes, available for various languages and dimensions (e.g., 50d, 100d, 300d)</span> | <span class="elmo">Yes, pretrained models are available, generally in English (like in the AllenNLP library)</span> | <span class="bert">Yes, widely available for many languages (BERT, multilingual BERT, and variants like RoBERTa, DistilBERT)</span> |
| **Fine-tuning Flexibility** | <span class="glove">Limited; embeddings are fixed and typically not trainable</span> | <span class="elmo">Somewhat flexible; can be fine-tuned with added layers for specific tasks</span> | <span class="bert">Highly flexible; easily fine-tuned for downstream tasks using labeled data</span> |
| **Popular Libraries**    | <span class="glove">Gensim, SpaCy, NLP4J</span> | <span class="elmo">AllenNLP, Flair</span> | <span class="bert">Hugging Face Transformers, TensorFlow, PyTorch</span> |
| **Real-World Examples**  | <span class="glove">- Google Search <br> - Basic sentiment analysis applications</span> | <span class="elmo">- Semantic search engines <br> - Document summarization <br> - Disambiguation in news articles</span> | <span class="bert">- Google BERT-based Search <br> - Language translation (DeepL) <br> - Virtual assistants (Google Assistant, Siri)</span> |

### Summary

Each model has unique strengths and limitations depending on the NLP task requirements:
- **GloVe** is a lightweight option suitable for basic NLP tasks where contextual understanding is not crucial.
- **ELMo** offers richer embeddings for applications where understanding context and multiple meanings of words is essential.
- **BERT** is the most versatile and powerful model for tasks needing deep language comprehension and context sensitivity, although it requires significant computational resources.
