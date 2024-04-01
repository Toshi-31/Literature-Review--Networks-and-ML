Problem statement:
The problem statement revolves around the limitations of traditional machine learning workflows in the field of network security. Specifically, it highlights challenges related to generalization, dataset quality, feature selection, and manual labeling. The existing workflow relies heavily on high-quality labeled data and manual feature engineering, leading to difficulties in capturing crucial relationships and generalizing effectively. These challenges hinder the performance of machine learning models in complex production settings, such as traffic classification, network intrusion detection, and APT detection. The proposed solution, netFound, aims to address these limitations by leveraging self-supervised pre-training on unlabeled network packet traces to capture hierarchical and multi-modal attributes of network traffic, improving robustness and generalization across diverse network environments.

Challenges:
The challenges in developing network foundation models can be summarized as follows:

1. Consideration of Multi-modal Nature: Network data encompasses various contexts and perspectives, such as packet headers, payload, and network conditions. To effectively capture this multi-modal nature, foundation models must go beyond traditional transformer-based approaches and employ innovative input embedding methods to process diverse data modalities present in network traffic.

2. Handling Hierarchical Structure: Network data follows a hierarchical structure, organized into sequences of packets grouped into bursts, flows, services, hosts, etc. Developing models that can handle this hierarchical structure is crucial, as different network security learning problems make decisions at varying granularity levels within this hierarchy. Existing solutions often struggle to capture comprehensive information without resorting to sequence trimming, highlighting the need for modular foundation models that support easy extension to downstream tasks at different granularities while leveraging the hierarchical structure effectively.

Current Use of ML: 
Network Traffic Classification:
- Involves categorization or labeling of network traffic based on characteristics, patterns, or content.
- Commonly used for application identification such as web browsing, file sharing, etc.
- ML techniques treat this as a multi-class classification problem and train supervised classifiers from labeled training datasets.
- Effective even under various network communication setups (e.g., VPN, Tor) and when traffic is encrypted.
- Limitations include the need for well-labeled training datasets and potential challenges in handling encrypted traffic.

Network Intrusion/Anomaly Detection:
- Identifies and classifies network traffic indicating malicious activities or intrusion attempts.
- Can be coarse-grained (binary classification) or fine-grained (multi-class classification) based on the level of detail.
- Techniques include supervised classification and unsupervised learning to detect deviations from normal traffic patterns.
- Limitations include the need for labeled datasets for supervised learning and the challenge of detecting novel attacks.

Advanced Persistent Threats (APT) Detection:
- APTs infiltrate target systems and move laterally over long intervals to evade detection.
- Machine learning used to identify specific attacks involved in APTs and which hosts are compromised.
- Challenges include the complexity of APT attacks and the need for effective feature representation.

Other Network Security Applications:
- Includes botnet detection, application fingerprinting, vulnerability assessment, etc.
- Modeled as either supervised classification or unsupervised outlier detection problems.
- Limitations vary depending on the specific application but can include the need for labeled data and the challenge of detecting subtle anomalies.

Existing Techniques and Limitations:
- Current techniques rely heavily on supervised learning with labeled datasets.
- Challenges include the need for high-quality labeled data, the complexity of handling encrypted traffic, and the difficulty of detecting novel attacks.
- Some techniques explore unsupervised learning for anomaly detection but may struggle with false positives.

Existing techniques and methods:

Task-Specific Techniques:
- Techniques involve extracting features from network traffic data and training supervised models for specific tasks.
- Features are often categorized into aggregate flow statistics, raw bytes in packet headers or payloads, and time series features.
- Existing solutions vary in feature categories considered, feature extraction methods, and model specifications.
- Limitations include over-reliance on high-quality labeled datasets, which can lead to poor generalization and underspecification issues due to the scarcity of representative data.

Task-Agnostic Techniques:
- Develop foundation models to learn intermediate network data representations using unlabeled data, which can then be fine-tuned for downstream tasks.
- Solutions vary in feature categories, feature extraction methods, and representation learning models.
- Transformer-based foundation models leverage self-attention mechanisms and domain-specific learning objectives.
- Limitations include the inability to capture unique domain-specific attributes of network data, lack of consideration for the hierarchical structure of networking data, and focus solely on generating representations from a subset of raw bytes.

Takeaways:
- ML-based solutions offer potential benefits for various network security challenges.
- Task-specific techniques encounter issues with generalizability due to a scarcity of labeled data, while task-agnostic techniques are better suited for maximizing the utility of available network data.
- However, existing solutions neglect the intrinsic hierarchy and multi-modality present in network data, restricting their ability to balance performance, generalizability, and scalability effectively.

Design choices 

Selecting the Model Architecture:
- Choose the transformer architecture due to its ability to capture long-term dependencies, handle various input modalities, and support self-supervised learning.
- Transformers offer advantages over other architectures like autoencoders, including better adaptability to different datasets and reduced risk of overfitting.

Capturing Multi-modal Inputs:
- Designed to capture raw bytes by slicing packets into fixed-size tokens, each representing a specific portion of the packet payload.
- Considered two-byte tokens to reduce the model's vulnerability to underspecification issues and avoided learning shortcuts through specific packet content.
- Embedded time series, statistical features, and metadata directly with each token to enable the model to learn cross-modal dependencies.

Capturing Hierarchical Structures:
- Proposed a hierarchical transformer architecture to capture the inherent hierarchy in network data.
- Addressed limitations of naive hierarchical structures by introducing skip connections between layers to capture token dependencies within and across bursts effectively.
- Incorporated additional CLS tokens at each granularity level to obtain holistic representations for downstream tasks.

Overall:
- The model aims to learn latent representations for sequences of network packets grouped at different granularities using self-supervised learning methods.
- Design choices prioritize adaptability to diverse network environments, effective representation learning from multi-modal inputs, and capturing hierarchical structures for scalability and performance in downstream tasks.

netFound’s Workflow
1. Data Extraction:
   - Packet traces are split into smaller files, one for each flow.
   - Flows with 1-2 packets are discarded as they often represent noisy scanning activities.
   - Packets within a flow are categorized into "bursts" based on direction and inter-packet gap.

2. Standardization:
   - Number of packets per burst and bursts per flow are standardized to enable batch operations.
   - Balanced approach chosen: up to 12 bursts per flow and six packets per burst.

3. Featurization:
   - Various network, transport, and application layer packet fields are extracted for each packet in a burst.
   - Flow identifiers are eschewed to focus on understanding networking context.
   - Metadata fields like direction, bytes per burst, start time, etc., are also extracted.

4. Tokenization:
   - Packet-level feature vectors are tokenized.
   - Two-byte (16-bit) tokens chosen for balancing sequence length and vocabulary size.
   - Special tokens like [PAD], [CLS-B], [CLS-F], and [MASK] are introduced to enhance learning and facilitate batch operation.

Token Embedding
1. Token Embedding Importance:
   - Token embedding converts discrete token representations into continuous, differentiable ones, crucial for model training.
   - It enables the model to understand the semantic relationships between tokens in the input sequence, facilitating effective learning.

2. Types of Embedding: 
   - Three types of embedding considered: packet field, positional, and metadata.
   - Packet field embedding transforms token representations into continuous vectors using a learnable weight matrix.
   - Positional embedding adds information about the position of tokens in the sequence, both at the token and burst levels.
   - Metadata embedding transforms additional information about each token, like direction and burst statistics, into continuous vectors.

3. Embedding Computation: 
   - Token embedding uses a linear operation with a learnable weight matrix.
   - Positional embedding is added to tokens based on their position in the sequence.
   - Metadata embedding transforms additional information using a similar linear operation.
   - Final embedding for each token is computed as the sum of its token, positional, and metadata embeddings, providing a comprehensive representation for the model to understand the token's context within the sequence.

Pre-Training netFound
1. Transformer Architecture: 
   - The model employs a series of attention layers based on the transformer architecture, allowing it to capture both global and local correlations within input tokens.
   - Each attention layer consists of multiple attention heads, which compute attention weights based on key, query, and value representations of the input tokens, capturing correlations within the sequence.
   - Stacking multiple attention layers enables the model to capture various correlations and dependencies between input tokens, enhancing its understanding of the input data.

2. Hierarchical Transformers with Skip Connection: 
   - The model is structured hierarchically to capture dependencies at different granularity levels within packet traces.
   - Skip connections are introduced in the flow-level transformer to focus on learning inter-burst dependencies while reducing the complexity of processing ultra-long sequences.
   - This design enhances the model's capability to handle long input sequences, allowing it to process input with up to 1296 tokens and capture long-term dependencies within flows.

3. Self-Supervised Pre-training: 
   - The model is pre-trained in a self-supervised manner using masking, where a portion of tokens in each input sequence are randomly masked.
   - The model predicts the masked tokens using a classification layer stacked on top of the foundation model, minimizing token prediction errors via negative log-likelihood loss.
   - This pre-training objective helps the model learn meaningful representations of the input data in an unsupervised fashion, facilitating downstream tasks.

Fine Tuning
During fine-tuning, a shallow MLP model is added on top of the pre-trained netFound architecture for each task, utilizing either the [CLS-B] or [CLS-F] token output. Parameters connected to these tokens are updated, customizing representations for specific tasks while maintaining efficiency. While training only the shallow model is computationally simpler, updating the pre-trained model yields better performance on downstream tasks.

Evaluation of netFound’s Fine-tuned Models
Experimental Set-up
1. Experiment Setup: 
   - Three network security applications are considered: traffic classification, intrusion detection, and APT detection.
   - Datasets for each task are curated, including a campus traffic dataset, the CIC-IDS-2017 dataset, and an APT detection dataset from a Multi-Cloud environment.
   - Pre-trained netFound representations are used for each flow, followed by training shallow DNN classifiers for classification tasks.

2. Baselines: 
   - Comparison is made with task-agnostic methods nPrintML and ET-BERT, which generate flow representations from raw bytes.
   - Curtains, a state-of-the-art task-specific method, is also included as a baseline.

3. Evaluation Metrics:
   - Precision, recall, and F1 score are used as evaluation metrics.
   - Per-class metrics are computed using the sklearn package and then macro-averaged to obtain final metrics, penalizing models for predicting only the majority classes, which is beneficial for imbalanced datasets.

Result
1. Effectiveness on Downstream Tasks (Q1):
   - netFound outperforms Curtains, nPrintML, and ET-BERT on traffic classification, intrusion detection, and APT detection tasks.
   - Demonstrates superiority of learning-based representations over rule-based methods and effectiveness of customized foundation model designs.
2. Robustness against Missing Labels (Q2):
   - netFound maintains similar performance even with relatively high missing label rates (up to 40%), showcasing its robustness against missing labels.
3. Robustness against Label Noises (Q3):
   - netFound maintains decent performance even with up to 40% of training samples having noisy labels, indicating its ability to mitigate the effects of label errors.
   - Shows enhanced generalizability and informative representations learned by netFound.
4. Ablation Study (Q4):
   - netFound outperforms variants with removed metadata embedding and flow encoder, affirming their necessity in capturing hidden networking context effectively.
   - Pre-training significantly improves model performance and convergence, demonstrating the efficacy of pre-training in learning general correlations within the input data.
Discussion and Conclusion
In the Discussion, the focus shifts towards acknowledging the potential of extending netFound to handle heterogeneous data sources and addressing concerns about adversarial robustness. The authors outline future directions, including expanding the model's hierarchy, exploring alternative objective functions, and adapting it for online setups. Additionally, they propose collecting more data for pre-training and plan to conduct broader comparisons with existing methods. Overall, netFound presents a promising approach for deriving meaningful representations from networking data, with avenues for further enhancement and application in diverse network security scenarios.
