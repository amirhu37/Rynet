// Loss functions, also known as cost functions or objective functions, measure how well a model's predictions match the actual target values. They are crucial for training machine learning models, as they provide the feedback signal used to adjust the model's parameters during optimization.

/// ### 1. **Mean Squared Error Loss (MSELoss)**
/// Measures the average squared difference between predicted and actual values. Commonly used in regression tasks.
/// ```
/// MSELoss(reduction='mean')
/// ```
/// #### Parameters:
/// - `reduction`: Specifies the reduction to apply to the output. Options are `'none'`, `'mean'`, and `'sum'`.
/// Example:
/// ```
/// criterion = MSELoss()
/// loss = criterion(predictions, targets)
/// ```
struct MSELoss;
impl MSELoss {}

/// ### 2. **Cross Entropy Loss (CrossEntropyLoss)**
/// Combines `LogSoftmax` and `NLLLoss` in one single class. It is useful for classification tasks.
/// ```
/// CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduction='mean')
/// ```
/// #### Parameters:
/// - `weight`: A manual rescaling weight given to each class.
/// - `ignore_index`: Specifies a target value that is ignored.
/// - `reduction`: Specifies the reduction to apply to the output.
/// Example:
/// ```
/// criterion = CrossEntropyLoss()
/// loss = criterion(predictions, targets)
/// ```
struct CrossEntropyLoss;
impl CrossEntropyLoss {}

/// ### 3. **Binary Cross Entropy Loss (BCELoss)**
/// Applies to binary classification tasks.
/// #### Parameters:
/// ```
/// BCELoss(weight=None, size_average=None, reduction='mean')
/// ```
/// - `weight`: A manual rescaling weight given to each class.
/// - `reduction`: Specifies the reduction to apply to the output.
struct BCELoss;
impl BCELoss {}

/// ### 4. **Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)**
/// Combines a `Sigmoid` layer and the `BCELoss` in one single class.
/// #### Constructor:
/// ```
/// BCEWithLogitsLoss(weight=None, size_average=None, reduction='mean', pos_weight=None)
/// ```
/// #### Parameters:
/// - `weight`: A manual rescaling weight given to each class.
/// - `pos_weight`: A weight of positive examples.
/// #### Example:
/// ```
/// criterion = BCEWithLogitsLoss()
/// loss = criterion(predictions, targets)
/// ```
struct BCEWithLogitsLoss;
impl BCEWithLogitsLoss {}
/// ### 5. **Negative Log-Likelihood Loss (NLLLoss)**
/// Often used in classification problems involving a log-probability output.

/// #### Constructor:
/// ```
/// NLLLoss(weight=None, size_average=None, ignore_index=-100, reduction='mean')
/// ```

/// #### Parameters:
/// - `weight`: A manual rescaling weight given to each class.
/// - `ignore_index`: Specifies a target value that is ignored.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = NLLLoss()
/// loss = criterion(predictions, targets)
/// ```
struct NLLLoss;
impl NLLLoss {}
/// ### 6. **Kullback-Leibler Divergence Loss (KLDivLoss)**
/// Measures the Kullback-Leibler divergence between two distributions.

/// #### Constructor:
/// ```
/// KLDivLoss(size_average=None, reduction='batchmean', log_target=False)
/// ```

/// #### Parameters:
/// - `reduction`: Specifies the reduction to apply to the output.
/// - `log_target`: Specifies whether `target` is passed in the log space.

/// #### Example:
/// ```
/// criterion = KLDivLoss()
/// loss = criterion(predictions, targets)
/// ```
struct KLDivLoss;
impl KLDivLoss {}
/// ### 7. **L1 Loss (L1Loss)**
/// Measures the mean absolute error between the predicted and actual values.

/// #### Constructor:
/// ```
/// L1Loss(size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = L1Loss()
/// loss = criterion(predictions, targets)
/// ```
struct L1Loss;
impl L1Loss {}
/// ### 8. **Hinge Embedding Loss (HingeEmbeddingLoss)**
/// Used for learning embeddings.

/// #### Constructor:
/// ```
/// HingeEmbeddingLoss(margin=1.0, size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `margin`: Margin for the loss.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = HingeEmbeddingLoss()
/// loss = criterion(predictions, targets)
/// ```
struct HingeEmbeddingLoss;
impl HingeEmbeddingLoss {}
/// ### 9. **Huber Loss (SmoothL1Loss)**
/// Combines the advantages of L1 and L2 loss functions, used for regression tasks.

/// #### Constructor:
/// ```
/// SmoothL1Loss(size_average=None, reduction='mean', beta=1.0)
/// ```

/// #### Parameters:
/// - `beta`: The threshold at which to change between L1 and L2 loss.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = SmoothL1Loss()
/// loss = criterion(predictions, targets)
/// ```
struct SmoothL1Loss;
impl SmoothL1Loss {}
/// ### 10. **Cosine Embedding Loss (CosineEmbeddingLoss)**
/// Measures the loss given inputs `x1`, `x2`, and a label `y` with values 1 or -1.

/// #### Constructor:
/// ```
/// CosineEmbeddingLoss(margin=0.0, size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `margin`: Margin for the loss.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = CosineEmbeddingLoss()
/// loss = criterion(x1, x2, targets)
/// ```
struct CosineEmbeddingLoss;
impl CosineEmbeddingLoss {}
/// ### 11. **Margin Ranking Loss (MarginRankingLoss)**
/// Creates a criterion that measures the loss given inputs `x1`, `x2`, and a label tensor `y` with values 1 or -1.

/// #### Constructor:
/// ```
/// MarginRankingLoss(margin=0.0, size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `margin`: Margin for the loss.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = MarginRankingLoss()
/// loss = criterion(x1, x2, targets)
/// ```
struct MarginRankingLoss;
impl MarginRankingLoss {}
/// ### 12. **Multi-label Margin Loss (MultiLabelMarginLoss)**
/// Measures the loss given inputs `x` and multi-label targets `y`.

/// #### Constructor:
/// ```
/// MultiLabelMarginLoss(size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = MultiLabelMarginLoss()
/// loss = criterion(predictions, targets)
/// ```
struct MultiLabelMarginLoss;
impl MultiLabelMarginLoss {}
/// ### 13. **Multi-label Soft Margin Loss (MultiLabelSoftMarginLoss)**
/// Measures the loss for multi-label classification tasks.

/// #### Constructor:
/// ```
/// MultiLabelSoftMarginLoss(weight=None, size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `weight`: A manual rescaling weight given to each class.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = MultiLabelSoftMarginLoss()
/// loss = criterion(predictions, targets)
/// ```
struct MultiLabelSoftMarginLoss;
impl MultiLabelSoftMarginLoss {}
/// ### 14. **Soft Margin Loss (SoftMarginLoss)**
/// Measures the binary cross entropy between input logits and target labels.

/// #### Constructor:
/// ```
/// SoftMarginLoss(size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = SoftMarginLoss()
/// loss = criterion(predictions, targets)
/// ```
struct SoftMarginLoss;
impl SoftMarginLoss {}
/// ### 15. **Triplet Margin Loss (TripletMarginLoss)**
/// Measures the triplet loss, used for training embeddings.

/// #### Constructor:
/// ```
/// TripletMarginLoss(margin=1.0, p=2, eps=1e-6, swap=False, size_average=None, reduction='mean')
/// ```

/// #### Parameters:
/// - `margin`: Margin for the triplet loss.
/// - `p`: The norm degree for pairwise distance.
/// - `eps`: Small value to avoid division by zero.
/// - `swap`: Whether to use the negative swap for the loss calculation.
/// - `reduction`: Specifies the reduction to apply to the output.

/// #### Example:
/// ```
/// criterion = TripletMarginLoss()
/// loss = criterion(anchor, positive, negative)
/// ```
struct TripletMarginLoss;
impl TripletMarginLoss {}

// ### Common Methods for Loss Functions
// - **`forward(input, target)`**: Computes the loss between the input and the target. This method is called internally when you use `criterion(input, target)`.
// - **`__call__(input, target)`**: Calls the `forward` method and returns the loss.
