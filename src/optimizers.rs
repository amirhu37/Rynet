// ### Common Methods for All Optimizers
// - **`step(closure=None)`**: Performs a single optimization step (parameter update). For some optimizers like `LBFGS`, `closure` is a callable that re-evaluates the model and returns
//  the loss.
// - **`zero_grad()`**: Sets the gradients of all optimized tensors to zero. This is typically called before the backward pass to prevent accumulation of gradients from multiple passes.

/// ### 1. **SGD (Stochastic Gradient Descent)**
/// SGD is one of the simplest and most commonly used optimizers. It updates the parameters using the gradient of the loss function.
/// #### Constructor:
/// ```
/// SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `momentum`: Momentum factor.
/// - `dampening`: Dampening for momentum.
/// - `weight_decay`: Weight decay (L2 penalty).
/// - `nesterov`: Enables Nesterov momentum.
/// #### Methods:
/// - `step()`: Performs a single optimization step (parameter update).
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.
struct SGD;
impl SGD {}

/// ### 2. **Adam (Adaptive Moment Estimation)**
/// Adam combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
/// #### Constructor:
/// ```
/// Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `betas`: Coefficients used for computing running averages of gradient and its square.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// - `weight_decay`: Weight decay (L2 penalty).
/// - `amsgrad`: Whether to use the AMSGrad variant of this algorithm.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct Adam;
impl Adam {}
/// ### 3. **RMSprop (Root Mean Square Propagation)**
/// RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton.
/// #### Constructor:
/// ```
/// RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `alpha`: Smoothing constant.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// - `weight_decay`: Weight decay (L2 penalty).
/// - `momentum`: Momentum factor.
/// - `centered`: If `True`, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct RMSprop;
impl RMSprop {}
/// ### 4. **Adagrad (Adaptive Gradient Algorithm)**
/// Adagrad adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters.
/// #### Constructor:
/// ```
/// Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `lr_decay`: Learning rate decay.
/// - `weight_decay`: Weight decay (L2 penalty).
/// - `eps`: Term added to the denominator to improve numerical stability.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct Adagrad;
impl Adagrad {}
/// ### 5. **Adadelta**
/// Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
/// #### Constructor:
/// ```
/// Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `rho`: Coefficient used for computing a running average of squared gradients.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// - `weight_decay`: Weight decay (L2 penalty).
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct Adadelta;
impl Adadelta {}
/// ### 6. **AdamW (Adam with Weight Decay Regularization)**
/// AdamW is a variant of Adam that decouples weight decay from the gradient update.
/// #### Constructor:
/// ```
/// AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `betas`: Coefficients used for computing running averages of gradient and its square.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// - `weight_decay`: Weight decay (L2 penalty).
/// - `amsgrad`: Whether to use the AMSGrad variant of this algorithm.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct AdamW;
impl AdamW {}
/// ### 7. **SparseAdam**
/// SparseAdam is a variant of Adam optimized for sparse tensors.
/// #### Constructor:
/// ```
/// SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `betas`: Coefficients used for computing running averages of gradient and its square.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct SparseAdam;
impl SparseAdam {}
/// ### 8. **Adamax**
/// Adamax is a variant of Adam based on the infinity norm.
/// #### Constructor:
/// ```
/// Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `betas`: Coefficients used for computing running averages of gradient and its square.
/// - `eps`: Term added to the denominator to improve numerical stability.
/// - `weight_decay`: Weight decay (L2 penalty).
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct Adamax;
impl Adamax {}
/// ### 9. **LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)**
/// LBFGS is an optimization algorithm in the family of quasi-Newton methods.
/// #### Constructor:
/// ```
/// LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `max_iter`: Maximum number of iterations per optimization step.
/// - `max_eval`: Maximum number of function evaluations per optimization step.
/// - `tolerance_grad`: Termination tolerance on first order optimality.
/// - `tolerance_change`: Termination tolerance on function value/parameter changes.
/// - `history_size`: Update history size.
/// - `line_search_fn`: Line search algorithm to use.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct LBFGS;
impl LBFGS {}
/// ### 10. **ASGD (Averaged Stochastic Gradient Descent)**
/// ASGD averages the parameters over time, which is effective in reducing variance.
/// #### Constructor:
/// ```
/// ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `lambd`: Decay term.
/// - `alpha`: Power for averaging.
/// - `t0`: Point at which to start averaging.
/// - `weight_decay`: Weight decay (L2 penalty).
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.

struct ASGD;
impl ASGD {}
/// ### 11. **Rprop (Resilient Backpropagation)**
/// Rprop is a gradient-based optimization method that only uses the sign of the gradient.
/// #### Constructor:
/// ```
/// Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50))
/// ```
/// #### Parameters:
/// - `params`: Iterable of parameters to optimize.
/// - `lr`: Learning rate.
/// - `etas`: Multiplicative increase and decrease factors.
/// - `step_sizes`: Minimum and maximum step sizes.
/// #### Methods:
/// - `step()`: Performs a single optimization step.
/// - `zero_grad()`: Sets the gradients of all optimized tensors to zero.
struct Rprop;
impl Rprop {}
