def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       power,
                       staircase=False,
                       name=None):
  """Applies inverse time decay, with an additional polynomial argument p to the initial learning rate.
  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an inverse decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.
  The function returns the decayed learning rate.  It is computed as:
  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
  decay_step)
  ```
  or, if `staircase` is `True`, as:
  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step /
  decay_step))
  ```
  Example: decay 1/t with a rate of 0.5:
  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  decay_steps = 1.0
  decay_rate = 0.5
  learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
  decay_steps, decay_rate)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```
  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'InverseTimeDecay'.
  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for inverse_time_decay.")
  with ops.name_scope(name, "InverseTimeDecay",
                      [learning_rate, global_step, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    power = math_ops.cast(power, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    const = math_ops.cast(constant_op.constant(1), learning_rate.dtype)
    denom = math_ops.pow(math_ops.add(const, math_ops.multiply(decay_rate, p)),power)

    return math_ops.div(learning_rate, denom, name=name)
