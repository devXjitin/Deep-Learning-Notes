## Neural Networks

- Biological neuron vs artificial neuron
- Structure of an artificial neuron (inputs, weights, bias)
- Weighted sum and output calculation
- Single-layer neural network
- Multi-layer neural network intuition
- Role of layers in learning representations

---

# Biological Neuron vs Artificial Neuron

The concept of neural networks is inspired by the human brain. To understand artificial neurons, it is helpful to first understand how a biological neuron works in a simplified way.

A biological neuron is a cell in the brain that receives signals, processes them, and sends signals to other neurons. It has three main parts. **Dendrites** receive signals from other neurons. The **cell body** processes these incoming signals. The **axon** sends the output signal to the next neuron. If the combined input signals are strong enough, the neuron activates, or “fires,” and passes the signal forward.

An important idea is that a biological neuron does not respond to a single signal. It combines many incoming signals and then decides whether to fire or remain inactive. This decision-making behavior is what inspired the design of artificial neurons.

An artificial neuron is a mathematical model that imitates this behavior using numbers. Inputs represent incoming signals. Each input has an associated **weight**, which indicates the importance of that input. The neuron multiplies each input by its weight and then adds all the results together.

After combining the weighted inputs, a **bias** is added. The bias allows the neuron to adjust its activation threshold, similar to how a biological neuron requires a certain signal strength to fire. The final value is passed through an **activation function**, which decides the neuron’s output.

In simple terms, a biological neuron decides whether to fire based on electrical and chemical signals, while an artificial neuron decides whether to activate based on numerical calculations. Both follow the same basic idea: receive inputs, process them, and produce an output.

Biological neurons are extremely complex, adaptive, and energy-efficient. Artificial neurons are much simpler and capture only a small portion of real brain behavior. However, when millions of artificial neurons are connected together in layers, they can solve complex problems effectively.

**In summary**, artificial neurons are not exact replicas of biological neurons. They are inspired by biological neurons, simplified for computation, and designed to work efficiently on machines.

---