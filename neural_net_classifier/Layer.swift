//
//  Layer.swift
//  neural_net_classifier
//
//  Created by Richard Dzubko on 12.11.2024.
//

class Layer {
    var neurons: [Neuron]
    var inputs: [Double]
    var outputs: [Double]
    let isOuterLayer: Bool
    
    init(size: Int, inputSize: Int, activationFunc: @escaping (Double) -> Double, isOuterLayer: Bool = false) {
        outputs = Array(repeating: 0, count: size)
        inputs = Array(repeating: 0, count: inputSize)
        neurons = []
        self.isOuterLayer = isOuterLayer
        for _ in 0..<size {
            neurons.append(Neuron(size: inputSize, activationFunc: activationFunc))
        }
    }
    
    private func Compute() {
        var outputs = [Double]()
        for neuron in neurons {
            neuron.update(inputs: inputs)
            outputs.append(neuron.activate())
        }
        self.outputs = outputs
    }
    
    private func normalizedCompute() {
        var outputs = [Double]()
        for neuron in neurons {
            neuron.update(inputs: inputs)
            outputs.append(neuron.sum())
        }
        
        outputs = ActivationFunctions.softmax(outputs)
        
        self.outputs = outputs
    }

    func getOutput() -> [Double] {
        if isOuterLayer {
            normalizedCompute()
        } else {
            Compute()
        }
        
        return outputs
    }
    
    func setInput(_ inputs: [Double]) {
        self.inputs = inputs
    }
}
