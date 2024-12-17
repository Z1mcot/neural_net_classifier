//
//  Network.swift
//  neural_net_classifier
//
//  Created by Richard Dzubko on 12.11.2024.
//


import Foundation

class Network {
    // Fields
    private var layers: [Layer] = []
    private var input: [Double]
    private var output: [Double]
    
    // Initializer
    init(inputSize: Int, networkSize: Int, layerSize: [Int], activationFuncs: [(Double) -> Double]) {
        self.output = Array(repeating: 0.0, count: layerSize.last!)
        self.input = Array(repeating: 0.0, count: inputSize)
        layers.append(Layer(size: layerSize[0], inputSize: inputSize, activationFunc: activationFuncs[0]))
        
        for i in 1..<(networkSize - 1) {
            layers.append(Layer(size: layerSize[i], inputSize: layerSize[i - 1], activationFunc: activationFuncs[i]))
        }
        
        layers.append(Layer(size: layerSize[networkSize-1], inputSize: layerSize[networkSize-2], activationFunc: activationFuncs[networkSize-1], isOuterLayer: true))
    }
    
    // Set input values
    func setInput(input: [Double]) {
        self.input = input
    }
    
    // Compute network output
    private func compute() {
        var processedLayer = input
        
        for layer in layers {
            layer.setInput(processedLayer)
            processedLayer = layer.getOutput()
        }
        output = processedLayer
    }
    
    // Get network output
    func getOutput() -> [Double] {
        compute()
        return output
    }
    
    // Training function
    func train(filePath: String, epochs: Int, step: Double) {
        var loss = [Double](repeating: 0.0, count: epochs)
        var lastError = 0.0
        
        for epoch in 0..<epochs {
            if let fileReader = try? String(contentsOfFile: filePath, encoding: .utf8) {
                let lines = fileReader.split(separator: "\n")
                
                for line in lines {
                    let stringInput = line.dropLast().map {
                        convertSymbol(symbol: $0)
                    }
                    input = stringInput
                    
                    let resultIndex = convertSymbolToInt(symbol: line.last!)
                    let realOutput = getOutput()
                    
                    let desiredOutput = getWantedOutputs(index: resultIndex, size: output.count)
                    lastError = sparseCrossEntropy(target: resultIndex, actual: realOutput)
                    
                    var previousGradient = Array(repeating: 0.0, count: layers.last!.neurons.count)
                    
                    // Update gradients for the last layer
                    for (j, neuron) in layers.last!.neurons.enumerated() {
                        let error = realOutput[j] - desiredOutput[j]
                        previousGradient[j] = gradientForSigmoid(error: error, neuronResult: realOutput[j])
                        for k in 0..<neuron.weights.count {
                            neuron.weights[k] -= step * previousGradient[j] * neuron.inputs[k]
                        }
                        neuron.bias -= step * previousGradient.reduce(0, +)
                    }
                    
                    // Update gradients for the remaining layers
                    for l in (2...layers.count).reversed() {
                        var localGradient = Array(repeating: 0.0, count: layers[l - 2].neurons.count)
                        
                        for (m, neuron) in layers[l - 2].neurons.enumerated() {
                            var tempSum = 0.0
                            for n in 0..<previousGradient.count {
                                tempSum += previousGradient[n] * layers[l - 1].neurons[n].weights[m]
                            }
                            localGradient[m] = gradientForSigmoid(error: tempSum, neuronResult: neuron.activate())
                            for k in 0..<neuron.weights.count {
                                neuron.weights[k] -= step * localGradient[m] * neuron.inputs[k]
                            }
                            neuron.bias -= step * previousGradient.reduce(0, +)
                        }
                        previousGradient = localGradient
                    }
                }
                loss[epoch] = lastError
            }
        }
    }
    
    // Generate desired output array for training
    private func getWantedOutputs(index: Int, size: Int) -> [Double] {
        var output = Array(repeating: 0.0, count: size)
        output[index - 1] = 1.0
        return output
    }
    
    // Calculate sparse cross-entropy loss
    private func sparseCrossEntropy(target: Int, actual: [Double]) -> Double {
        return -log(actual[target - 1])
    }
    
    // Calculate gradient for sigmoid function
    private func gradientForSigmoid(error: Double, neuronResult: Double) -> Double {
        return error * neuronResult * (1 - neuronResult)
    }
}
