//
//  Neuron.swift
//  neural_net
//
//  Created by Richard Dzubko on 12.11.2024.
//

import Foundation

public class Neuron {
    var inputs: [Double]
    var weights: [Double]
    var bias: Double
    var activationFunc: (Double) -> Double
    
    private init(inputs: [Double], weights: [Double], bias: Double, activationFunc: @escaping (Double) -> Double) {
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activationFunc = activationFunc
    }
    
    init(size: Int, activationFunc: @escaping (Double) -> Double) {
        inputs = Array(repeating: 0, count: size)
        weights = generateWeights(size)
        self.activationFunc = activationFunc
        bias = Double.random(in: 0..<1)
    }
    
    func update(inputs: [Double]? = nil, weights: [Double]? = nil, bias: Double? = nil) {
        self.inputs = inputs ?? self.inputs
        self.weights = weights ?? self.weights
        self.bias = bias ?? self.bias
    }
    
    func sum() -> Double {
        return bias + zip(inputs, weights).reduce(0) {
            return $0 + ($1.0 * $1.1)
        }
    }
    
    func activate() -> Double {
        return activationFunc(sum())
    }
}

fileprivate func generateWeights(_ size: Int) -> [Double] {
    var generatedWeights: [Double] = Array(repeating: 0, count: size)
    for i in 0..<size {
        generatedWeights[i] = Double.random(in: 0..<1)
    }
    
    return generatedWeights
}
