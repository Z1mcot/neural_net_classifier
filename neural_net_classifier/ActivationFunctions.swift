//
//  ActivationFunctions.swift
//  neural_net_classifier
//
//  Created by Richard Dzubko on 12.11.2024.
//


import Foundation

class ActivationFunctions {
    // Threshold function
    static func thresholdFunction(_ value: Double) -> Double {
        return value < 0 ? 0 : 1
    }

    // Signature function
    static func signatureFunction(_ value: Double) -> Double {
        return value > 0 ? 1 : -1
    }

    // Sigmoid function (logistic)
    static func sigmoidFunction(_ value: Double) -> Double {
        return 1 / (1 + exp(-value))
    }

    // Differential sigmoid function
    static func difSigmoidFunction(_ value: Double) -> Double {
        let sigmoid = sigmoidFunction(value)
        return sigmoid * (1 - sigmoid)
    }

    // Tangential function
    static func tangentialFunction(_ value: Double) -> Double {
        return 2 / (1 + exp(-2 * value)) - 1
    }

    // Half-linear function
    static func halfLinearFunction(_ value: Double) -> Double {
        return value > 0 ? value : 0
    }

    // Linear function
    static func linearFunction(_ value: Double) -> Double {
        return value
    }

    // Gauss function
    static func gaussFunction(_ value: Double) -> Double {
        return exp(-pow(value, 2))
    }
    
    static func softmax(_ values: [Double]) -> [Double] {
        let exponents = values.map {
            exp($0)
        }
        let divider = exponents.reduce(0, +)
        
        return exponents.map {
            $0 / divider
        }
    }
}
