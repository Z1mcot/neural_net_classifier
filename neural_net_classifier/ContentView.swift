//
//  ContentView.swift
//  neural_net
//
//  Created by Richard Dzubko on 12.11.2024.
//

import SwiftUI

func scaleVector(_ input: String) -> String {
    // Ensure the input is not empty
    guard !input.isEmpty else { return "" }
    
    // Calculate the scaling factor
    let inputLength = input.count
    let scalingFactor = 9.0 / Double(inputLength)
    
    // Create the scaled vector
    var scaledVector = ""
    for char in input {
        // Determine the number of repetitions for each character
        let repeatCount = Int(round(scalingFactor))
        scaledVector += String(repeating: char, count: repeatCount)
    }
    
    // Adjust to exactly 9 characters if there's any rounding mismatch
    while scaledVector.count > 9 {
        scaledVector.removeLast()
    }
    while scaledVector.count < 9 {
        scaledVector.append(input.last ?? "-")
    }
    
    return scaledVector
}

struct ContentView: View {
    @State private var viewModel = ViewModel()
    
    var body: some View {
        VStack {
            Text("Классификация")
                .font(.title)
                .bold()
            Text(viewModel.neuralOutput)
                .font(.largeTitle)
                .frame(height: 250)
                .padding(.horizontal, 60)
                .background(Color.gray.opacity(0.1))
            TextField("Данные для обучения", text: $viewModel.trainData)
                .padding(.vertical, 10)
            HStack {
                Button(action: viewModel.setTrain) {
                    Label("Обучить", systemImage: "book")
                        .padding(.all)
                }
                Button(action: viewModel.startTrain) {
                    Label("Классификация", systemImage: "arrow.2.circlepath")
                        .padding(.all)
                }
            }
        }
        .frame(width: 400, height: 480)
        .padding()
    }
}

#Preview {
    ContentView()
}

//MARK: - ViewModel

extension ContentView {
    @Observable
    class ViewModel {
        var trainData: String = ""
        private(set) var neuralOutput: String = "Начните обучение"
        
        private let network = Network(inputSize: 9, networkSize: 2, layerSize: [10, 3], activationFuncs: [
            ActivationFunctions.sigmoidFunction,
            ActivationFunctions.sigmoidFunction,
        ])
        
        func setTrain() {
            neuralOutput = "Обучение завершено!"
            let inputFile = Bundle.main.path(forResource: "trainData", ofType: "txt")!
            network.train(filePath: inputFile, epochs: 200, step: 0.85)
        }
        
        func startTrain() {
            let data = scaleVector(
                trainData
            )
            
            let input: [Double] = data.map {
                convertSymbol(symbol: $0)
            }

            network.setInput(input: input);
            let result = network.getOutput();

            let max = result.max()!;
            let idx = result.firstIndex(of: max)!;
            
            neuralOutput = "Класс: \(convertToSymbol(idx))"
        }
    }
}
