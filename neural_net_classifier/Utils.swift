//
//  Utils.swift
//  neural_net_classifier
//
//  Created by Richard Dzubko on 13.11.2024.
//

func convertSymbolToInt(symbol: String.Element) -> Int {
    switch symbol {
    case "*": return 1
    case "-": return 2
    case "/": return 3
    default:
        assert(false, "Invalid input")
        return 0
    }
}

func convertSymbol(symbol: String.Element) -> Double {
    switch symbol {
    case "*": return 1.0
    case "-": return 2.0
    case "/": return 3.0
    default:
        assert(false, "Invalid input")
        return 0.0
    }
}

func convertToSymbol(_ value: Int) -> String {
    switch value {
    case 0: return "*"
    case 1: return "-"
    case 2: return "/"
    default:
        assert(false, "Invalid input")
        return "()"
    }
}
