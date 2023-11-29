#include <string>
#include <cctype>
#include <iostream>

int main() {
    std::string signal = "A200V100";
    std::string temp;
    std::string firstNumberStr, secondNumberStr;

    for (char c : signal) {
        if (std::isdigit(c)) {
            temp += c;
        } else {
            if (!temp.empty()) {
                if (firstNumberStr.empty()) {
                    firstNumberStr = temp;
                } else if (secondNumberStr.empty()) {
                    secondNumberStr = temp;
                }
                temp.clear();
            }
        }
    }
    if (!temp.empty() && secondNumberStr.empty()) {
        secondNumberStr = temp;
    }

    int firstNumber = std::stoi(firstNumberStr);
    int secondNumber = std::stoi(secondNumberStr);

    std::cout << "Angle: " << firstNumber << "\n";
    std::cout << "Velocity: " << secondNumber << "\n";

    return 0;
}