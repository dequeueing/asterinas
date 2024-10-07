# main.py

from calculator import SimpleCalculator
from utils import print_welcome_message

def main():
    print_welcome_message()
    
    calc = SimpleCalculator()
    
    while True:
        print("\nOptions:")
        print("Enter 'add' to add two numbers")
        print("Enter 'subtract' to subtract two numbers")
        print("Enter 'multiply' to multiply two numbers")
        print("Enter 'divide' to divide two numbers")
        print("Enter 'quit' to end the program")
        user_input = input(": ")

        if user_input == "quit":
            print("Thank you for using the Simple Calculator. Goodbye!")
            break
        elif user_input in ['add', 'subtract', 'multiply', 'divide']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))

            if user_input == 'add':
                print("The result is:", calc.add(num1, num2))
            elif user_input == 'subtract':
                print("The result is:", calc.subtract(num1, num2))
            elif user_input == 'multiply':
                print("The result is:", calc.multiply(num1, num2))
            elif user_input == 'divide':
                print("The result is:", calc.divide(num1, num2))
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
