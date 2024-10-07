class Calculator:
    def __init__(self):
        self.current_value = 0
        self.operator = None

    def calculate(self, value):
        if self.operator == '+':
            self.current_value += value
        elif self.operator == '-':
            self.current_value -= value
        elif self.operator == '*':
            self.current_value *= value
        elif self.operator == '/':
            if value == 0:
                print("Error: Division by zero!")
                return None
            self.current_value /= value
        else:
            print("Error: Unknown operator!")
            return None
        return self.current_value

    def input_command(self, command):
        parts = command.split()
        if len(parts) != 2:
            print("Error: Invalid command format!")
            return None

        operator, value_str = parts
        if operator not in ['+', '-', '*', '/']:
            print("Error: Invalid operator!")
            return None

        try:
            value = float(value_str)
        except ValueError:
            print("Error: Invalid number!")
            return None

        self.operator = operator
        result = self.calculate(value)
        if result is not None:
            print(f"Result: {result}")
        return result


def main():
    calc = Calculator()
    print("Welcome to the Command Line Calculator!")
    print("Enter commands in the format: <operator> <number>")
    print("Example: + 5")
    print("Type 'exit' to quit.")

    while True:
        command = input("Enter command: ")
        if command.lower() == 'exit':
            break
        calc.input_command(command)


if __name__ == "__main__":
    main()