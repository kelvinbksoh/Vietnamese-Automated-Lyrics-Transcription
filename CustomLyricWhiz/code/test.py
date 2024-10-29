# -*- coding: utf-8 -*-
from vietnam_number import n2w
import re

def convert_numbers_in_string(input_string):
    """
    Extract numeric substrings from input, convert them to Vietnamese words,
    and replace them in the original string.
    """
    # Use regex to find all numeric sequences
    numeric_parts = re.findall(r'\d+', input_string)

    # Replace numeric sequences with their word equivalents
    for num in numeric_parts:
        word = n2w(num)  # Convert to Vietnamese words
        input_string = input_string.replace(num, word, 1)  # Replace only the first occurrence

    return input_string

# Example usage
mixed_string = "Tổng số tiền là 115205201211 VND, số dư 101 VND."
result = convert_numbers_in_string(mixed_string)
print(result)
