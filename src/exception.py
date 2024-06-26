import sys
import logging

# Assuming logging is imported correctly from src.logger.logging

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"\nError occurred in Python script: {file_name},\n line number: {line_number},\n error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
