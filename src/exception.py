import sys
from src.loggers import logging
def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message
    
    
    
class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_details)
            
    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        1/0
    except Exception as e:
        logging.info("Logging has started")
        logging.error("Exception: %s",str(CustomException(e,sys)))
        raise CustomException(e,sys)
    
