import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('running.log')

# Set levels for handlers
console_handler.setLevel(logging.WARNING)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# example usages:
# logger.debug('This is a debug message')
# logger.info('Informational message')
# logger.warning('Warning occurred')
# logger.error('An error happened')
# logger.critical('Critical issue')
