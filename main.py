from datetime import date
from distutils.log import Log
import logging
from structlog import get_logger, configure
import structlog
from structlog.stdlib import LoggerFactory
import openai

# configure logger
today = date.today().strftime("%m_%d_%Y")
logging.basicConfig(filename='log/history_{}'.format(today), encoding='utf-8', level=logging.DEBUG)
configure(
    processors=[
        structlog.processors.UnicodeEncoder(),
        structlog.processors.KeyValueRenderer(
            key_order=['event', 'model', 'prompt', 'data'],
        ),
    ], 
    logger_factory=LoggerFactory())
log = get_logger()

# configure openai client
openai.organization = ('org-UQAqGjWYD9fRg5PJebJV7gkt')
# TODO: store env variables elsewhere
openai.api_key = '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' 

def list_models():
    return openai.Model.list()

def completion(model, tokens, nucleus, prompt):
    response = openai.Completion.create(
        model=model,
        max_tokens=tokens,
        top_p=nucleus,
        temperature=1,
        prompt=prompt,
        echo=True,
    )
    log.info('Completion', model=model, prompt=prompt, data=response)
    return response["choices"][0]["text"]

def main():
    return

if __name__ == "__main__":
    main()