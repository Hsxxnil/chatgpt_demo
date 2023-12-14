from openai import OpenAI
import json
import logging
from config import config

client = OpenAI(
    api_key=config.openai_api_key,
    organization=config.openai_organization_id
)


def configure_logging(file_path):
    """
    Configures logging settings.
    """
    logging.basicConfig(filename=f'{file_path}/test.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def read_messages_from_jsonl(filename):
    """
    Read messages from a .jsonl file.

    :param filename: Path to the .jsonl file.
    :return: A list of messages.
    """
    messages_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            messages_list.append(entry["messages"])
    return messages_list


def test_by_case(model_id, message):
    """
    Test the model with a given message.

    :param model_id: ID of the model to test.
    :param message: Message to test the model.
    :return: Response from the model.
    """
    completion = client.chat.completions.create(
        model=model_id,
        messages=message
    )
    return completion.choices[0].message.content


def test(model_id, file_name):
    """
    Test the model using the messages from a given file.

    :param model_id: ID of the model to test.
    :param file_name: Path to the .jsonl file containing messages.
    """
    messages_list = read_messages_from_jsonl(file_name)
    for message in messages_list:
        system_message = next((msg["content"] for msg in message if msg["role"] == "system"), None)
        user_question = next((msg["content"] for msg in message if msg["role"] == "user"), None)
        result = test_by_case(model_id, message)
        logger.info(f'System >>> {system_message}')
        logger.info(f'User: {user_question}')
        logger.info(f'GPT: {result}')
        logger.info('-------------------------')


if __name__ == '__main__':
    # Configure logger
    log_file_path = input("Please input the test log file path (e.g. 230101_v1): ")
    logger = configure_logging(log_file_path)

    # Retrieve the model ID
    fine_tuning_job_id = input("Please input fine_tuning_job_id: ")
    train_result = client.fine_tuning.jobs.retrieve(
        fine_tuning_job_id=fine_tuning_job_id
    )
    print(f"Model ID >>> {train_result.fine_tuned_model}")
    logger.info(f"Model ID >>> {train_result.fine_tuned_model}")
    logger.info('-------------------------')

    # test the model
    test_type = input("Please input test type (1: test by chat case; 2: test by file): ")
    if test_type == "1":
        message = input("User: ")
        logger.info(f'User: {message}')
        result = test_by_case(train_result.fine_tuned_model, message)
        print(f"GPT: {result}")
        logger.info(f"GPT: {result}")
        logger.info('-------------------------')
    else:
        model_id = train_result.fine_tuned_model
        file_name = input("Please input the validation file path (e.g. 230101_v1/test_data.jsonl): ")
        test(model_id, file_name)

    print("Test completed.")
