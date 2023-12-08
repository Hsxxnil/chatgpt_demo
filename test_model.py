from openai import OpenAI
import json
import logging

api_key = ""
organization = ""
client = OpenAI(
    api_key=api_key,
    organization=organization,
)


def configure_logging():
    """
    Configures logging settings.
    """
    logging.basicConfig(filename='test_output.log', level=logging.INFO,
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
    logger = configure_logging()

    # Enter your OpenAI API key and organization ID
    api_key = input("Enter your OpenAI API key: ")
    organization = input("Enter your OpenAI organization ID: ")

    # Update the client instance with new API key and organization ID
    client = OpenAI(
        api_key=api_key,
        organization=organization,
    )

    # Model ID will be obtained from the e-mail when the training is complete.
    # Alternatively, you can use the following code to get the model_id:
    fine_tuning_job_id = input("Please input fine_tuning_job_id: ")
    train_result = client.fine_tuning.jobs.retrieve(
        fine_tuning_job_id=fine_tuning_job_id
    )
    print(f"Model ID >>> {train_result.fine_tuned_model}")
    logger.info(f"Model ID >>> {train_result.fine_tuned_model}")
    logger.info('-------------------------')

    # test the model
    model_id = train_result.fine_tuned_model
    file_name = input("Please input the validation file name (e.g. validation.jsonl): ")
    test(model_id, file_name)
    print("Test completed.")
