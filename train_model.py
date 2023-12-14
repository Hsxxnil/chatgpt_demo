from openai import OpenAI
import time
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
    logging.basicConfig(filename=f'{file_path}/train.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def upload_file(file_name):
    """
    Uploads a file to OpenAI for fine-tuning.

    :param file_name: Path to the file to be uploaded.
    :return: Uploaded file object.
    """
    # Note: For a 400KB train_file, it takes about 1 minute to upload.
    file_upload = client.files.create(file=open(file_name, "rb"), purpose="fine-tune")
    print(f"Uploaded file with id: {file_upload.id}")
    logger.info(f"Uploaded file with id: {file_upload.id}")

    while True:
        print("Waiting for file to process...")
        logger.info("Waiting for file to process...")
        file_handle = client.files.retrieve(file_id=file_upload.id)

        if file_handle.status == "processed":
            print("File processed")
            logger.info("File processed")
            break
        time.sleep(60)

    return file_upload


if __name__ == '__main__':
    # Configure logger
    log_file_name = input("Please input the train log file path (e.g. 230101_v1): ")
    logger = configure_logging(log_file_name)

    # Upload train file
    train_file_name = input("Please input the train file path (e.g. 231208_v1_success/train_data.jsonl): ")
    uploaded_train_file = upload_file(train_file_name)
    logger.info(f"Uploaded train file with id: {uploaded_train_file.id}")

    # Upload valid file
    valid_file_name = input("Please input the validation file path (e.g. 230101_v1/valid_data.jsonl): ")
    uploaded_valid_file = upload_file(valid_file_name)
    logger.info(f"Uploaded valid file with id: {uploaded_valid_file.id}")

    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=uploaded_train_file.id,
        validation_file=uploaded_valid_file.id,
        model="gpt-3.5-turbo-0613"
    )
    print(f"Job created with id: {job.id}")
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)
