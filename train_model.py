from datetime import datetime, timezone

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
    logging.basicConfig(filename=f"{file_path}/train.log", level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s")
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


if __name__ == "__main__":
    # Configure logger
    log_file_name = input("Please input the train log file path (e.g. 230101_v1): ")
    logger = configure_logging(log_file_name)

    # Upload train file
    train_file_name = input("Please input the train file path (e.g. 231208_v1/train_data.jsonl): ")
    uploaded_train_file = upload_file(train_file_name)
    logger.info(f"Uploaded train file with id: {uploaded_train_file.id}")

    # Upload valid file
    valid_file_name = input("Please input the validation file path (e.g. 230101_v1/valid_data.jsonl): ")
    uploaded_valid_file = upload_file(valid_file_name)
    logger.info(f"Uploaded valid file with id: {uploaded_valid_file.id}")

    # choose model
    model_type = input("Please input the model type (1:custom; 2:pretrained): ")
    if model_type == "1":
        # list all the models fine-tuned.
        result = client.fine_tuning.jobs.list(limit=10)
        result_list = []
        for item in result:
            created_at_utc = datetime.fromtimestamp(item.created_at, tz=timezone.utc)
            created_at_formatted = created_at_utc.strftime('%Y-%m-%d %H:%M:%S')
            fine_tuned_model_info = {
                "model_name": item.fine_tuned_model,
                "created_at": created_at_formatted,
            }
            result_list.append(fine_tuned_model_info)

        for model_info in result_list:
            print(f"Fine-tuning model >>> {model_info}")

        # choose model
        train_model = input("Please input fine_tuning_model_id: ")

    else:
        train_model = "gpt-3.5-turbo-0613"

    # Create a fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=uploaded_train_file.id,
        validation_file=uploaded_valid_file.id,
        model=train_model
    )
    print(f"Model ID >>> {train_model}")
    logger.info(f"Model ID >>> {train_model}")
    print(f"Job created with id: {job.id}")
    logger.info(f"Job created with id: {job.id}")
