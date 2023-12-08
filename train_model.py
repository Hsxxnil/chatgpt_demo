from openai import OpenAI
import time
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
    logging.basicConfig(filename='train.log', level=logging.INFO,
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
    logger = configure_logging()

    # Enter your OpenAI API key and organization ID
    api_key = input("Enter your OpenAI API key: ")
    organization = input("Enter your OpenAI organization ID: ")

    # Update the client instance with new API key and organization ID
    client = OpenAI(
        api_key=api_key,
        organization=organization,
    )

    file_name = "train_data_example.jsonl"
    uploaded_file = upload_file(file_name)

    logger.info(uploaded_file)
    job = client.fine_tuning.jobs.create(training_file=uploaded_file.id, model="gpt-3.5-turbo-0613")
    print(f"Job created with id: {job.id}")
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)