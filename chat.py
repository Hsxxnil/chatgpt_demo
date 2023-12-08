from openai import OpenAI

api_key = ""
organization = ""
client = OpenAI(
    api_key=api_key,
    organization=organization,
)

if __name__ == '__main__':
    # Enter your OpenAI API key and organization ID
    api_key = input("Enter your OpenAI API key: ")
    organization = input("Enter your OpenAI organization ID: ")

    # Update the client instance with new API key and organization ID
    client = OpenAI(
        api_key=api_key,
        organization=organization,
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": user_input}
            ]
        )

        model_response = completion.choices[0].message.content
        print("AI:", model_response)
