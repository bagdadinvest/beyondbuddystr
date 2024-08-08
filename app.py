from dotenv import load_dotenv
import os
import base64
import re
import json
import streamlit as st
import openai
from openai import AssistantEventHandler
from tools import TOOL_MAP
from typing_extensions import override

load_dotenv()

# Helper function to convert string to boolean
def str_to_bool(str_input):
    if not isinstance(str_input, str):
        return False
    return str_input.lower() == "true"

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
enabled_file_upload_message = os.getenv("ENABLED_FILE_UPLOAD_MESSAGE", "Upload a file")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
authentication_required = str_to_bool(os.getenv("AUTHENTICATION_REQUIRED", "False"))

# Initialize the OpenAI client based on provided configurations
if azure_openai_endpoint and azure_openai_key:
    client = openai.AzureOpenAI(
        api_key=azure_openai_key,
        api_version="2024-05-01-preview",
        azure_endpoint=azure_openai_endpoint,
    )
else:
    client = openai.OpenAI(api_key=openai_api_key)

def format_annotation(text):
    """
    Formats the assistant's response text, handling annotations like links and citations.

    Args:
        text (str): The response text from the assistant.

    Returns:
        str: The formatted text without annotations and citations.
    """
    # Check if text has 'annotations' attribute
    if hasattr(text, "annotations"):
        # Process annotations
        for annotation in text.annotations:
            if annotation.type == "link":
                # Replace links with plain text or custom placeholder
                text.value = text.value.replace(annotation.text, f"[{annotation.text}]({annotation.url})")
            elif annotation.type == "citation":
                # Replace citations with plain text or custom placeholder
                text.value = text.value.replace(annotation.text, "")

    # Return cleaned text
    return text.value

# Event handler class for managing assistant events
class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        pass

    @override
    def on_text_created(self, text):
        st.session_state.current_message = ""
        with st.chat_message("Assistant"):
            st.session_state.current_markdown = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        if snapshot.value:
            text_value = re.sub(r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", "Download Link", snapshot.value)
            st.session_state.current_message = text_value
            st.session_state.current_markdown.markdown(st.session_state.current_message, True)

    @override
    def on_text_done(self, text):
        format_text = format_annotation(text)
        st.session_state.current_markdown.markdown(format_text, True)
        st.session_state.chat_log.append({"name": "assistant", "msg": format_text})

    @override
    def on_tool_call_created(self, tool_call):
        if tool_call.type == "code_interpreter":
            st.session_state.current_tool_input = ""
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

    @override
    def on_tool_call_delta(self, delta, snapshot):
        if 'current_tool_input_markdown' not in st.session_state:
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                st.session_state.current_tool_input += delta.code_interpreter.input
                input_code = f"### code interpreter\ninput:\npython\n{st.session_state.current_tool_input}\n"
                st.session_state.current_tool_input_markdown.markdown(input_code, True)

    @override
    def on_tool_call_done(self, tool_call):
        st.session_state.tool_calls.append(tool_call)
        if tool_call.type == "code_interpreter":
            if tool_call.id in [x.id for x in st.session_state.tool_calls if x.id == tool_call.id]:
                return
            input_code = f"### code interpreter\ninput:\npython\n{tool_call.code_interpreter.input}\n"
            st.session_state.current_tool_input_markdown.markdown(input_code, True)
            st.session_state.chat_log.append({"name": "assistant", "msg": input_code})
            st.session_state.current_tool_input_markdown = None
            
            for output in tool_call.code_interpreter.outputs:
                if output.type == "logs":
                    output_msg = f"### code interpreter\noutput:\n{output.logs}\n"
                    with st.chat_message("Assistant"):
                        st.markdown(output_msg, True)
                        st.session_state.chat_log.append({"name": "assistant", "msg": output_msg})
        elif (
            tool_call.type == "function"
            and self.current_run.status == "requires_action"  # Check if current_run is accessible
        ):
            with st.chat_message("Assistant"):
                msg = f"### Function Calling: {tool_call.function.name}"
                st.markdown(msg, True)
                st.session_state.chat_log.append({"name": "assistant", "msg": msg})
                
            tool_outputs = []
            for submit_tool_call in self.current_run.required_action.submit_tool_outputs.tool_calls:
                tool_function_name = submit_tool_call.function.name
                tool_function_arguments = json.loads(submit_tool_call.function.arguments)
                tool_function_output = TOOL_MAP.get(tool_function_name)(*tool_function_arguments)
                tool_outputs.append({
                    "tool_call_id": submit_tool_call.id,
                    "output": tool_function_output,
                })
                
            with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=st.session_state.thread.id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()

# Function to create a new thread in the OpenAI assistant
def create_thread(content, file):
    try:
        response = client.beta.threads.create()
        print("Thread created successfully:", response)
        return response
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        return None
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        return None
    except openai.error.APIError as e:
        print(f"API error: {e}")
        return None
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Function to send a message within an existing thread
def create_message(thread, content, file):
    try:
        attachments = []
        if file is not None:
            attachments.append({
                "file_id": file.id,
                "tools": [{"type": "code_interpreter"}, {"type": "file_search"}]
            })
        response = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
            attachments=attachments
        )
        print("Message sent successfully:", response)
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
    except openai.error.APIError as e:
        print(f"API error: {e}")
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Function to create a download link for a file
def create_file_link(file_name, file_id):
    try:
        content = client.files.content(file_id)
        content_type = content.response.headers["content-type"]
        b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
        link_tag = f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'
        return link_tag
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        return None
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        return None
    except openai.error.APIError as e:
        print(f"API error: {e}")
        return None
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Function to handle the streaming of messages
def run_stream(user_input, file, selected_assistant_id):
    if "thread" not in st.session_state:
        st.session_state.thread = create_thread(user_input, file)
    create_message(st.session_state.thread, user_input, file)

    # Streaming the messages
    with client.beta.threads.runs.stream(
        thread_id=st.session_state.thread.id,
        assistant_id=selected_assistant_id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

# Function to handle uploaded files
def handle_uploaded_file(uploaded_file):
    try:
        file = client.files.create(file=uploaded_file, purpose="assistants")
        print("File uploaded successfully:", file)
        return file
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        return None
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        return None
    except openai.error.APIError as e:
        print(f"API error: {e}")
        return None
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Function to render chat history
def render_chat():
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.markdown(chat["msg"], True)

# Initialize session states
if "tool_call" not in st.session_state:
    st.session_state.tool_calls = []

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "in_progress" not in st.session_state:
    st.session_state.in_progress = False

# Function to disable the form during processing
def disable_form():
    st.session_state.in_progress = True

# Function to handle login logic
def login():
    if st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")

# Function to reset chat
def reset_chat():
    st.session_state.chat_log = []
    st.session_state.in_progress = False

# Function to load the chat screen
def load_chat_screen(assistant_id, assistant_title):
    if enabled_file_upload_message:
        uploaded_file = st.sidebar.file_uploader(
            enabled_file_upload_message,
            type=[
                "txt",
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "csv",
                "json",
                "geojson",
                "xlsx",
                "xls",
            ],
            disabled=st.session_state.in_progress,
        )
    else:
        uploaded_file = None

    st.title(assistant_title if assistant_title else "")
    user_msg = st.chat_input("Message", on_submit=disable_form, disabled=st.session_state.in_progress)
    if user_msg:
        render_chat()
        with st.chat_message("user"):
            st.markdown(user_msg, True)
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})

        file = None
        if uploaded_file is not None:
            file = handle_uploaded_file(uploaded_file)
        run_stream(user_msg, file, assistant_id)
        st.session_state.in_progress = False
        st.session_state.tool_call = None
        st.rerun()

    render_chat()

# Main function to run the Streamlit app
def main():
    # Check if multi-agent settings are defined
    multi_agents = os.getenv("OPENAI_ASSISTANTS", None)
    single_agent_id = os.getenv("ASSISTANT_ID", None)
    single_agent_title = os.getenv("ASSISTANT_TITLE", "Assistants API UI")

    if (
        authentication_required
        and "credentials" in st.secrets
        and authenticator is not None
    ):
        authenticator.login()
        if not st.session_state["authentication_status"]:
            login()
            return
        else:
            authenticator.logout(location="sidebar")

    if multi_agents:
        assistants_json = json.loads(multi_agents)
        assistants_object = {f'{obj["title"]}': obj for obj in assistants_json}
        selected_assistant = st.sidebar.selectbox(
            "Select an assistant profile?",
            list(assistants_object.keys()),
            index=None,
            placeholder="Select an assistant profile...",
            on_change=reset_chat,  # Call the reset function on change
        )
        if selected_assistant:
            load_chat_screen(
                assistants_object[selected_assistant]["id"],
                assistants_object[selected_assistant]["title"],
            )
    elif single_agent_id:
        load_chat_screen(single_agent_id, single_agent_title)
    else:
        st.error("No assistant configurations defined in environment variables.")

if __name__ == "__main__":
    main()
