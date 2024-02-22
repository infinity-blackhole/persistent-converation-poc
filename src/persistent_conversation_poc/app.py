import asyncio
import os
from operator import itemgetter
import hashlib
import streamlit as st
from google.cloud import discoveryengine, storage
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.globals import set_debug, set_verbose
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_message_histories.firestore import (
    FirestoreChatMessageHistory,
)
from langchain_community.retrievers.google_vertex_ai_search import (
    GoogleVertexAISearchRetriever,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableField, ConfigurableFieldSpec
from langchain_google_vertexai.chat_models import ChatVertexAI
from streamlit.runtime.uploaded_file_manager import UploadedFile

condense_question_template = (
    "Given the following conversation and a follow up question, rephrase the "
    "follow up question to be a standalone question, in French language.\n"
    "\n"
    "Chat History:\n"
    "{chat_history}\n"
    "Follow Up Input: {question}\n"
    "Standalone question:"
)
condense_question_prompt = PromptTemplate.from_template(condense_question_template)

combine_docs_template = (
    "Use the following optional pieces of information to fullfil the user's "
    "request in French and in markdown format.\n"
    "---"
    "{context}"
)
combine_docs_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(combine_docs_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


@st.cache_resource
def get_session_history(user_id: str, session_id: str):
    return FirestoreChatMessageHistory(
        collection_name="chat-history",
        user_id=user_id,
        session_id=session_id,
    )


@st.cache_resource
def create_retriever(
    project: str,
    location: str,
    search_location: str,
    search_data_store: str,
    search_filter: str,
):
    llm = ChatVertexAI(
        project=project,
        location=location,
        model_name="gemini-pro",
        max_output_tokens=4096,
        temperature=0.5,
        streaming=True,
        convert_system_message_to_human=True,
    ).configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )
    base_retriever = GoogleVertexAISearchRetriever(
        project_id=project,
        location_id=search_location,
        data_store_id=search_data_store,
        filter=search_filter,
    )
    retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=base_retriever)
    chain = (
        {
            "question": itemgetter("input"),
            "chat_history": itemgetter("history"),
        }
        | ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
        )
        | {
            "history": itemgetter("chat_history"),
            "input": itemgetter("question"),
            "output": itemgetter("answer"),
            "source_documents": itemgetter("source_documents"),
        }
    )
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the session.",
                is_shared=True,
            ),
        ],
    )


@st.cache_resource
def get_retriever(
    project: str,
    location: str,
    search_location: str,
    search_data_store: str,
    user_id: str,
    session_id: str,
):
    return create_retriever(
        project=project,
        location=location,
        search_location=search_location,
        search_data_store=search_data_store,
        search_filter=f'user_id: ANY("{user_id}") AND session_id: ANY("{session_id}")',
    ).with_config(
        configurable={
            "user_id": user_id,
            "session_id": session_id,
        }
    )


def _import_documents(
    data_store_id: str,
    user_id: str,
    session_id: str,
    bucket_uri: str,
    uploaded_files: list[UploadedFile],
    storage_client: storage.Client,
    search_client: discoveryengine.DocumentServiceClient,
):
    uris = [
        _generate_uploaded_file_uri(
            bucket_uri=bucket_uri, session_id=session_id, uploaded_file=uploaded_file
        )
        for uploaded_file in uploaded_files
    ]
    ids = [
        _generate_uploaded_file_id(session_id=session_id, uploaded_file=uploaded_file)
        for uploaded_file in uploaded_files
    ]
    for i, uploaded_file in enumerate(uploaded_files):
        blob = storage.Blob.from_string(uris[i], storage_client)
        blob.upload_from_file(uploaded_file)
    documents = [
        discoveryengine.Document(
            id=ids[i],
            struct_data={
                "user_id": user_id,
                "session_id": session_id,
            },
            content=discoveryengine.Document.Content(
                uri=uris[i],
                mime_type=uploaded_file.type,
            ),
        )
        for i, uploaded_file in enumerate(uploaded_files)
    ]
    request = discoveryengine.ImportDocumentsRequest(
        parent=data_store_id,
        inline_source=discoveryengine.ImportDocumentsRequest.InlineSource(
            documents=documents
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )
    result = search_client.import_documents(request=request).result()
    if len(result.error_samples) != 0:
        st.error(f"Failed to import documents: {result.error_samples}")


def _generate_uploaded_file_id(session_id: str, uploaded_file: UploadedFile):
    return hashlib.md5(
        "{}:{}".format(session_id.encode("utf-8"), uploaded_file.name)
    ).hexdigest()


def _generate_uploaded_file_uri(
    bucket_uri: str, session_id: str, uploaded_file: UploadedFile
):
    return f"{bucket_uri}/{session_id}/{uploaded_file.name}"


def _purge_documents(
    data_store_id: str,
    user_id: str,
    session_id: str,
    bucket_uri: str,
    storage_client: storage.Client,
    search_client: discoveryengine.DocumentServiceClient,
):
    request = discoveryengine.PurgeDocumentsRequest(
        parent=data_store_id,
        # FIX: filter is not yet supported, only wildcard is supported 🥲
        # filter=f"user_id: {user_id} AND session_id: {session_id}",
        filter="*",
    )
    search_client.purge_documents(request=request)
    bucket = storage.Bucket.from_string(bucket_uri, storage_client)
    blobs = storage_client.list_blobs(bucket, prefix=f"{session_id}/")
    for blob in blobs:
        blob.delete(if_generation_match=blob.generation)


def run(
    project: str,
    location: str,
    bucket_uri: str,
    search_location: str,
    search_data_store: str,
    storage_client: storage.Client,
    search_client: discoveryengine.DocumentServiceClient,
):
    st.set_page_config(page_title="Persistent Conversation PoC", page_icon="🐬")

    user_id = st.sidebar.text_input("User", "default")
    session_id = st.sidebar.text_input("Session", "default")
    uploaded_files = st.sidebar.file_uploader("Documents", accept_multiple_files=True)

    if len(uploaded_files) != 0:
        _import_documents(
            data_store_id=search_client.branch_path(
                project=project,
                location=search_location,
                data_store=search_data_store,
                branch="default_branch",
            ),
            user_id=user_id,
            session_id=session_id,
            bucket_uri=bucket_uri,
            uploaded_files=uploaded_files,
            storage_client=storage_client,
            search_client=search_client,
        )

    msgs = get_session_history(user_id=user_id, session_id=session_id)
    chain = get_retriever(
        project=project,
        location=location,
        search_location=search_location,
        search_data_store=search_data_store,
        user_id=user_id,
        session_id=session_id,
    )
    if st.sidebar.button("Clear message history"):
        msgs.clear()
        _purge_documents(
            data_store_id=search_client.branch_path(
                project=project,
                location=search_location,
                data_store=search_data_store,
                branch="default_branch",
            ),
            user_id=user_id,
            session_id=session_id,
            bucket_uri=bucket_uri,
            storage_client=storage_client,
            search_client=search_client,
        )

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            response = chain.invoke({"input": user_query, "history": msgs})
            st.markdown(response["output"])


if __name__ == "__main__":
    if os.environ.get("DEBUG") == "1":
        set_debug(True)
        set_verbose(True)

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    search_location = os.environ.get("SEARCH_LOCATION", "global")
    data_store_id = os.environ["DATA_STORE_ID"]
    bucket_uri = os.environ["BUCKET_URI"]

    storage_client = storage.Client(project=project)
    search_client = discoveryengine.DocumentServiceClient()

    asyncio.set_event_loop(asyncio.new_event_loop())

    run(
        project=project,
        location=location,
        bucket_uri=bucket_uri,
        search_location=search_location,
        search_data_store=data_store_id,
        storage_client=storage_client,
        search_client=search_client,
    )
